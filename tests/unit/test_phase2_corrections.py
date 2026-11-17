from datetime import datetime, timedelta, timezone
import inspect
import json
from types import SimpleNamespace
import numpy as np
import pytest
from pydantic import ValidationError

from aiqon.config.loader import load_config, load_routes, RouteProfile
from aiqon.prediction.features import FEATURE_COLUMNS, build_features
from aiqon.prediction.propagation import (
    estimate_customs_timing, estimate_remaining_duration,
    next_berth_slot_at_or_after, next_customs_opening, environment_value,
)
from aiqon.prediction.replay import replay_snapshots
from aiqon.prediction.service import PredictionService, validate_artifact_runtime_versions
from aiqon.prediction.training import BUNDLE_VERSION, FEATURE_SCHEMA_VERSION, _parse_snapshot, _read_table, load_training_frame
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import Forecast

UTC=timezone.utc

@pytest.fixture(scope="module")
def sample():
    config=load_config("configs/simulation.yaml")
    routes=load_routes(config.routes.config_path).routes
    shipments={r["shipment_id"]:Shipment.model_validate(r) for r in _read_table(".","shipments").to_dict("records")}
    frame=_read_table(".","prediction_snapshots")
    records={}
    for cp in ("departure","port_arrival","customs_cleared"):
        row=frame[frame.checkpoint_type==cp].iloc[0]
        snap=_parse_snapshot(row); records[cp]=(snap,shipments[snap.shipment_id])
    return config,routes,records


@pytest.mark.parametrize(("at","slots","expected"),[
    (datetime(2025,1,1,7,30,tzinfo=UTC),(8,16),datetime(2025,1,1,8,tzinfo=UTC)),
    (datetime(2025,1,1,8,0,tzinfo=UTC),(8,16),datetime(2025,1,1,8,tzinfo=UTC)),
    (datetime(2025,1,1,8,0,1,tzinfo=UTC),(8,16),datetime(2025,1,1,16,tzinfo=UTC)),
    (datetime(2025,1,1,19,0,tzinfo=UTC),(8,16),datetime(2025,1,2,8,tzinfo=UTC)),
    (datetime(2025,1,1,23,59,59,tzinfo=UTC),(0,12),datetime(2025,1,2,0,tzinfo=UTC)),
    (datetime(2025,1,1,7,30,tzinfo=timezone(timedelta(hours=5,minutes=30))),(8,16),datetime(2025,1,1,8,tzinfo=UTC)),
])
def test_berth_slot_is_at_or_after_current_time_and_utc(at,slots,expected):
    actual=next_berth_slot_at_or_after(at,slots)
    assert actual==expected and actual>=at.astimezone(UTC) and actual.tzinfo==UTC


def test_berth_slot_requires_timezone_awareness():
    with pytest.raises(ValueError,match="timezone-aware"):
        next_berth_slot_at_or_after(datetime(2025,1,1,7),(8,16))


def test_predictor_and_simulator_agree_on_exact_slot_boundary():
    from aiqon.simulation.engine import next_berth_slot
    at=datetime(2025,1,1,12,tzinfo=UTC)
    assert next_berth_slot_at_or_after(at,(0,12))==next_berth_slot(at,(0,12))==at


def _make_departure(sample, at, handling_hours, port_slots=(0,12), forecasts=()):
    config,routes,records=sample
    original,shipment=records["departure"]
    route=next(r for r in routes if r.route_id==shipment.route_id).model_copy(update={
        "scheduled_transit_hours":1.0,"berth_slot_hours":port_slots})
    snapshot=original.model_copy(update={"checkpoint_at":at,"observed_event_times":{"departure_at":at},
                                         "forecasts":tuple(forecasts)})
    world=config.world.model_copy(update={"berth_service_base_hours":handling_hours})
    return snapshot,shipment,route,world


def test_customs_calendar_features_include_berth_and_handling_timeline(sample):
    friday=datetime(2025,1,10,10,tzinfo=UTC)
    snapshot,shipment,route,world=_make_departure(sample,friday,1.0)
    complete,start=estimate_customs_timing(snapshot,route,world)
    assert complete==datetime(2025,1,10,13,tzinfo=UTC)
    assert start==complete  # open weekday, before closing
    features=build_features(snapshot,shipment,route,world)
    assert features["expected_customs_open"]==1
    assert features["estimated_customs_wait_hours"]==0


def test_customs_waits_through_weekend_closure_and_changes_with_handling(sample):
    friday=datetime(2025,1,10,10,tzinfo=UTC)
    short=_make_departure(sample,friday,1.0)
    long=_make_departure(sample,friday,8.0)
    short_complete,short_start=estimate_customs_timing(short[0],short[2],short[3])
    long_complete,long_start=estimate_customs_timing(long[0],long[2],long[3])
    assert short_complete<datetime(2025,1,10,18,tzinfo=UTC)
    assert short_start==short_complete
    assert long_complete>datetime(2025,1,10,18,tzinfo=UTC)
    assert long_start==datetime(2025,1,13,8,tzinfo=UTC)
    f=build_features(long[0],long[1],long[2],long[3])
    assert f["expected_customs_open"]==0
    assert f["estimated_customs_start_hour"]==8
    assert f["estimated_customs_wait_hours"]>0


def test_customs_calendar_weekend_and_next_open_rules(sample):
    weekday_route=next(r for r in sample[1] if r.customs.operating_days==("monday","tuesday","wednesday","thursday","friday"))
    saturday_route=next(r for r in sample[1] if "saturday" in r.customs.operating_days)
    saturday=datetime(2025,1,11,10,tzinfo=UTC)
    assert next_customs_opening(saturday,weekday_route)==datetime(2025,1,13,8,tzinfo=UTC)
    assert next_customs_opening(saturday,saturday_route)==saturday
    assert next_customs_opening(datetime(2025,1,10,19,tzinfo=UTC),weekday_route)==datetime(2025,1,13,8,tzinfo=UTC)


def test_incomplete_forecasts_use_deterministic_route_fallback(sample):
    config,routes,records=sample
    snap,shipment=records["port_arrival"]
    route=next(r for r in routes if r.route_id==shipment.route_id)
    a=build_features(snap,shipment,route,config.world)
    b=build_features(snap,shipment,route,config.world)
    assert a==b
    assert a["forecast_port_congestion"] is None or 0 <= a["forecast_port_congestion"] <= 1
    assert "forecast_customs_workload" in FEATURE_COLUMNS


def test_port_arrival_propagation_uses_eligible_congestion_and_slots(sample):
    config,routes,records=sample
    snap,shipment=records["port_arrival"]
    route=next(r for r in routes if r.route_id==shipment.route_id).model_copy(update={
        "berth_slot_hours":(0,12),"berth_capacity":1})
    at=datetime(2025,1,10,10,30,tzinfo=UTC)
    snap=snap.model_copy(update={"checkpoint_at":at,"observed_event_times":{
        "departure_at":at-timedelta(hours=2),"port_arrival_at":at},
        "observed_stage_fields":{**snap.observed_stage_fields,"observed_queue_length":0},
        "current_environment":{**snap.current_environment,"port_congestion":.3},"forecasts":()})
    baseline=estimate_remaining_duration(snap,route,config.world)[1]["berth_wait"]
    observed_busy=snap.model_copy(update={"current_environment":{**snap.current_environment,"port_congestion":.95}})
    observed_wait=estimate_remaining_duration(observed_busy,route,config.world)[1]["berth_wait"]
    assert observed_wait>baseline
    forecast=Forecast(location_id=route.destination_port,forecast_issued_at=at,forecast_valid_at=at+timedelta(hours=1,minutes=30),
                      forecast_variable="port_congestion",forecast_value=1.0)
    informed=snap.model_copy(update={"forecasts":(forecast,)})
    informed_wait=estimate_remaining_duration(informed,route,config.world)[1]["berth_wait"]
    assert informed_wait>baseline
    assert estimate_remaining_duration(informed,route,config.world)==estimate_remaining_duration(informed,route,config.world)
    # Queue and slot waiting are sequential: overlapping initial slot wait is not added a second time.
    queue=snap.model_copy(update={"observed_stage_fields":{**snap.observed_stage_fields,"observed_queue_length":3}})
    queue_wait=estimate_remaining_duration(queue,route,config.world)[1]["berth_wait"]
    assert queue_wait>=baseline
    assert "aiqon.simulation" not in inspect.getsource(estimate_remaining_duration)


def test_future_forecast_is_rejected_by_feature_builder(sample):
    config,routes,records=sample
    snap,shipment=records["departure"]
    route=next(r for r in routes if r.route_id==shipment.route_id)
    forecast=Forecast(location_id=route.destination_port,forecast_issued_at=snap.checkpoint_at,
        forecast_valid_at=snap.checkpoint_at+timedelta(hours=2),forecast_variable="port_congestion",forecast_value=.9)
    late=forecast.model_copy(update={"forecast_issued_at":snap.checkpoint_at+timedelta(seconds=1)})
    with pytest.raises(ValueError,match="issued after checkpoint"):
        build_features(snap.model_copy(update={"forecasts":(late,)}),shipment,route,config.world)


def test_environment_fallback_priority_and_forecast_validity(sample):
    _,routes,records=sample
    snap,shipment=records["port_arrival"]
    route=next(r for r in routes if r.route_id==shipment.route_id)
    event_at=snap.checkpoint_at+timedelta(hours=2,minutes=30)
    observed=snap.current_environment.get("port_congestion")
    forecast=Forecast(location_id=route.destination_port,forecast_issued_at=snap.checkpoint_at,
        forecast_valid_at=event_at,forecast_variable="port_congestion",forecast_value=.987)
    eligible=snap.model_copy(update={"forecasts":(forecast,)})
    assert environment_value(eligible,"port_congestion",event_at,route,route.destination_port)==(.987,"forecast")
    late=forecast.model_copy(update={"forecast_issued_at":snap.checkpoint_at+timedelta(seconds=1)})
    with pytest.raises(ValueError,match="issued after checkpoint"):
        environment_value(snap.model_copy(update={"forecasts":(late,)}),"port_congestion",event_at,route,route.destination_port)
    expired=forecast.model_copy(update={"forecast_valid_at":event_at-timedelta(hours=1,seconds=1)})
    outside=forecast.model_copy(update={"forecast_valid_at":event_at+timedelta(hours=1)})
    for invalid in (expired,outside):
        actual,source=environment_value(snap.model_copy(update={"forecasts":(invalid,)}),"port_congestion",event_at,route,route.destination_port)
        assert source==("observation" if observed is not None else "route_baseline")
    no_observation=snap.model_copy(update={"current_environment":{k:v for k,v in snap.current_environment.items() if k!="port_congestion"},"forecasts":()})
    assert environment_value(no_observation,"port_congestion",event_at,route,route.destination_port)==(route.congestion.baseline,"route_baseline")
    no_forecast=snap.model_copy(update={"forecasts":()})
    if observed is not None:
        assert environment_value(no_forecast,"port_congestion",event_at,route,route.destination_port)==(float(observed),"observation")
    customs,shipment=records["customs_cleared"]
    inland_route=next(r for r in routes if r.route_id==shipment.route_id)
    inland_at=customs.checkpoint_at+timedelta(hours=inland_route.inland.travel_hours)
    observed_inland=customs.current_environment.get("inland_traffic")
    forecast_inland=Forecast(location_id=inland_route.final_destination,forecast_issued_at=customs.checkpoint_at,
        forecast_valid_at=inland_at,forecast_variable="inland_traffic",forecast_value=.876)
    assert environment_value(customs.model_copy(update={"forecasts":(forecast_inland,)}),"inland_traffic",inland_at,
        inland_route,inland_route.final_destination)==(.876,"forecast")
    no_inland_forecast=customs.model_copy(update={"forecasts":()})
    if observed_inland is not None:
        assert environment_value(no_inland_forecast,"inland_traffic",inland_at,inland_route,
                                 inland_route.final_destination)==(float(observed_inland),"observation")
    no_inland_observation=customs.model_copy(update={"forecasts":(),"current_environment":{
        k:v for k,v in customs.current_environment.items() if k!="inland_traffic"}})
    assert environment_value(no_inland_observation,"inland_traffic",inland_at,inland_route,
                             inland_route.final_destination)==(inland_route.inland.traffic_baseline,"route_baseline")


def test_each_environment_variable_uses_its_own_configured_fallback(sample):
    config,routes,records=sample
    departure,shipment=records["departure"]
    departure=departure.model_copy(update={"forecasts":()})
    route=next(r for r in routes if r.route_id==shipment.route_id)
    when=departure.checkpoint_at+timedelta(days=6)
    observed_missing=departure.model_copy(update={"current_environment":{}})
    port=environment_value(observed_missing,"port_congestion",when,route,route.destination_port)
    customs_snap=records["customs_cleared"][0].model_copy(update={"forecasts":(),"current_environment":{}})
    customs=environment_value(customs_snap,"customs_workload",when,route,route.destination_port)
    inland=environment_value(observed_missing,"inland_traffic",when,route,route.final_destination)
    assert port==(route.congestion.baseline,"route_baseline")
    assert customs==(route.customs.workload_baseline,"route_baseline")
    assert inland==(route.inland.traffic_baseline,"route_baseline")
    changed=route.model_copy(update={
        "congestion":route.congestion.model_copy(update={"baseline":.91}),
        "customs":route.customs.model_copy(update={"workload_baseline":.12}),
        "inland":route.inland.model_copy(update={"traffic_baseline":.88})})
    assert environment_value(observed_missing,"port_congestion",when,changed,changed.destination_port)[0] != port[0]
    assert environment_value(customs_snap,"customs_workload",when,changed,changed.destination_port)[0] != customs[0]
    assert environment_value(observed_missing,"inland_traffic",when,changed,changed.final_destination)[0] != inland[0]
    raw_a=estimate_remaining_duration(departure,route,config.world)[0]
    raw_b=estimate_remaining_duration(departure,changed,config.world)[0]
    assert raw_a!=raw_b


def test_variable_baselines_validate_range_and_default_for_old_route_yaml(sample):
    route=sample[1][0]
    legacy=route.model_dump()
    legacy["customs"].pop("workload_baseline")
    legacy["inland"].pop("traffic_baseline")
    parsed=RouteProfile.model_validate(legacy)
    assert parsed.customs.workload_baseline==.50
    assert parsed.inland.traffic_baseline==.45
    invalid=route.model_dump(); invalid["customs"]["workload_baseline"]=1.01
    with pytest.raises(ValidationError): RouteProfile.model_validate(invalid)
    invalid=route.model_dump(); invalid["inland"]["traffic_baseline"]=-.01
    with pytest.raises(ValidationError): RouteProfile.model_validate(invalid)


def test_snapshot_missingness_indicators_and_preprocessing_are_stable(sample):
    config,routes,records=sample
    snap,shipment=records["port_arrival"]
    route=next(r for r in routes if r.route_id==shipment.route_id)
    missing=snap.model_copy(update={"observed_stage_fields":{**snap.observed_stage_fields,"transit_duration_hours":None},
                                    "feature_missingness":{**snap.feature_missingness,"transit_duration_hours":True}})
    features=build_features(missing,shipment,route,config.world)
    assert features["actual_transit_hours__missing"]==1
    from sklearn.dummy import DummyRegressor
    from aiqon.prediction.training import _pipeline
    X,_=load_training_frame(".",config,sample[1])
    pipeline=_pipeline(DummyRegressor())
    pipeline.fit(X.iloc[:300],np.zeros(300))
    encoded=pipeline.named_steps["preprocess"].transform(X.iloc[[0]])
    altered=X.iloc[[0]].copy()
    altered["actual_transit_hours"]=np.nan
    altered["actual_transit_hours__missing"]=1
    altered["origin_condition"]=np.nan
    altered["origin_condition__missing"]=1
    encoded_missing=pipeline.named_steps["preprocess"].transform(altered)
    assert encoded.shape==encoded_missing.shape
    assert np.isfinite(encoded_missing).all()


def test_bundle_runtime_schema_validation_is_explicit(tmp_path,monkeypatch):
    class Model:
        feature_names_in_=np.asarray(FEATURE_COLUMNS,dtype=object)
        def predict(self,X): return np.zeros(len(X))
        def predict_proba(self,X): return np.zeros((len(X),2))
    bundle={"bundle_version":BUNDLE_VERSION,"direct":Model(),"residual":Model(),"classifier":Model(),"routes":(),"world":None,
            "model_version":"test","feature_schema_version":FEATURE_SCHEMA_VERSION,
            "feature_columns":FEATURE_COLUMNS,"python_version":"3.13.1",
            "sklearn_version":"1.6.1","joblib_version":"1.4.2"}
    changed=dict(bundle); changed["feature_columns"]=tuple(reversed(bundle["feature_columns"]))
    with pytest.raises(ValueError,match="Feature name/order mismatch"):
        PredictionService(changed)
    changed=dict(bundle); changed["bundle_version"]=-1
    with pytest.raises(ValueError,match="Unsupported model bundle version"):
        PredictionService(changed)
    changed=dict(bundle); changed["feature_schema_version"]="unknown-schema"
    with pytest.raises(ValueError,match="Feature schema mismatch"):
        PredictionService(changed)
    changed=dict(bundle); changed.pop("classifier")
    with pytest.raises(ValueError,match="missing required components"):
        PredictionService(changed)
    artifact=tmp_path/"phase2_bundle.joblib"
    artifact.write_bytes(b"not a pickle; metadata preflight must reject first")
    import aiqon.prediction.service as service_module
    runtime={"python_version":"3.13.8","sklearn_version":"1.6.1","joblib_version":"1.4.2"}
    metadata={"bundle_version":BUNDLE_VERSION,"feature_schema_version":FEATURE_SCHEMA_VERSION,
              "feature_columns":list(FEATURE_COLUMNS),**{"python_version":"3.13.1","sklearn_version":"1.6.1","joblib_version":"1.4.2"}}
    (tmp_path/"phase2_metadata.json").write_text(json.dumps(metadata),encoding="utf-8")
    monkeypatch.setattr(service_module,"platform",SimpleNamespace(python_version=lambda:runtime["python_version"]))
    monkeypatch.setattr(service_module,"sklearn",SimpleNamespace(__version__=runtime["sklearn_version"]))
    monkeypatch.setattr(service_module,"joblib",SimpleNamespace(__version__=runtime["joblib_version"],load=lambda _:bundle))
    loaded=PredictionService.load(artifact)
    assert loaded.bundle["sklearn_version"]==runtime["sklearn_version"]


@pytest.mark.parametrize(("stored","runtime","message"),[
    ({"python_version":"3.13.1","sklearn_version":"1.6.1","joblib_version":"1.4.2"},{"python_version":"3.13.1","sklearn_version":"1.6.1","joblib_version":"1.4.2"},None),
    ({"python_version":"3.13.1","sklearn_version":"1.6.1","joblib_version":"1.4.2"},{"python_version":"3.13.9","sklearn_version":"1.6.1","joblib_version":"1.4.2"},None),
    ({"python_version":"3.12.8","sklearn_version":"1.6.1","joblib_version":"1.4.2"},{"python_version":"3.13.1","sklearn_version":"1.6.1","joblib_version":"1.4.2"},"Python artifact version mismatch"),
    ({"python_version":"3.13.1","sklearn_version":"1.6.1","joblib_version":"1.4.1"},{"python_version":"3.13.1","sklearn_version":"1.6.1","joblib_version":"1.4.2"},"joblib_version mismatch"),
    ({"python_version":"3.13.1","sklearn_version":"1.6.0","joblib_version":"1.4.2"},{"python_version":"3.13.1","sklearn_version":"1.6.1","joblib_version":"1.4.2"},"sklearn_version mismatch"),
    ({"python_version":"3.13.1","sklearn_version":"1.6.1","joblib_version":"1.4.2"},{"python_version":"3.13.1","sklearn_version":"1.8.0","joblib_version":"1.4.2"},"stored=1.6.1, current=1.8.0"),
])
def test_artifact_compatibility_policy_is_runtime_independent(stored,runtime,message):
    if message is None:
        validate_artifact_runtime_versions(stored,runtime)
    else:
        with pytest.raises(ValueError,match=message):
            validate_artifact_runtime_versions(stored,runtime)


@pytest.mark.parametrize(("key","stored_version","message"),[
    ("python_version","3.12.9","Python artifact version mismatch"),
    ("sklearn_version","1.6.0","sklearn_version mismatch"),
    ("joblib_version","1.4.1","joblib_version mismatch"),
])
def test_loader_rejects_incompatible_runtime_before_deserialization(tmp_path,monkeypatch,key,stored_version,message):
    import aiqon.prediction.service as service_module
    runtime={"python_version":"3.13.8","sklearn_version":"1.6.1","joblib_version":"1.4.2"}
    metadata={"bundle_version":BUNDLE_VERSION,"feature_schema_version":FEATURE_SCHEMA_VERSION,
              "feature_columns":list(FEATURE_COLUMNS),**runtime}
    metadata[key]=stored_version
    path=tmp_path/"phase2_bundle.joblib"
    path.write_bytes(b"not-deserializable")
    (tmp_path/"phase2_metadata.json").write_text(json.dumps(metadata),encoding="utf-8")
    monkeypatch.setattr(service_module,"platform",SimpleNamespace(python_version=lambda:runtime["python_version"]))
    monkeypatch.setattr(service_module,"sklearn",SimpleNamespace(__version__=runtime["sklearn_version"]))
    monkeypatch.setattr(service_module,"joblib",SimpleNamespace(__version__=runtime["joblib_version"],
        load=lambda _:pytest.fail("incompatible artifact must be rejected before deserialization")))
    with pytest.raises(ValueError,match=message):
        PredictionService.load(path)


class _RecordingService:
    def __init__(self): self.seen=[]
    def predict(self,snapshot,shipment,variant):
        self.seen.append(snapshot)
        return snapshot.checkpoint_type


def test_replay_requires_exact_order_and_isolates_each_prediction(sample):
    _,_,records=sample
    snapshots=[records[cp][0] for cp in ("departure","port_arrival","customs_cleared")]
    shipment=records["departure"][1]
    service=_RecordingService()
    assert replay_snapshots(service,shipment,snapshots)==["departure","port_arrival","customs_cleared"]
    assert service.seen==snapshots # one as-of snapshot is passed for each individual prediction
    with pytest.raises(ValueError,match="missing=.*customs_cleared"):
        replay_snapshots(service,shipment,snapshots[:2])
    with pytest.raises(ValueError,match="duplicates=.*departure"):
        replay_snapshots(service,shipment,[*snapshots,snapshots[0]])
    with pytest.raises(ValueError,match="must be ordered"):
        replay_snapshots(service,shipment,[snapshots[1],snapshots[0],snapshots[2]])
    non_chrono=[snapshots[0],snapshots[1].model_copy(update={"checkpoint_at":snapshots[0].checkpoint_at}),snapshots[2]]
    with pytest.raises(ValueError,match="strictly chronological"):
        replay_snapshots(service,shipment,non_chrono)
