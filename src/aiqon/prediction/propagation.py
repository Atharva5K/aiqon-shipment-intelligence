"""Transparent, deliberately coarse propagation estimate; independent of simulator internals."""
from datetime import datetime, timedelta, timezone

from aiqon.schemas.common import require_aware_utc


def _hours(a, b):
    return (b - a).total_seconds() / 3600


def next_berth_slot_at_or_after(at: datetime, slot_hours) -> datetime:
    """Match the simulator: return the first configured UTC slot >= an aware timestamp."""
    current = require_aware_utc(at)
    slots = tuple(sorted(set(slot_hours)))
    if not slots:
        raise ValueError("at least one configured berth slot is required")
    if any(not isinstance(hour, int) or not 0 <= hour <= 23 for hour in slots):
        raise ValueError("berth slot hours must be integers from 0 through 23")
    day = current.replace(hour=0, minute=0, second=0, microsecond=0)
    for day_offset in range(2):
        date = day + timedelta(days=day_offset)
        for hour in slots:
            candidate = date.replace(hour=hour)
            if candidate >= current:
                return candidate.astimezone(timezone.utc)
    raise RuntimeError("no eligible berth slot found on current or following day")


def _within_valid_hour(forecast, when):
    event_at=require_aware_utc(when)
    valid_start=require_aware_utc(forecast.forecast_valid_at).replace(minute=0,second=0,microsecond=0)
    return valid_start <= event_at < valid_start+timedelta(hours=1)


def select_eligible_forecast(snapshot, name, when, location_id=None):
    """Select latest issued vintage valid in the UTC hour containing ``when``.

    Forecast records have a single valid timestamp; Phase 1 environment and
    forecasts use hourly UTC buckets, so that timestamp denotes [hour, hour+1h).
    """
    checkpoint=require_aware_utc(snapshot.checkpoint_at)
    for forecast in snapshot.forecasts:
        if require_aware_utc(forecast.forecast_issued_at)>checkpoint:
            raise ValueError("forecast issued after checkpoint")
    candidates=[f for f in snapshot.forecasts if f.forecast_variable==name
                and (location_id is None or f.location_id==location_id)
                and require_aware_utc(f.forecast_issued_at)<=checkpoint
                and _within_valid_hour(f,when)]
    if not candidates:
        return None
    return max(candidates,key=lambda f:(f.forecast_issued_at,f.forecast_valid_at))


_OBSERVATION_KEYS={"port_congestion":"port_congestion","customs_workload":"customs_workload",
                   "inland_traffic":"inland_traffic"}

def _environment_baseline(name,route):
    return {"port_congestion":route.congestion.baseline,
            "customs_workload":route.customs.workload_baseline,
            "inland_traffic":route.inland.traffic_baseline}[name]

def environment_value(snapshot, name, when, route, location_id=None, *, forecast_trace=None):
    """Resolve an as-of value and optionally expose the exact selected vintage."""
    forecast=select_eligible_forecast(snapshot,name,when,location_id)
    if forecast is not None:
        if forecast_trace is not None:
            forecast_trace[name] = {"forecast": forecast, "requested_at": require_aware_utc(when),
                                    "location_id": location_id, "source": "forecast"}
        return float(forecast.forecast_value),"forecast"
    observed=snapshot.current_environment.get(_OBSERVATION_KEYS[name])
    if observed is not None:
        if forecast_trace is not None:
            forecast_trace[name] = {"forecast": None, "requested_at": require_aware_utc(when),
                                    "location_id": location_id, "source": "observation"}
        return float(observed),"observation"
    if forecast_trace is not None:
        forecast_trace[name] = {"forecast": None, "requested_at": require_aware_utc(when),
                                "location_id": location_id, "source": "route_baseline"}
    return float(_environment_baseline(name,route)),"route_baseline"


def _berth_congestion(snapshot, arrival, route, world, queue_length, *, forecast_trace=None):
    preliminary,_,_,_=estimate_berth_timing(arrival,route,world,queue_length,route.congestion.baseline)
    estimated_service=arrival+timedelta(hours=preliminary)
    return environment_value(snapshot,"port_congestion",estimated_service,route,route.destination_port,
                             forecast_trace=forecast_trace)[0]


def next_customs_opening(at, route):
    """Return the next instant customs can accept processing under its configured calendar."""
    at = require_aware_utc(at)
    days = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    t = at
    for _ in range(9):
        if days[t.weekday()] in route.customs.operating_days:
            opening = t.replace(hour=route.customs.opening_hour_utc, minute=0, second=0, microsecond=0)
            closing = t.replace(hour=route.customs.closing_hour_utc, minute=0, second=0, microsecond=0)
            if t < opening:
                return opening
            if t < closing:
                return t
        t = (t + timedelta(days=1)).replace(hour=route.customs.opening_hour_utc, minute=0, second=0, microsecond=0)
    return t


def estimate_berth_timing(arrival_at, route, world, queue_length=0.0, congestion=None):
    """Estimate queue and residual slot waits without counting overlapping wait twice.

    Queue time is a deterministic estimate from observed queue length and/or the
    eligible forecast; when absent, configured route congestion baseline is used.
    The slot is then selected after that queue-ready time;
    only the remaining wait from queue-ready to that slot is added.
    """
    arrival = require_aware_utc(arrival_at)
    forecast = route.congestion.baseline if congestion is None else float(congestion)
    observed_queue_wait = max(0.0, float(queue_length or 0.0)) * world.berth_service_base_hours / route.berth_capacity
    forecast_queue_wait = max(0.0, forecast - 0.25) * world.berth_service_base_hours / route.berth_capacity
    queue_wait = max(observed_queue_wait, forecast_queue_wait)
    queue_ready = arrival + timedelta(hours=queue_wait)
    slot_at = next_berth_slot_at_or_after(queue_ready, route.berth_slot_hours)
    slot_wait = max(0.0, _hours(queue_ready, slot_at))
    return queue_wait + slot_wait, queue_wait, slot_wait, slot_at


def estimate_customs_start(snapshot, route, world):
    """Estimate first customs processing time after future port service and calendar wait.

    Departure uses configured transit; port arrival uses its triggering arrival
    time. Queue, slot, and handling time use only the snapshot/configuration and
    eligible forecasts. Customs-cleared has no future customs start to estimate.
    """
    return estimate_customs_timing(snapshot, route, world)[1]


def estimate_customs_timing(snapshot, route, world):
    """Return (port complete estimate, first customs opening); both None post-clearance."""
    if snapshot.checkpoint_type == "customs_cleared":
        return None, None
    if snapshot.checkpoint_type == "departure":
        port_arrival = snapshot.checkpoint_at + timedelta(hours=route.scheduled_transit_hours)
        queue_length = 0.0
    else:
        port_arrival = snapshot.checkpoint_at
        queue_length = snapshot.observed_stage_fields.get("observed_queue_length", 0) or 0
    congestion = _berth_congestion(snapshot,port_arrival,route,world,queue_length)
    wait, _, _, slot_at = estimate_berth_timing(port_arrival, route, world, queue_length, congestion)
    # Use the actual estimated start from the berth timeline, not arrival time.
    berth_service_start = port_arrival + timedelta(hours=wait)
    if berth_service_start < slot_at:
        berth_service_start = slot_at
    port_complete = berth_service_start + timedelta(hours=world.berth_service_base_hours)
    return port_complete, next_customs_opening(port_complete, route)


def _next_gate(ready, route):
    cutoff = ready.replace(hour=route.inland.gate_cutoff_hour_utc, minute=0, second=0, microsecond=0)
    if ready <= cutoff:
        return ready
    return (ready + timedelta(days=1)).replace(hour=route.inland.gate_cutoff_hour_utc, minute=0, second=0, microsecond=0)


def estimate_remaining_duration(snapshot, route, world, *, forecast_trace=None):
    """Return raw positive-components sum and transparent component breakdown."""
    now, cp = snapshot.checkpoint_at, snapshot.checkpoint_type
    comps = dict(transit=0., berth_wait=0., port_handling=0., customs_wait=0., customs_processing=0., inland_wait=0., inland_travel=0.)
    current = now
    if cp == "departure":
        comps["transit"] = route.scheduled_transit_hours
        current += timedelta(hours=comps["transit"])
        congestion = _berth_congestion(snapshot,current,route,world,0.0,forecast_trace=forecast_trace)
        wait, _, _, slot = estimate_berth_timing(current, route, world, congestion=congestion)
        comps["berth_wait"] = wait
        current += timedelta(hours=wait)
        if current < slot:
            current = slot
    elif cp == "port_arrival":
        q = snapshot.observed_stage_fields.get("observed_queue_length", 0) or 0
        congestion = _berth_congestion(snapshot,now,route,world,q,forecast_trace=forecast_trace)
        comps["berth_wait"], _, _, slot = estimate_berth_timing(now, route, world, q, congestion)
        current += timedelta(hours=comps["berth_wait"])
        if current < slot:
            current = slot
    else:
        # Clearance has completed; customs stages are omitted.
        pass
    if cp != "customs_cleared":
        comps["port_handling"] = world.berth_service_base_hours
        current += timedelta(hours=comps["port_handling"])
        opened = next_customs_opening(current, route)
        comps["customs_wait"] = max(0., _hours(current, opened))
        current = opened
        workload,_ = environment_value(snapshot,"customs_workload",current+timedelta(hours=route.customs.base_processing_hours),
                                       route,route.destination_port,forecast_trace=forecast_trace)
        comps["customs_processing"] = route.customs.base_processing_hours * max(.5, 1 + .35 * (workload - .5))
        current += timedelta(hours=comps["customs_processing"])
    traffic,_ = environment_value(snapshot,"inland_traffic",current+timedelta(hours=route.inland.travel_hours),
                                  route,route.final_destination,forecast_trace=forecast_trace)
    comps["inland_travel"] = route.inland.travel_hours * max(.65, 1 + .25 * (traffic - .5))
    ready = current + timedelta(hours=comps["inland_travel"])
    gate = _next_gate(ready, route)
    comps["inland_wait"] = max(0., _hours(ready, gate))
    raw = sum(comps.values())
    return raw, comps

