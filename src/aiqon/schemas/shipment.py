"""Shipment master record contract."""

from datetime import datetime
from pydantic import Field, model_validator
from .common import TimestampModel


class Shipment(TimestampModel):
    shipment_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    origin: str = Field(min_length=1)
    destination_port: str = Field(min_length=1)
    final_destination: str = Field(min_length=1)
    shipment_type: str = Field(min_length=1)
    volume: float = Field(gt=0)
    scheduled_departure_at: datetime
    scheduled_port_arrival_at: datetime
    scheduled_customs_clearance_at: datetime
    scheduled_delivery_at: datetime
    simulation_seed: int
    scenario_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_schedule_order(self):
        times = [self.scheduled_departure_at, self.scheduled_port_arrival_at,
                 self.scheduled_customs_clearance_at, self.scheduled_delivery_at]
        if times != sorted(times):
            raise ValueError("scheduled timestamps must be chronological")
        return self
