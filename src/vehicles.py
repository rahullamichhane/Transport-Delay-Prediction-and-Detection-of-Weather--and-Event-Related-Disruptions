

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Vehicle:
    """Base class for all public transport vehicles."""

    vehicle_id: str
    route_id: str
    stop_id: str
    route_type: str
    temperature: float
    humidity: float
    weather_condition: str
    event_type: str
    hour: int
    day_of_week: int
    is_peak_hour: int
    is_weekend: int

    def __post_init__(self) -> None:
        type(self).registry[self.vehicle_id] = self

    @property
    def base_delay(self) -> float:
        """Base delay in minutes for this vehicle type (override in subclasses)."""
        raise NotImplementedError

    @property
    def vehicle_class(self) -> str:
        """Human-readable vehicle class name."""
        return self.__class__.__name__

    def weather_impact(self) -> float:
        """Delay impact from weather conditions."""
        impacts = {
            "clear": 0.0,
            "cloudy": 1.0,
            "rain": 5.0,
            "storm": 15.0,
            "snow": 12.0,
            "fog": 4.0,
        }
        return impacts.get(self.weather_condition, 0.0)

    def event_impact(self) -> float:
        """Delay impact from events near the stop."""
        impacts = {
            "none": 0.0,
            "concert": 6.0,
            "sports": 8.0,
            "protest": 10.0,
            "festival": 7.0,
            "parade": 5.0,
        }
        return impacts.get(self.event_type, 0.0)

    def peak_impact(self) -> float:
        """Delay impact during peak hours."""
        return 6.0 if self.is_peak_hour else 0.0

    def weekend_impact(self) -> float:
        """Delay impact on weekends."""
        return 2.0 if self.is_weekend else 0.0

    def winter_impact(self, month: int) -> float:
        """Extra delay during Toronto winter months for snow."""
        if month in (12, 1, 2) and self.weather_condition == "snow":
            return 3.0
        return 0.0

    def compute_delay(self, month: int, noise: float = 0.0) -> float:
        """Compute total predicted delay in minutes."""
        delay = (
            self.base_delay
            + self.weather_impact()
            + self.event_impact()
            + self.peak_impact()
            + self.weekend_impact()
            + self.winter_impact(month)
            + noise
        )
        return max(0.0, round(delay, 2))

    def is_significantly_delayed(self, delay_minutes: float) -> bool:
        """Classify a delay as significant (>= 10 minutes)."""
        return delay_minutes >= 10.0

    def to_record(self, record_id: int, timestamp: str, delay_minutes: float) -> dict:
        """Convert the vehicle state to a streaming record dict."""
        return {
            "record_id": record_id,
            "timestamp": timestamp,
            "vehicle_id": self.vehicle_id,
            "route_id": self.route_id,
            "route_type": self.route_type,
            "vehicle_class": self.vehicle_class,
            "stop_id": self.stop_id,
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "is_peak_hour": self.is_peak_hour,
            "is_weekend": self.is_weekend,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "weather_condition": self.weather_condition,
            "event_type": self.event_type,
            "delay_minutes": delay_minutes,
            "delay_label": int(self.is_significantly_delayed(delay_minutes)),
        }


@dataclass
class Bus(Vehicle):
    """TTC Bus vehicle type."""

    @property
    def base_delay(self) -> float:
        return 5.0


@dataclass
class Streetcar(Vehicle):
    """TTC Streetcar vehicle type (more affected by traffic)."""

    @property
    def base_delay(self) -> float:
        return 7.0


@dataclass
class Subway(Vehicle):
    """TTC Subway vehicle type (least affected by surface conditions)."""

    @property
    def base_delay(self) -> float:
        return 3.0

    def weather_impact(self) -> float:
        """Subway is less affected by surface weather."""
        base = super().weather_impact()
        return base * 0.4


# Per-class registry dicts (class attributes, populated in __post_init__)
Vehicle.registry: Dict[str, Vehicle] = {}
Bus.registry: Dict[str, Bus] = {}
Streetcar.registry: Dict[str, Streetcar] = {}
Subway.registry: Dict[str, Subway] = {}


VEHICLE_CLASS_MAP: Dict[str, type[Vehicle]] = {
    "bus": Bus,
    "streetcar": Streetcar,
    "subway": Subway,
}

VEHICLE_DESCRIPTIONS: Dict[str, str] = {
    "bus": "TTC Bus - surface route, moderate traffic impact",
    "streetcar": "TTC Streetcar - surface route, high traffic impact",
    "subway": "TTC Subway - underground, lowest delay exposure",
}
