"""Toronto (TTC) data generator for live streaming public transport delay records.

Uses the vehicle class hierarchy (Bus, Streetcar, Subway) to generate realistic
Toronto transport, weather, and event data. Each call to stream_records() yields
one record at a time, mimicking a live feed.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
import pandas as pd

from src.vehicles import Bus, Streetcar, Subway, VEHICLE_CLASS_MAP

# Real TTC route numbers / names by type
TTC_ROUTES = {
    "bus": ["Bayview", "Bloor", "Danforth", "Dundas", "Eglinton",
            "Finch", "Kingston", "Leslie", "Sheppard", "Steeles"],
    "streetcar": ["501 Queen", "504 King", "505 Dundas", "506 Carlton",
                  "509 Harbourfront", "510 Spadina", "512 St. Clair"],
    "subway": ["Line 1 Yonge-University", "Line 2 Bloor-Danforth",
               "Line 4 Sheppard"],
}

# Toronto stops / neighbourhoods / landmarks
TTC_STOPS = [
    "Union Station", "Bloor-Yonge", "St. George", "Eglinton", "Finch",
    "Queen's Park", "Kensington Market", "Distillery District", "Harbourfront",
    "St. Clair West", "Dundas West", "High Park", "The Beaches", "Scarborough",
    "North York Centre", "Yorkdale", "Etobicoke Centre", "Don Mills",
    "Lawrence", "Sheppard-Yonge",
]

WEATHER_CONDITIONS = ["clear", "cloudy", "rain", "storm", "snow", "fog"]
EVENT_TYPES = ["none", "concert", "sports", "protest", "festival", "parade"]
PEAK_HOURS = list(range(7, 10)) + list(range(16, 19))


class TransportDataGenerator:
    """Generates a continuous stream of TTC transport delay records using vehicle classes."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _random_vehicle(self) -> Vehicle:
        route_type = random.choice(list(VEHICLE_CLASS_MAP.keys()))
        cls = VEHICLE_CLASS_MAP[route_type]
        route_id = random.choice(TTC_ROUTES[route_type])
        stop_id = random.choice(TTC_STOPS)
        vehicle_id = f"TTC-{route_type}-{random.randint(1000, 9999)}"

        return cls(
            vehicle_id=vehicle_id,
            route_id=route_id,
            stop_id=stop_id,
            route_type=route_type,
            temperature=round(np.random.normal(14, 8), 1),
            humidity=round(np.random.uniform(30, 90), 1),
            weather_condition=random.choice(WEATHER_CONDITIONS),
            event_type=random.choice(EVENT_TYPES),
            hour=random.randint(0, 23),
            day_of_week=random.randint(0, 6),
            is_peak_hour=0,  # set below based on hour
            is_weekend=0,    # set below based on day_of_week
        )

    def _generate_record(self, timestamp: datetime, record_id: int) -> dict:
        vehicle = self._random_vehicle()
        # Recompute derived flags from the actual timestamp
        vehicle.hour = timestamp.hour
        vehicle.day_of_week = timestamp.weekday()
        vehicle.is_peak_hour = int(timestamp.hour in PEAK_HOURS)
        vehicle.is_weekend = int(timestamp.weekday() >= 5)

        noise = np.random.normal(0, 3.0)
        delay_minutes = vehicle.compute_delay(month=timestamp.month, noise=noise)

        return vehicle.to_record(record_id, timestamp.isoformat(), delay_minutes)

    def stream_records(self, n_records: int) -> Iterator[dict]:
        """Yield ``n_records`` streaming records, one at a time (live feed)."""
        start = datetime.now() - timedelta(minutes=n_records)
        for i in range(n_records):
            timestamp = start + timedelta(minutes=i + 1)
            yield self._generate_record(timestamp, i + 1)

    def generate_historical(self, n_records: int = 5000) -> pd.DataFrame:
        """Generate a static historical dataset for model training."""
        records = list(self.stream_records(n_records))
        return pd.DataFrame(records)
