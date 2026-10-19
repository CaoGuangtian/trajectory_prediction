"""Small, dependency-free trajectory quality helpers."""

from __future__ import annotations

import math


EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two WGS84 longitude/latitude pairs."""

    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(
        math.radians, (lon1, lat1, lon2, lat2)
    )
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(value)))
