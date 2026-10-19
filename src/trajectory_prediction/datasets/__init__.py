"""Dataset readers and dataset-specific schema definitions."""

from .opensky import OpenSkySubset, discover_subset_ids, load_aircraft_metadata

__all__ = ["OpenSkySubset", "discover_subset_ids", "load_aircraft_metadata"]
