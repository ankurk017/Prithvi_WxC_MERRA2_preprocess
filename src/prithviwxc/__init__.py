"""
Prithvi-WxC MERRA-2 preprocessing package.

Usage:
    from prithviwxc import get_default_merra_variables, process_day, extract_merra_vars
"""

from .config import get_default_merra_variables, DEFAULT_MERRA_HOME
from .aggregation import aggregate_merra
from .extract import (
    check_same_dates,
    extract_merra_static_vars,
    extract_merra_vars,
)
from .process import process_day, process_sfc_data, process_pres_data

__all__ = [
    "get_default_merra_variables",
    "DEFAULT_MERRA_HOME",
    "aggregate_merra",
    "check_same_dates",
    "extract_merra_static_vars",
    "extract_merra_vars",
    "process_day",
    "process_sfc_data",
    "process_pres_data",
]
