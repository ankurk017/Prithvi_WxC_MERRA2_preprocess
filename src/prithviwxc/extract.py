"""
MERRA-2 extraction: variables and static fields.
"""

import glob
import logging
import numpy as np
import xarray as xr

from .config import get_default_merra_variables
from .aggregation import aggregate_merra

logger = logging.getLogger(__name__)


def check_same_dates(data_array: xr.DataArray) -> bool:
    """Return True if all times in the DataArray are on the same calendar day."""
    dates_np = data_array.time.values.astype("datetime64[D]")
    return bool(np.all(dates_np == dates_np[0]))


def extract_merra_static_vars(
    home_folder: str,
    variable_base_variable: str = "static1",
) -> xr.Dataset:
    """Load static MERRA-2 variables (e.g. PHIS, FRLAND or FROCEAN, FRACI)."""
    variables = get_default_merra_variables()
    variable_kw = list(variables[variable_base_variable].keys())[0]
    merra_variables = list(variables[variable_base_variable].values())[0]
    path = f"{home_folder.rstrip('/')}/MERRA2.{variable_kw}.00000000.nc4"
    static_data = xr.open_dataset(path)[merra_variables]
    return static_data.drop_vars("time", errors="ignore").squeeze()


def extract_merra_vars(
    home_folder: str,
    variable_type: str,
    year: str,
    month: str,
    day: str,
    *,
    extract_time: bool = False,
    extract_levels: bool = False,
    aggregation: bool = False,
    aggregation_type: str = "sfc",
) -> xr.Dataset:
    """Extract MERRA-2 variables for one day.

    Parameters
    ----------
    home_folder : str
        Base directory (e.g. .../MERRA2_all/) containing YYYYY/MMM/ structure.
    variable_type : str
        One of sfc, pres, soil, flux, pcp, radiation.
    year, month, day : str
        YYYY, MM, DD.
    extract_time : bool
        If True, keep only hours 0, 3, 6, 9, 12, 15, 18, 21.
    extract_levels : bool
        If True (for pres), subset to terrain_selected_levels.
    aggregation : bool
        If True, aggregate to daily (mean for sfc-type, sum for pcp).
    aggregation_type : str
        Passed to aggregate_merra: 'sfc' or 'pcp'.

    Returns
    -------
    xarray.Dataset
    """
    variables = get_default_merra_variables()
    variable_kw = list(variables[variable_type].keys())[0]
    merra_variables = list(variables[variable_type].values())[0]

    # Support both Y2021/M01 and flat */* structure
    folder = f"{home_folder.rstrip('/')}/Y{year}/M{month}"
    file_pattern = f"{folder}/*{variable_kw}.{year}{month}{day}*"
    files = sorted(glob.glob(file_pattern))
    if not files:
        # Fallback flat pattern
        file_pattern = f"{home_folder.rstrip('/')}/*/*{variable_kw}.{year}{month}{day}*"
        files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for {year}-{month}-{day} (pattern: {file_pattern})")

    data = xr.open_mfdataset(files)[merra_variables]

    if extract_time:
        data = data.sel(time=data["time"].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21]))

    if extract_levels:
        data = data.sel(lev=variables["terrain_selected_levels"])

    if aggregation:
        data = aggregate_merra(data, grouping_type=aggregation_type)

    return data
