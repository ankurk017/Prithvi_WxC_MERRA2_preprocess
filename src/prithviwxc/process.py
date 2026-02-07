"""
High-level processing: surface and pressure-level daily outputs.
"""

import logging
import xarray as xr

from .config import get_default_merra_variables
from .extract import extract_merra_vars, extract_merra_static_vars

logger = logging.getLogger(__name__)


def process_sfc_data(
    year_str: str,
    month_str: str,
    day_str: str,
    output_folder: str,
    home_folder: str,
) -> None:
    """Process and save MERRA-2 surface-level data for one day."""
    sfc_data = extract_merra_vars(
        home_folder, "sfc", year_str, month_str, day_str, extract_time=True
    )
    soil_data = extract_merra_vars(
        home_folder, "soil", year_str, month_str, day_str,
        aggregation=True, aggregation_type="sfc"
    )
    flux_data = extract_merra_vars(
        home_folder, "flux", year_str, month_str, day_str,
        aggregation=True, aggregation_type="sfc"
    )
    pcp_data = extract_merra_vars(
        home_folder, "pcp", year_str, month_str, day_str,
        aggregation=True, aggregation_type="pcp"
    )
    radiation_data = extract_merra_vars(
        home_folder, "radiation", year_str, month_str, day_str,
        aggregation=True, aggregation_type="sfc"
    )
    static_data1 = extract_merra_static_vars(home_folder, "static1")
    static_data2 = (
        extract_merra_static_vars(home_folder, "static2")
        .isel(time=int(month_str) - 1)
        .squeeze()
    )

    merged = xr.merge([
        sfc_data, soil_data, flux_data, pcp_data, radiation_data,
        static_data1, static_data2,
    ])
    out_path = f"{output_folder.rstrip('/')}/MERRA2_sfc_{year_str}{month_str}{day_str}.nc"
    merged.to_netcdf(out_path)
    logger.info("Processed surface data for %s-%s-%s -> %s", year_str, month_str, day_str, out_path)


def process_pres_data(
    year_str: str,
    month_str: str,
    day_str: str,
    output_folder: str,
    home_folder: str,
) -> None:
    """Process and save MERRA-2 pressure-level data for one day."""
    pres_data = extract_merra_vars(
        home_folder, "pres", year_str, month_str, day_str,
        extract_time=True, extract_levels=True
    )
    out_path = f"{output_folder.rstrip('/')}/MERRA2_pres_{year_str}{month_str}{day_str}.nc"
    pres_data.to_netcdf(out_path)
    logger.info("Processed pressure data for %s-%s-%s -> %s", year_str, month_str, day_str, out_path)


def process_day(
    date_str: str,
    output_folder: str,
    home_folder: str,
    surface: bool = True,
    pressure: bool = True,
) -> None:
    """Process one day (YYYYMMDD): surface and/or pressure-level outputs.

    Parameters
    ----------
    date_str : str
        Date as YYYYMMDD (e.g. '20210205').
    output_folder : str
        Directory for output NetCDF files.
    home_folder : str
        MERRA-2 data root.
    surface : bool
        If True, write MERRA2_sfc_YYYYMMDD.nc.
    pressure : bool
        If True, write MERRA2_pres_YYYYMMDD.nc.
    """
    year_str = date_str[:4]
    month_str = date_str[4:6]
    day_str = date_str[6:8]

    if surface:
        process_sfc_data(year_str, month_str, day_str, output_folder, home_folder)
    if pressure:
        process_pres_data(year_str, month_str, day_str, output_folder, home_folder)
