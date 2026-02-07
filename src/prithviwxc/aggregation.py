"""
Time aggregation for MERRA-2 data (e.g. sub-daily to daily).
"""

import xarray as xr


def aggregate_merra(data: xr.Dataset, grouping_type: str = "sfc") -> xr.Dataset:
    """Aggregate MERRA-2 data in time.

    - For 'sfc': daily mean (typical for fluxes, soil, radiation).
    - For 'pcp': daily sum (precipitation).

    Parameters
    ----------
    data : xarray.Dataset
        Dataset with a 'time' dimension.
    grouping_type : str
        One of 'sfc' (daily mean) or 'pcp' (daily sum).

    Returns
    -------
    xarray.Dataset
        Time-aggregated dataset.
    """
    if grouping_type == "pcp":
        return data.resample(time="1D").sum(skipna=True)
    # default: sfc-style daily mean
    return data.resample(time="1D").mean(skipna=True)
