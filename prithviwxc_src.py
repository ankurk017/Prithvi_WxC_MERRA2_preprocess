import numpy as np
import xarray as xr
import glob
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename="merra_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_default_merra_variables():
    """Define MERRA-2 variables to be extracted.
    
    This function returns a dictionary containing the default MERRA-2 variables 
    organized by category. Each category specifies the data collection and 
    variables to extract.
    
    Categories:
        sfc: Surface variables from inst1_2d_asm_Nx collection
            - U10M, V10M: 10-meter wind components
            - T2M: 2-meter air temperature
            - QV2M: 2-meter specific humidity
            - PS: Surface pressure
            - TS: Surface skin temperature
            - TQI, TQL, TQV: Total ice, liquid water, and water vapor content
            
        pres: Pressure-level variables from inst3_3d_asm_Nv collection
            - U, V: Wind components
            - T: Temperature
            - QV: Specific humidity
            - OMEGA: Vertical velocity
            
        soil: Land surface variables from tavg1_2d_lnd_Nx collection
            - GWETROOT: Root zone soil wetness
            - LAI: Leaf area index
            
        flux: Surface flux variables from tavg1_2d_flx_Nx collection
            - EFLUX: Latent heat flux
            - HFLUX: Sensible heat flux
            - Z0M: Surface roughness
            
        pcp: Precipitation from tavg1_2d_flx_Nx collection
            - PRECTOT: Total precipitation
            
        radiation: Radiation variables from tavg1_2d_rad_Nx collection
            - LWGEM: Longwave ground emission
            - LWGAB: Longwave absorbed by ground
            - LWTUP: Upward longwave flux at top of atmosphere
            - SWGNT: Surface net downward shortwave flux
            - SWTNT: TOA net downward shortwave flux
            
        static1: Time-invariant fields from const_2d_asm_Nx collection
            - PHIS: Surface geopotential
            - FRLAND: Land fraction
            
        static2: Time-varying constants from const_2d_ctm_Nx collection
            - FROCEAN: Ocean fraction
            - FRACI: Ice fraction
            
        terrain_selected_levels: Selected vertical levels for terrain following
            
    Returns:
        dict: Dictionary containing MERRA-2 variable definitions by category
    """
    merra_variable_list = {
        "sfc": {
            "inst1_2d_asm_Nx": [
                "U10M", "V10M", "T2M", "QV2M", "PS",
                "TS", "TQI", "TQL", "TQV",
            ]
        },
        "pres": {
            "inst3_3d_asm_Nv": [
                "U", "V", "T", "QV", "OMEGA",
                # "PL", "H", "CLOUD", "QI", "QL",
            ]
        },
        "soil": {"tavg1_2d_lnd_Nx": ["GWETROOT", "LAI"]},
        "flux": {"tavg1_2d_flx_Nx": ["EFLUX", "HFLUX", "Z0M"]},
        "pcp": {"tavg1_2d_flx_Nx": ["PRECTOT"]},
        "radiation": {"tavg1_2d_rad_Nx": ["LWGEM", "LWGAB", "LWTUP", "SWGNT", "SWTNT"]},
        "static1": {"const_2d_asm_Nx": ["PHIS", "FRLAND"]},
        "static2": {"const_2d_ctm_Nx": ["FROCEAN", "FRACI"]},
        "terrain_selected_levels": [
            # 72, 71, 68, 63, 56, 53, 51, 48, 45, 44, 43, 41, 39, 34,
            44, 50,
        ],
    }
    return merra_variable_list


def check_same_dates(data_array):
    """Check if all dates in the given DataArray are the same.
    
    This function verifies that all timestamps in a given xarray DataArray 
    correspond to the same calendar date. It converts the time values to 
    datetime64[D] format (day resolution) and checks if all dates match
    the first date in the array.
    
    Args:
        data_array (xarray.DataArray): Input DataArray containing a time dimension
            with datetime64 values.
            
    Returns:
        bool: True if all dates are the same, False otherwise.
        
    Example:
        >>> import xarray as xr
        >>> import numpy as np
        >>> times = np.array(['2021-01-01T00:00', '2021-01-01T12:00'], dtype='datetime64')
        >>> da = xr.DataArray(data=[1,2], coords={'time': times}, dims=['time'])
        >>> check_same_dates(da)
        True
    """
    dates_np = data_array.time.values.astype('datetime64[D]')
    return np.all(dates_np == dates_np[0])


def extract_merra_static_vars(variable_base_variable="static1"):
    """Extract static variables from MERRA-2 reanalysis data.
    
    This function loads static variables (like land fraction, terrain height etc.) 
    from MERRA-2 reanalysis dataset. These variables typically don't change over time
    and are stored in special constant files.
    
    Args:
        variable_base_variable (str, optional): Type of static variable to extract.
            Can be either "static1" (for PHIS, FRLAND) or "static2" (for FROCEAN, FRACI).
            Defaults to "static1".
            
    Returns:
        xarray.Dataset: Dataset containing the requested static variables with
            dimensions [lat, lon]. Time dimension is dropped and any singleton
            dimensions are squeezed.
            
    Example:
        >>> static_data = extract_merra_static_vars("static1")
        >>> print(static_data.variables)
        {'PHIS': <xarray.Variable>, 'FRLAND': <xarray.Variable>}
    """
    variables = get_default_merra_variables()
    home_folder = "/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/"

    variable_kw = list(variables[variable_base_variable].keys())[0]
    merra_variables = list(variables[variable_base_variable].values())[0]
    static_data = xr.open_dataset(home_folder + f"MERRA2.{variable_kw}.00000000.nc4")[merra_variables]
    return static_data.drop("time").squeeze()


def process_sfc_data(year_str, month_str, day_str, output_folder):
    """Process and combine MERRA-2 surface-level data for a specific day.
    
    This function extracts and combines various surface-level variables from MERRA-2 
    reanalysis data, including:
    - Surface variables (temperature, pressure, etc.)
    - Soil variables (moisture, temperature)
    - Surface fluxes (heat, moisture)
    - Precipitation variables
    - Radiation variables
    - Static surface properties
    
    The data is merged into a single dataset and saved as a NetCDF file.
    
    Args:
        year_str (str): Year in YYYY format
        month_str (str): Month in MM format 
        day_str (str): Day in DD format
        output_folder (str): Path to folder where output file will be saved
        
    Returns:
        None
        
    Raises:
        FileNotFoundError: If required MERRA-2 input files are not found
        
    Example:
        >>> process_sfc_data('2021', '02', '05', '/path/to/output')
    """
    variables = get_default_merra_variables()
    home_folder = "/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/"
    home_folder = "/rtmp/akumar/MERRA2_v2/dataset/"
    
    # Process surface variables
    sfc_data = extract_merra_vars(home_folder, "sfc", year_str, month_str, day_str, extract_time=True)
    soil_data = extract_merra_vars(home_folder, "soil", year_str, month_str, day_str, aggregation=True)
    flux_data = extract_merra_vars(home_folder, "flux", year_str, month_str, day_str, aggregation=True)
    pcp_data = extract_merra_vars(home_folder, "pcp", year_str, month_str, day_str, aggregation=True)
    radiation_data = extract_merra_vars(home_folder, "radiation", year_str, month_str, day_str, aggregation=True)
    
    # Get static data
    static_data1 = extract_merra_static_vars("static1")
    static_data2 = extract_merra_static_vars("static2").isel(time=int(month_str)-1).squeeze()
    
    # Merge all data
    merged_data = xr.merge([
        sfc_data, soil_data, flux_data, pcp_data, radiation_data,
        static_data1, static_data2
    ])
    
    # Save to netCDF
    output_file = f"{output_folder}/MERRA2_sfc_{year_str}{month_str}{day_str}.nc"
    merged_data.to_netcdf(output_file)
    logging.info(f"Successfully processed surface data for {year_str}-{month_str}-{day_str}")


def process_pres_data(year_str, month_str, day_str, output_folder):
    """Process MERRA-2 pressure level data for a specific day.
    
    This function extracts and processes MERRA-2 pressure level variables for a given date.
    It reads the raw MERRA-2 data files, extracts variables at specific pressure levels and 
    time steps, and saves the processed data as a NetCDF file.
    
    Args:
        year_str (str): Year in YYYY format
        month_str (str): Month in MM format
        day_str (str): Day in DD format 
        output_folder (str): Path to folder where output file will be saved
        
    Returns:
        None
        
    Raises:
        FileNotFoundError: If required MERRA-2 pressure level input files are not found
        
    Example:
        >>> process_pres_data('2021', '02', '05', '/path/to/output')
    """
    variables = get_default_merra_variables()
    home_folder = "/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/"
    home_folder = "/rtmp/akumar/MERRA2_v2/dataset/"
    
    # Process pressure level variables
    pres_data = extract_merra_vars(
        home_folder, "pres", year_str, month_str, day_str,
        extract_time=True, extract_levels=True
    )
    
    # Save to netCDF
    output_file = f"{output_folder}/MERRA2_pres_{year_str}{month_str}{day_str}.nc"
    pres_data.to_netcdf(output_file)
    logging.info(f"Successfully processed pressure level data for {year_str}-{month_str}-{day_str}")
        
def extract_merra_vars(home_folder, variable_type, year, month, day, extract_time=False,
                      extract_levels=False):
    """Extract MERRA-2 variables for a specific day.
    
    This function extracts variables from MERRA-2 data files for a given date. It searches for 
    files matching the specified variable type and date pattern, reads them using xarray, and 
    optionally filters by specific time steps and pressure levels.
    
    Args:
        home_folder (str): Base directory containing MERRA-2 data files
        variable_type (str): Type of variables to extract (e.g., 'pres', 'sfc')
        year (str): Year in YYYY format
        month (str): Month in MM format 
        day (str): Day in DD format
        extract_time (bool, optional): If True, extract data only for specific hours (0,3,6,...,21).
            Defaults to False.
        extract_levels (bool, optional): If True, extract data only for terrain-following pressure 
            levels. Defaults to False.
    
    Returns:
        xarray.Dataset: Dataset containing the extracted MERRA-2 variables
        
    Raises:
        FileNotFoundError: If no files are found matching the date and variable type pattern
        
    Example:
        >>> data = extract_merra_vars('/path/to/merra2', 'pres', '2021', '02', '05', 
        ...                          extract_time=True, extract_levels=True)
    """
    variables = get_default_merra_variables()
    variable_kw = list(variables[variable_type].keys())[0]
    merra_variables = list(variables[variable_type].values())[0]
    
    file_pattern = f"{home_folder}/*/*{variable_kw}.{year}{month}{day}*"
    print(file_pattern)

    files = sorted(glob.glob(file_pattern))
    print(files)
    if not files:
        raise FileNotFoundError(f"No files found for {year}-{month}-{day}")
    
    data = xr.open_mfdataset(files)[merra_variables]
    
    if extract_time:
        data = data.sel(time=data["time"].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21]))
    
    if extract_levels:
        data = data.sel(lev=variables["terrain_selected_levels"])
    
    return data


def process_day(date_str, output_folder):
    """Process both surface and pressure level data for a specific day.
    
    This function processes MERRA-2 data for a given date by splitting the date string
    into components and calling the appropriate processing functions. Currently configured
    to process pressure level data only.
    
    Args:
        date_str (str): Date string in YYYYMMDD format (e.g., '20210205')
        output_folder (str): Path to folder where processed files will be saved
        
    Returns:
        None
        
    Example:
        >>> process_day('20210205', '/path/to/output')
    """
    year_str = date_str[:4]
    month_str = date_str[4:6]
    day_str = date_str[6:]
    
    # process_sfc_data(year_str, month_str, day_str, output_folder)
    process_pres_data(year_str, month_str, day_str, output_folder)

