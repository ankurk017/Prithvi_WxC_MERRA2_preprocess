import numpy as np
import xarray as xr
import glob
import logging
import os
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor
from src.aggregation import aggregate_merra

# Set up logging
logging.basicConfig(
    filename="merra_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def get_default_merra_variables():
    """Define MERRA-2 variables to be extracted."""
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
                "PL", "H", "CLOUD", "QI", "QL",
            ]
        },
        "soil": {"tavg1_2d_lnd_Nx": ["GWETROOT", "LAI"]},
        "flux": {"tavg1_2d_flx_Nx": ["EFLUX", "HFLUX", "Z0M"]},
        "pcp": {"tavg1_2d_flx_Nx": ["PRECTOT"]},
        "radiation": {"tavg1_2d_rad_Nx": ["LWGEM", "LWGAB", "LWTUP", "SWGNT", "SWTNT"]},
        "static1": {"const_2d_asm_Nx": ["PHIS", "FRLAND"]},
        "static2": {"const_2d_ctm_Nx": ["FROCEAN", "FRACI"]},
        "terrain_selected_levels": [
            72, 71, 68, 63, 56, 53, 51, 48, 45, 44, 43, 41, 39, 34,
        ],
    }
    return merra_variable_list

def check_same_dates(data_array):
    """Check if all dates in the given DataArray are the same."""
    dates_np = data_array.time.values.astype('datetime64[D]')
    return np.all(dates_np == dates_np[0])

def extract_merra_static_vars(variable_base_variable="static1"):
    """Extract static variables from MERRA-2."""
    variables = get_default_merra_variables()
    home_folder = "/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/"

    variable_kw = list(variables[variable_base_variable].keys())[0]
    merra_variables = list(variables[variable_base_variable].values())[0]
    static_data = xr.open_dataset(home_folder + f"MERRA2.{variable_kw}.00000000.nc4")[merra_variables]
    return static_data.drop("time").squeeze()

def process_sfc_data(year_str, month_str, day_str, output_folder):
    """Process surface data for a specific day."""
    try:
        variables = get_default_merra_variables()
        home_folder = "/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/"
        
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
        
    except Exception as e:
        logging.error(f"Error processing surface data for {year_str}-{month_str}-{day_str}: {str(e)}")

def process_pres_data(year_str, month_str, day_str, output_folder):
    """Process pressure level data for a specific day."""
    try:
        variables = get_default_merra_variables()
        home_folder = "/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/"
        
        # Process pressure level variables
        pres_data = extract_merra_vars(
            home_folder, "pres", year_str, month_str, day_str,
            extract_time=True, extract_levels=True
        )
        
        # Save to netCDF
        output_file = f"{output_folder}/MERRA2_pres_{year_str}{month_str}{day_str}.nc"
        pres_data.to_netcdf(output_file)
        logging.info(f"Successfully processed pressure level data for {year_str}-{month_str}-{day_str}")
        
    except Exception as e:
        logging.error(f"Error processing pressure level data for {year_str}-{month_str}-{day_str}: {str(e)}")

def extract_merra_vars(home_folder, variable_type, year, month, day, extract_time=False, 
                      extract_levels=False, aggregation=False):
    """Extract MERRA-2 variables for a specific day."""
    variables = get_default_merra_variables()
    variable_kw = list(variables[variable_type].keys())[0]
    merra_variables = list(variables[variable_type].values())[0]
    
    file_pattern = f"{home_folder}/Y{year}/M{month}/*{variable_kw}.{year}{month}{day}*"
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found for {year}-{month}-{day}")
    
    data = xr.open_mfdataset(files)[merra_variables]
    
    if extract_time:
        data = data.sel(time=data["time"].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21]))
    
    if extract_levels:
        data = data.sel(lev=variables["terrain_selected_levels"])
    
    if aggregation:
        data = aggregate_merra(data, grouping_type='sfc')
    
    return data

def process_day(year, month, day, output_folder):
    """Process both surface and pressure level data for a specific day."""
    year_str = f"{year:04d}"
    month_str = f"{month:02d}"
    day_str = f"{day:02d}"
    
    process_sfc_data(year_str, month_str, day_str, output_folder)
    process_pres_data(year_str, month_str, day_str, output_folder)

if __name__ == "__main__":
    start_year = 1980
    end_year = 2023
    output_folder = "/discover/nobackup/sroy14/MERRA2_WFM_training_data/processed_data/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process data for each year
    for year in range(start_year, end_year + 1):
        logging.info(f"Processing year {year}")
        
        for month in range(1, 13):
            # Get number of days in month
            if month in [4, 6, 9, 11]:
                days_in_month = 30
            elif month == 2:
                days_in_month = 29 if year % 4 == 0 else 28
            else:
                days_in_month = 31
            
            # Process each day in parallel
            with ProcessPoolExecutor(max_workers=12) as executor:
                futures = []
                for day in range(1, days_in_month + 1):
                    future = executor.submit(process_day, year, month, day, output_folder)
                    futures.append(future)
                
                # Wait for all tasks to complete
                for future in futures:
                    future.result() 