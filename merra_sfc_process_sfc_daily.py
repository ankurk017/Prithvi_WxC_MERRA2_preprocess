
import numpy as np
import xarray as xr
import glob
import logging
from src.aggregation import aggregate_merra
import os
from datetime import datetime

# Set up logging
logging.basicConfig(filename='merra_extraction_2016.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_default_merra_variables():
    # Variable List : https://docs.google.com/document/d/10chLPRe0b2-EcTrSfkUipiqqqsiRg-cAO9ZJAvLnU3E/edit
    merra_variable_list = {
        "sfc": {
            "inst1_2d_asm_Nx": [
                "U10M",
                "V10M",
                "T2M",
                "QV2M",
                "PS",
                "SLP", # Arlindo suggested not to add this variable since it is redundant and filled with fictious data
                "TS",
                "TQI",
                "TQL",
                "TQV",
            ]
        },
        "pres": {
            "inst3_3d_asm_Nv": [
                "U",
                "V",
                "T",
                "QV",
                "OMEGA",
                "PL",
                "H",
                "CLOUD",
                "QI",
                "QL",
            ]
        },
        "soil": {"tavg1_2d_lnd_Nx": ["GWETROOT", "LAI"]},
        "flux": {"tavg1_2d_flx_Nx": ["EFLUX", "HFLUX",  "Z0M"]},
        "pcp": {"tavg1_2d_flx_Nx": ["PRECTOT", ]}, # not in the "flux" group because its aggregation method is different from others
        "radiation": {"tavg1_2d_rad_Nx": ["LWGEM", "LWGAB", "LWTUP", "SWGNT", "SWTNT"]},
        "static1": {"const_2d_asm_Nx": ["PHIS", "FRLAND"]},
        "static2": {"const_2d_ctm_Nx": ["FROCEAN", "FRACI"]},
        "terrain_selected_levels": [
            72, 71,  68,   63,  56, 53, 51, 48, 45, 44, 43, 41, 39, 34,],
    }
    return merra_variable_list


def check_same_dates(data_array):
    """
    Check if all dates in the given DataArray are the same.
    
    Parameters:
    data_array (xarray.DataArray): DataArray containing datetime64 values.
    
    Returns:
    bool: True if all dates are the same, False otherwise.
    """
    # Extract the datetime values as numpy array
    dates_np = data_array.time.values.astype('datetime64[D]')
    
    # Check if all elements are the same
    dates_are_same = np.all(dates_np == dates_np[0])
    
    return dates_are_same


def extract_merra_static_vars(
    variable_base_variable="static1", 
):

    variables = get_default_merra_variables()
    home_folder = "/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/"

    variable_kw = list(variables[variable_base_variable].keys())[0]
    merra_variables = list(variables[variable_base_variable].values())[0]
    print(variable_kw, merra_variables)
    static_data = xr.open_dataset(home_folder+f'MERRA2.{variable_kw}.00000000.nc4')[merra_variables]
    return static_data.drop('time').squeeze()


def extract_merra_vars(home_folder='/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/',
    variable_base_variable="pres", extract_levels=False, extract_time=False, year='2022', month='01', to_netcdf=False, 
    aggregation=False, aggregation_type='sfc', 
    output_folder='./_tmp'
):

    variables = get_default_merra_variables()
    home_folder_month = f"{home_folder}/Y{year}/M{month}"

    variable_kw = list(variables[variable_base_variable].keys())[0]
    merra_variables = list(variables[variable_base_variable].values())[0]
    print(variable_kw, merra_variables)
    files = sorted(
        glob.glob(f"{home_folder_month}/*{variable_kw}.{year}{month}*"))
    merra_monthly = xr.open_mfdataset(files) 
    merra_monthly_cropped = merra_monthly[merra_variables]
    if extract_time:
        merra_monthly_cropped = merra_monthly_cropped.sel(
            time=merra_monthly["time"].dt.hour.isin(
                [0, 3, 6, 9, 12, 15, 18, 21])
        )
    if extract_levels:
        merra_monthly_cropped = merra_monthly_cropped.sel(
            lev=variables["terrain_selected_levels"]
        )
    if aggregation:
        merra_monthly_cropped = aggregate_merra(merra_monthly_cropped, grouping_type=aggregation_type)

    if to_netcdf:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            warnings.warn(f"Output folder '{output_folder}' did not exist and has been created.", UserWarning)

        output_file = f'{output_folder}/MERRA_{variable_base_variable}_{year}{month}.nc'
        merra_monthly_cropped.to_netcdf(output_file)
    
    return merra_monthly_cropped


def process_day(day_group):
    """
    Process a group for a specific day.
    """
    day = day_group.time[0].dt.strftime('%Y-%m-%d').item()
    
    #sfc_output_file = f'MERRA2_sfc_{year_str}{month_str}{str(day).zfill(2)}.nc'
    sfc_output_file = f"MERRA2_sfc_{str(day).zfill(2).replace('-', '')}.nc"
    print(output_folder + sfc_output_file)
    day_group.to_netcdf(output_folder + sfc_output_file)

import concurrent.futures

if __name__ == "__main__":
    to_netcdf = False
    start_year = 1980
    end_year = 2023
    output_folder='/discover/nobackup/sroy14/MERRA2_WFM_training_data/processed_data_sfc/'

    for year in range(start_year, end_year + 1):
        year_str = str(year)
        for month in range(1, 12 + 1):
            month_str = f"{month:02d}"

            print(month)
            
            logging.info(f"Extracting data for year {year_str} and month {month_str}")

            logging.info(f"Processing sfc files")
            sfc_data = extract_merra_vars(
                variable_base_variable='sfc', year=year_str, month=month_str, extract_time=True, aggregation=False)

            logging.info(f"Processing soil varibales")
            soil_data = extract_merra_vars(
                variable_base_variable='soil', year=year_str, month=month_str, extract_time=False, aggregation=True, aggregation_type='sfc')
            soil_data['GWETROOT'] = soil_data['GWETROOT'].fillna(1)
            soil_data['LAI'] = soil_data['LAI'].fillna(0)

            logging.info(f"Processing flux varibales")
            flux_data = extract_merra_vars(
                variable_base_variable='flux', year=year_str, month=month_str, extract_time=False, aggregation=True, aggregation_type='sfc')

            logging.info(f"Processing rainfall variable")
            pcp_data = extract_merra_vars(
                variable_base_variable='pcp', year=year_str, month=month_str, extract_time=False, aggregation=True, aggregation_type='pcp')

            logging.info(f"Processing radiation variables")
            radiation_data = extract_merra_vars(
                variable_base_variable='radiation', year=year_str, month=month_str, extract_time=False, aggregation=True, aggregation_type='sfc')

            logging.info(f"Processing Static variables")
            static_data1 = extract_merra_static_vars('static1')
            static_data2 = extract_merra_static_vars('static2').isel(time=int(month_str)-1).squeeze()

            logging.info(f"Merging all variables for {year_str}{month_str}")
            sfc_merra = xr.merge((sfc_data, soil_data, flux_data, pcp_data, radiation_data, static_data1, static_data2)).chunk({'time':1})

            grouped_data = sfc_merra.groupby('time.day')

            for day, group in grouped_data:
                sfc_output_file=f'MERRA2_sfc_{year_str}{month_str}{str(day).zfill(2)}.nc'
                print(output_folder+sfc_output_file)
                daily_timesteps = group.time.dt.strftime('%Y-%m-%d') 
                if check_same_dates(daily_timesteps):
                    group.to_netcdf(output_folder+sfc_output_file)
            """

           
            grouped_data = sfc_merra.groupby('time.day')

            with concurrent.futures.ProcessPoolExecutor(max_workers=len(grouped_data)) as executor:
                futures = [executor.submit(process_day, group) for _, group in grouped_data]

                #concurrent.futures.wait(futures)
            """

 
