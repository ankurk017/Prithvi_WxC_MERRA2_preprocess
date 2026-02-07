
import numpy as np
import xarray as xr
import glob
import logging
from src.aggregation import aggregate_merra
import os
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    filename="merra_extraction_groups.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


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
                "SLP",  # Arlindo suggested not to add this variable since it is redundant and filled with fictious data
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
        "flux": {"tavg1_2d_flx_Nx": ["EFLUX", "HFLUX", "Z0M"]},
        "pcp": {
            "tavg1_2d_flx_Nx": [
                "PRECTOT",
            ]
        },  # not in the "flux" group because its aggregation method is different from others
        "radiation": {"tavg1_2d_rad_Nx": ["LWGEM", "LWGAB", "LWTUP", "SWGNT", "SWTNT"]},
        "static1": {"const_2d_asm_Nx": ["PHIS", "FRLAND"]},
        "static2": {"const_2d_ctm_Nx": ["FROCEAN", "FRACI"]},
        "terrain_selected_levels": [
            72,
            71,
            68,
            63,
            56,
            53,
            51,
            48,
            45,
            44,
            43,
            41,
            39,
            34,
        ],
    }
    return merra_variable_list


def extract_merra_static_vars(
    variable_base_variable="static1",
):
    variables = get_default_merra_variables()
    home_folder = "/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/"

    variable_kw = list(variables[variable_base_variable].keys())[0]
    merra_variables = list(variables[variable_base_variable].values())[0]
    print(variable_kw, merra_variables)
    static_data = xr.open_dataset(home_folder + f"MERRA2.{variable_kw}.00000000.nc4")[
        merra_variables
    ]
    return static_data.drop("time").squeeze()


def extract_merra_vars(
    home_folder="/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/",
    variable_base_variable="pres",
    extract_levels=False,
    extract_time=False,
    year="2022",
    month="01",
    to_netcdf=False,
    aggregation=False,
    aggregation_type="sfc",
    output_folder="/discover/nobackup/sroy14/MERRA2_WFM_training_data/processed_data_by_groups/",
):
    if to_netcdf:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            warnings.warn(
                f"Output folder '{output_folder}' did not exist and has been created.",
                UserWarning,
            )
    variables = get_default_merra_variables()
    home_folder_month = f"{home_folder}/Y{year}/M{month}"

    variable_kw = list(variables[variable_base_variable].keys())[0]
    merra_variables = list(variables[variable_base_variable].values())[0]
    print(variable_kw, merra_variables)

    daily_files = sorted(glob.glob(f"{home_folder_month}/*{variable_kw}.{year}{month}*"))
    
    for files in daily_files:
        merra_monthly = xr.open_dataset(files)[merra_variables].sel(lev=variables["terrain_selected_levels"])
        date_str = str(merra_monthly.time[0].values)[8:10]
        output_file = f"{output_folder}/MERRA_{variable_base_variable}_{year}{month}{date_str}.nc"
        print(f"Started Writing {output_file} ...")
        logging.info(f"Started Writing {output_file} ...")
        merra_monthly.chunk({"time": 1}).to_netcdf(output_file)
        print(f"Ended Writing {output_file} ...")
        logging.info(f"Ended Writing {output_file} ...")

    return None

"""
if __name__ == "__main__":
    to_netcdf = False
    start_year = 1980
    end_year = 2023
    output_folder = (
        "/discover/nobackup/sroy14/MERRA2_WFM_training_data/processed_data_by_groups/"
    )

    for year in range(start_year, end_year + 1):
        year_str = str(year)

        for month in range(1, 12 + 1):
            month_str = f"{month:02d}"

            output_file = f"MERRA2_{year_str}{month_str}.nc"
            print(month)

            logging.info(f"Extracting data for year {year_str} and month {month_str}")

            logging.info(f"Processing pres varibales")
            pres_data = extract_merra_vars(
                variable_base_variable="pres",
                year=year_str,
                month=month_str,
                extract_time=True,
                extract_levels=True,
                aggregation=False,
                to_netcdf=False,
            )
"""
def process_month(year_str, month_str):
    output_file = f"MERRA2_{year_str}{month_str}.nc"
    print(month_str)

    logging.info(f"Extracting data for year {year_str} and month {month_str}")

    logging.info(f"Processing pres variables")
    pres_data = extract_merra_vars(
        variable_base_variable="pres",
        year=year_str,
        month=month_str,
        extract_time=True,
        extract_levels=True,
        aggregation=False,
        to_netcdf=False,
    )

if __name__ == "__main__":
    to_netcdf = False
    start_year = 1980
    end_year = 2023
    output_folder = (
        "/discover/nobackup/sroy14/MERRA2_WFM_training_data/processed_data_by_groups/"
    )

    for year in range(start_year, end_year + 1):
        year_str = str(year)


        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = []
            for month in range(1, 12 + 1):
                month_str = f"{month:02d}"
                future = executor.submit(process_month, year_str, month_str)
                futures.append(future)
    
            #for future in futures:
            #    future.result() 

