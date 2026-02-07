
import numpy as np
import xarray as xr
import glob
import logging
from src.aggregation import aggregate_merra
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    filename="merra_extraction_surface.log",
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
            #            63,
            #            56,
            #            53,
            #            51,
            #            48,
            #            45,
            #            44,
            #            43,
            #            41,
            #            39,
            #            34,
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


def preprocess(ds, variables):
    """keep only the specified variables"""
    return ds[variables]


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
    output_folder="./_tmp",
):
    variables = get_default_merra_variables()
    home_folder_month = f"{home_folder}/Y{year}/M{month}"

    variable_kw = list(variables[variable_base_variable].keys())[0]
    merra_variables = list(variables[variable_base_variable].values())[0]
    print(variable_kw, merra_variables)
    files = sorted(glob.glob(f"{home_folder_month}/*{variable_kw}.{year}{month}*"))

    # merra_monthly = xr.open_mfdataset(files)
    # merra_monthly_cropped = merra_monthly[merra_variables]

    merra_monthly_cropped = xr.open_mfdataset(
        files,
        parallel=True,
        engine="netcdf4",
        chunks={"time": 1},
        autoclose=True,
        preprocess=lambda ds: preprocess(ds, merra_variables),
    )

    if extract_time:
        merra_monthly_cropped = merra_monthly_cropped.sel(
            time=merra_monthly_cropped["time"].dt.hour.isin(
                [0, 3, 6, 9, 12, 15, 18, 21]
            )
        )
    if extract_levels:
        merra_monthly_cropped = merra_monthly_cropped.sel(
            lev=variables["terrain_selected_levels"]
        )
    if aggregation:
        merra_monthly_cropped = aggregate_merra(
            merra_monthly_cropped, grouping_type=aggregation_type
        )

    if to_netcdf:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            warnings.warn(
                f"Output folder '{output_folder}' did not exist and has been created.",
                UserWarning,
            )

        output_file = f"{output_folder}/MERRA_{variable_base_variable}_{year}{month}.nc"
        merra_monthly_cropped.to_netcdf(output_file)

    return merra_monthly_cropped


def merra_write_to_netcdf(year=2011, month=1, data_type="sfc"):
    year_str = str(year)
    month_str = f"{month:02d}"
    print(month)

    logging.info(f"Extracting data for year {year_str} and month {month_str}")
    if data_type == "sfc":
        logging.info(f"Processing sfc files")

        sfc_data = extract_merra_vars(
            variable_base_variable="sfc",
            year=year_str,
            month=month_str,
            extract_time=True,
            aggregation=False,
        )
        logging.info(f"Processing soil varibales")
        soil_data = extract_merra_vars(
            variable_base_variable="soil",
            year=year_str,
            month=month_str,
            extract_time=False,
            aggregation=True,
            aggregation_type="sfc",
        )
        logging.info(f"Processing flux varibales")
        flux_data = extract_merra_vars(
            variable_base_variable="flux",
            year=year_str,
            month=month_str,
            extract_time=False,
            aggregation=True,
            aggregation_type="sfc",
        )
        logging.info(f"Processing rainfall variable")
        pcp_data = extract_merra_vars(
            variable_base_variable="pcp",
            year=year_str,
            month=month_str,
            extract_time=False,
            aggregation=True,
            aggregation_type="pcp",
        )
        logging.info(f"Processing radiation variables")
        radiation_data = extract_merra_vars(
            variable_base_variable="radiation",
            year=year_str,
            month=month_str,
            extract_time=False,
            aggregation=True,
            aggregation_type="sfc",
        )
        logging.info(f"Processing Static variables")
        static_data1 = extract_merra_static_vars("static1")
        static_data2 = (
            extract_merra_static_vars("static2").isel(time=int(month_str) - 1).squeeze()
        )

        logging.info(f"Merging all variables for {year_str}{month_str}")
        #sfc_merra = xr.merge(
        #    (
        #        sfc_data,
        #        soil_data,
        #        flux_data,
        #        pcp_data,
        #        radiation_data,
        #        static_data1,
        #        static_data2,
        #    )
        #).chunk({"time": 1})
        sfc_output_file = f"MERRA2_sfc_{year_str}{month_str}.nc"
        print(f"Started Writing {output_folder+sfc_output_file} ...")
        logging.info(f"Started Writing {output_folder+sfc_output_file} ...")
        sfc_merra.to_netcdf(output_folder + sfc_output_file)
        logging.info(f"Ended Writing {output_folder+sfc_output_file} ")
        print(f"Ended Writing {output_folder+sfc_output_file} ...")

    elif data_type == "pres":
        logging.info(f"Processing pres varibales")
        pres_data = extract_merra_vars(
            variable_base_variable="pres",
            year=year_str,
            month=month_str,
            extract_time=True,
            extract_levels=True,
            aggregation=False,
        )
        prs_merra = pres_data.chunk({"time": 1})
        prs_output_file = f"MERRA2_prs_{year_str}{month_str}.nc"
        print(f"Started Writing {output_folder+prs_output_file} ...")
        logging.info(f"Started Writing {output_folder+prs_output_file} ...")
        prs_merra.to_netcdf(output_folder + prs_output_file)
        logging.info(f"Ended Writing {output_folder+prs_output_file} ")
        print.info(f"Ended Writing {output_folder+prs_output_file} ")


if __name__ == "__main__":
    to_netcdf = False
    start_year = 1980
    end_year = 2020
    output_folder = (
        "/discover/nobackup/sroy14/MERRA2_WFM_training_data/processed_data_sfc/"
    )

    #with ThreadPoolExecutor(max_workers=2) as executor:
    with ProcessPoolExecutor(max_workers=6) as executor:
        for year in range(start_year, end_year + 1):
            for month in range(1, 12 + 1):
                print(f"Year: {year}, Month: {month}")
                executor.submit(merra_write_to_netcdf, year, month, data_type="sfc")
                # merra_write_to_netcdf(year, month, data_type="sfc")

                #executor.submit(merra_write_to_netcdf, year, month, data_type="pres")