"""
This script preprocesses MERRA-2 weather data for a specific date.

The script creates an output directory and processes weather data for a given date
using the process_day function from prithviwxc_src. The processed data is saved
to the specified output folder.
"""

from prithviwxc_src import process_day
import os

if __name__ == "__main__":
    # Define output directory path
    output_folder = '/rtmp/akumar/delete_folder_Prithvi_MERRA_data_preprocessing/'
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Date to process in YYYYMMDD format
    date_str = '20210205'

    # Process the data for specified date
    print(f"Processing date: {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
    process_day(date_str, output_folder)
