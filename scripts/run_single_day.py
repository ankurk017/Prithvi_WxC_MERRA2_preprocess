#!/usr/bin/env python3
"""Process a single day of MERRA-2 data. Set MERRA_HOME and OUTPUT_DIR or edit below."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prithviwxc import process_day, DEFAULT_MERRA_HOME

if __name__ == "__main__":
    output_folder = os.environ.get("OUTPUT_DIR", "/tmp/merra_out")
    home_folder = os.environ.get("MERRA_HOME", DEFAULT_MERRA_HOME)
    date_str = os.environ.get("DATE", "20210205")
    os.makedirs(output_folder, exist_ok=True)
    process_day(date_str, output_folder, home_folder, surface=True, pressure=True)
