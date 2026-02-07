"""
MERRA-2 variable and path configuration for Prithvi-WxC preprocessing.
"""

# Default MERRA-2 data root (override via environment or caller)
DEFAULT_MERRA_HOME = "/discover/nobackup/projects/gmao/merra2/data/products/MERRA2_all/"


def get_default_merra_variables():
    """Default MERRA-2 variables by category and collection.

    Categories:
        sfc: Surface (inst1_2d_asm_Nx) - U10M, V10M, T2M, QV2M, PS, TS, TQI, TQL, TQV
        pres: Pressure levels (inst3_3d_asm_Nv) - U, V, T, QV, OMEGA
        soil: Land (tavg1_2d_lnd_Nx) - GWETROOT, LAI
        flux: Fluxes (tavg1_2d_flx_Nx) - EFLUX, HFLUX, Z0M
        pcp: Precipitation (tavg1_2d_flx_Nx) - PRECTOT
        radiation: Radiation (tavg1_2d_rad_Nx) - LWGEM, LWGAB, LWTUP, SWGNT, SWTNT
        static1: Constants (const_2d_asm_Nx) - PHIS, FRLAND
        static2: Constants (const_2d_ctm_Nx) - FROCEAN, FRACI
        terrain_selected_levels: Level indices for pressure-level extraction
    """
    return {
        "sfc": {
            "inst1_2d_asm_Nx": [
                "U10M", "V10M", "T2M", "QV2M", "PS",
                "TS", "TQI", "TQL", "TQV",
            ]
        },
        "pres": {
            "inst3_3d_asm_Nv": [
                "U", "V", "T", "QV", "OMEGA",
            ]
        },
        "soil": {"tavg1_2d_lnd_Nx": ["GWETROOT", "LAI"]},
        "flux": {"tavg1_2d_flx_Nx": ["EFLUX", "HFLUX", "Z0M"]},
        "pcp": {"tavg1_2d_flx_Nx": ["PRECTOT"]},
        "radiation": {"tavg1_2d_rad_Nx": ["LWGEM", "LWGAB", "LWTUP", "SWGNT", "SWTNT"]},
        "static1": {"const_2d_asm_Nx": ["PHIS", "FRLAND"]},
        "static2": {"const_2d_ctm_Nx": ["FROCEAN", "FRACI"]},
        "terrain_selected_levels": [44, 50],
    }
