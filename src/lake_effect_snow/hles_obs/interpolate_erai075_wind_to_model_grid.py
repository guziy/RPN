from pathlib import Path

from rpn import level_kinds

from domains.grid_config import gridconfig_from_gemclim_settings_file
from lake_effect_snow.hles_obs.spatial_interpolation import interpolate_wind_components_from_rpn_files


def entry_GL_domain_452x260():
    gc = gridconfig_from_gemclim_settings_file(fpath=Path("config_bundle/GL_coupling_configs/Config_current/gemclim_settings.nml"))
    inp_dir = "/home/huziy/data/big1/Projects/observations/ERAInterim075d_wind"
    out_dir = "/home/huziy/data/big1/Projects/observations/hles_GL_01deg_452x260"
    main(inp_dir, out_dir, target_grid_config=gc)


def main(inp_dir, out_dir, target_grid_config=None, wind_level=1000., wind_level_kind=level_kinds.PRESSURE):
    """
    :param out_dir:
    :param inp_dir: directory containing monthly input files
    """

    interpolate_wind_components_from_rpn_files(data_dir=Path(inp_dir),
                                               out_dir=Path(out_dir),
                                               target_grid_config=target_grid_config,
                                               wind_level=wind_level,
                                               wind_level_kind=wind_level_kind)


if __name__ == '__main__':
    entry_GL_domain_452x260()