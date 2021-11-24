"""
This is the latest version as of Oct 2019, reworked from the
validation plot script
"""
from collections import OrderedDict

from matplotlib import cm

from lake_effect_snow.data_utils import get_data
from lake_effect_snow.hles_cc import common_params
from lake_effect_snow.hles_cc_validation.validate_hles_and_related_params import calc_biases_and_pvals, plot_biases
from lake_effect_snow import data_utils, default_varname_mappings
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(img_type="png"):
    plt.rcParams["hatch.linewidth"] = 0.2
    season_to_months = OrderedDict([
        ("ND", (11, 12)),
        ("JF", (1, 2)),
        ("MA", (2, 3)),
    ])

    pval_max = 0.1

    known_variables = data_utils.all_known_variables

    # historically used mod (future) and obs (current)
    # to distinguish what should be subtracted
    data_query = {
        #current
        "obs": {
            "beg_year": 1989,
            "end_year": 2010,
            "root_dir": "/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/cc_coupled-GL_CanESM2/cur"
        },

        # future
        "mod": {
            "beg_year": 2079,
            "end_year": 2100,
            "root_dir": "/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/cc_coupled-GL_CanESM2/fut"
        },
    }

    v_to_data = OrderedDict()
    v_to_lons = OrderedDict()
    v_to_lats = OrderedDict()
    for v in known_variables:
        masking = v in [default_varname_mappings.LAKE_ICE_FRACTION]
        v_to_data[v], v_to_lons[v], v_to_lats[v] = get_data(v, season_to_months=season_to_months,
                                                            data_query=data_query, masking=masking)

    v_to_bias, v_to_pvalue, v_to_corr = calc_biases_and_pvals(v_to_data, multipliers=(1, -1))

    # add bias_ prefix to the variable names for the right colormap
    v_to_bias = OrderedDict([(f"bias_{k}", v) for k, v in v_to_bias.items()])
    v_to_pvalue = OrderedDict([(f"bias_{k}", v) for k, v in v_to_pvalue.items()])
    v_to_lons = OrderedDict([(f"bias_{k}", v) for k, v in v_to_lons.items()])
    v_to_lats = OrderedDict([(f"bias_{k}", v) for k, v in v_to_lats.items()])
    vname_to_clevs = OrderedDict([(f"bias_{k}", v) for k, v in common_params.cc_vname_to_clevels.items()])

    display_names_cc = OrderedDict([(f"bias_{k}", v) for k, v in common_params.var_display_names.items()])


    # create custom colormap for the temperature changes
    clevs = common_params.cc_vname_to_clevels[default_varname_mappings.T_AIR_2M]
    cmap = cm.get_cmap("Reds", len(clevs) - 1)
    cmap.set_under("darkblue")

    vname_to_cmap = {
        f"bias_{default_varname_mappings.T_AIR_2M}": cmap
    }


    var_list_to_mask_lakes = [
        f"bias_{default_varname_mappings.HLES_AMOUNT}",
        f"bias_{default_varname_mappings.HLES_FREQUENCY}"]

    plot_biases(v_to_bias, v_to_pvalue, v_to_lons, v_to_lats,
                pval_max=pval_max,
                exp_label="cc_canesm2",
                vname_to_clevs=vname_to_clevs, var_display_names=display_names_cc,
                img_type=img_type, varlist_to_mask_gl=var_list_to_mask_lakes, vname_to_cmap=vname_to_cmap)


if __name__ == '__main__':
    main()
