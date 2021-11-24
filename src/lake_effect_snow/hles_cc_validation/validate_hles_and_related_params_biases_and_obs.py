import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from collections import OrderedDict

from lake_effect_snow.data_utils import all_known_variables
from lake_effect_snow.default_varname_mappings import CAO, SNOWFALL_RATE, T_AIR_2M, TOTAL_PREC, LAKE_ICE_FRACTION, \
    HLES_AMOUNT, HLES_FREQUENCY
from lake_effect_snow.hles_cc import common_params
from lake_effect_snow.hles_cc.common_params import bias_vname_to_clevels
from lake_effect_snow.hles_cc_validation.validate_hles_and_related_params import get_data, calc_biases_and_pvals, \
    plot_biases

def main():
    beg_year = 1989
    end_year = 2010
    season_to_months = OrderedDict([
        ("ND", (11, 12)),
       ("JF", (1, 2)),
       ("MA", (2, 3)),
    ])

    known_variables = all_known_variables.copy()
    known_variables.remove(CAO)
    known_variables.remove(SNOWFALL_RATE)
    # known_variables.remove(HLES_AMOUNT)
    known_variables.remove(HLES_FREQUENCY)

    # for the 20200410 version of the validation plot
    known_variables.remove(T_AIR_2M)
    known_variables.remove(TOTAL_PREC)
    # known_variables.remove(LAKE_ICE_FRACTION)

    pval_max = 0.1

    v_to_data = OrderedDict()
    v_to_lons = OrderedDict()
    v_to_lats = OrderedDict()
    for v in known_variables:
        v_to_data[v], v_to_lons[v], v_to_lats[v] = get_data(v,
                                                            season_to_months=season_to_months,
                                                            beg_year=beg_year,
                                                            end_year=end_year)

    v_to_bias, v_to_pvalue, v_to_corr = calc_biases_and_pvals(v_to_data)

    v_to_obs, _, _ = calc_biases_and_pvals(v_to_data, multipliers=[0, 1])

    # update keys for biases to distinguish biases from values
    v_to_bias = OrderedDict([(f"bias_{key}", value) for key, value in v_to_bias.items()])
    v_to_pvalue = OrderedDict([(f"bias_{key}", value) for key, value in v_to_pvalue.items()])
    vname_to_clevs = OrderedDict([(f"bias_{key}", value) for key, value in bias_vname_to_clevels.items()])
    vname_to_clevs.update(common_params.obs_vname_to_clevels)
    v_to_corr = OrderedDict([(f"bias_{key}", value) for key, value in v_to_corr.items()])

    v_to_lons_bias = OrderedDict([(f"bias_{key}", value) for key, value in v_to_lons.items()])
    v_to_lats_bias = OrderedDict([(f"bias_{key}", value) for key, value in v_to_lats.items()])

    v_to_lons.update(v_to_lons_bias)
    v_to_lats.update(v_to_lats_bias)

    # add dummy pvalue of 0 to the obs values
    # v_to_pvalue.update(OrderedDict([(key, {season: v * 0 for season, v in value.items()}) for key, value in v_to_obs.items()]))
    v_to_pvalue.update(OrderedDict([(key, {season: v * 0 for season, v in value.items()}) for key, value in v_to_obs.items()]))

    v_to_obs.update(v_to_bias)
    v_to_bias = v_to_obs

    # Add obs to the labels
    common_params.var_display_names = {k: v if k.startswith("bias") else f"Obs: {v}"
                                       for k, v in common_params.var_display_names.items()}

    plt.rcParams["hatch.linewidth"] = 0.2
    plot_biases(v_to_bias, v_to_pvalue, v_to_lons, v_to_lats, pval_max=pval_max,
                exp_label="validation_canesm2c_excl_tt_and_pr_test_obs_and_biases",
                vname_to_clevs=vname_to_clevs, v_to_corr=v_to_corr,
                var_display_names=common_params.var_display_names, img_type="png")


if __name__ == '__main__':
    main()
