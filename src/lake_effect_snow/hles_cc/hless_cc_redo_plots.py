"""
Do all the plots for the HLES paper, mostly for documentation purposes
* figure1: src/lake_effect_snow/hles_cc/plot_monthly_histograms_cc_and_domain.py
* figure2: src/lake_effect_snow/hles_cc_validation/validate_hles_and_related_params_biases_and_obs.py
* figure3: src/lake_effect_snow/hles_cc/plot_cc_2d_all_variables_for_all_periods_001.py
* figure4: src/lake_effect_snow/hles_cc/hles_tt_and_pr_correlations_mean_ice_fraction.py
"""

from lake_effect_snow.hles_cc import plot_monthly_histograms_cc_and_domain as fig1
from lake_effect_snow.hles_cc_validation import validate_hles_and_related_params_biases_and_obs as fig2
from lake_effect_snow.hles_cc import plot_cc_2d_all_variables_for_all_periods as fig3
from lake_effect_snow.hles_cc import hles_tt_and_pr_correlations_mean_ice_fraction as fig4


def main():
    fig1.main(varname="hles_snow")
    fig2.main()  # validation
    fig3.entry_for_cc_canesm2_gl()
    fig4.entry_for_cc_canesm2_gl()


if __name__ == "__main__":
    main()
