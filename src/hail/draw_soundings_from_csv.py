from datetime import datetime
from pathlib import Path

from metpy.plots import SkewT
from metpy import calc

from application_properties import main_decorator
import pandas as pd

import numpy as np


import matplotlib.pyplot as plt

from metpy.units import units

mul_mpers_per_knot = 0.514444

# plot soundings using metpy

@main_decorator
def main():
    img_dir = Path("hail_plots/soundings/")

    if not img_dir.exists():
        img_dir.mkdir(parents=True)


    data_dir = Path("/HOME/huziy/skynet3_rech1/hail/soundings_from_erai/")

    dates = [datetime(1991, 9, 7), datetime(1991, 9, 7, 6), datetime(1991, 9, 7, 12), datetime(1991, 9, 7, 18),
             datetime(1991, 9, 8, 0), datetime(1991, 9, 8, 18)]

    dates.extend([datetime(1991, 9, 6, 0), datetime(1991, 9, 6, 6), datetime(1991, 9, 6, 12), datetime(1991, 9, 6, 18)])

    dates = [datetime(1990, 7, 7), datetime(2010, 7, 12), datetime(1991, 9, 8, 0)]


    def __date_parser(s):
        return pd.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


    tt = pd.read_csv(data_dir.joinpath("TT.csv"), index_col=0, parse_dates=['Time'])
    uu = pd.read_csv(data_dir.joinpath("UU.csv"), index_col=0, parse_dates=['Time'])
    vv = pd.read_csv(data_dir.joinpath("VV.csv"), index_col=0, parse_dates=['Time'])
    hu = pd.read_csv(data_dir.joinpath("HU.csv"), index_col=0, parse_dates=['Time'])


    print(tt.head())
    print([c for c in tt])
    print(list(tt.columns.values))




    temp_perturbation_degc = 2

    for the_date in dates:

        p = np.array([float(c) for c in tt])

        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig)

        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 60)


        tsel = tt.select(lambda d: d == the_date)
        usel = uu.select(lambda d: d == the_date)
        vsel = vv.select(lambda d: d == the_date)
        husel = hu.select(lambda d: d == the_date)


        tvals = tsel.values.mean(axis=0)
        uvals = usel.values.mean(axis=0) * mul_mpers_per_knot
        vvals = vsel.values.mean(axis=0) * mul_mpers_per_knot
        huvals = husel.values.mean(axis=0) * units("g/kg")


        # ignore the lowest level
        all_vars = [p, tvals, uvals, vvals, huvals]

        for i in range(len(all_vars)):
            all_vars[i] = all_vars[i][:-5]

        p, tvals, uvals, vvals, huvals = all_vars


        assert len(p) == len(huvals)

        tdvals = calc.dewpoint(calc.vapor_pressure(p * units.mbar, huvals).to(units.mbar))


        print(tvals, tdvals)
        # Calculate full parcel profile and add to plot as black line
        parcel_profile = calc.parcel_profile(p[::-1] * units.mbar, (tvals[-1] + temp_perturbation_degc) * units.degC, tdvals[-1]).to('degC')
        parcel_profile = parcel_profile[::-1]
        skew.plot(p, parcel_profile, 'k', linewidth=2)



        # Example of coloring area between profiles
        greater = tvals * units.degC >= parcel_profile
        skew.ax.fill_betweenx(p, tvals, parcel_profile, where=greater, facecolor='blue', alpha=0.4)
        skew.ax.fill_betweenx(p, tvals, parcel_profile, where=~greater, facecolor='red', alpha=0.4)



        skew.plot(p, tvals, "r")
        skew.plot(p, tdvals, "g")

        skew.plot_barbs(p, uvals, vvals)

        # Plot a zero degree isotherm
        l = skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)


        # Add the relevant special lines
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()

        plt.title("{} (dT={})".format(the_date, temp_perturbation_degc))

        img_path = "{}_dT={}.png".format(the_date.strftime("%Y%m%d_%H%M%S"), temp_perturbation_degc)
        img_path = img_dir.joinpath(img_path)
        fig.savefig(str(img_path), bbox_inches="tight")

        plt.close(fig)




if __name__ == '__main__':
    main()