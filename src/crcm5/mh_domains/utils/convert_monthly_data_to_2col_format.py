import calendar
from datetime import datetime

from pathlib import Path

from application_properties import main_decorator

import pandas as pd


@main_decorator
def main():
    """
    Iput file is a matrix with rows for years and cols for months
    (except the last one, since it is for the total)
    """
    in_file = "mh/obs_data/Churchill Historic Monthly Apportionable Flow_06EA002.csv.bak.original"

    in_file_p = Path(in_file)

    out_file = in_file_p.parent.joinpath("2_col_{}.csv".format(in_file_p.name[:in_file_p.name.index(".csv")]))


    df_in = pd.read_csv(in_file, skiprows=2)


    dates = []
    values = []

    for c in df_in:
        print(c)

        if c.lower() in ["total", "year"]:
            continue


        for y, v in zip(df_in.year, df_in[c]):
            dates.append(datetime.strptime("{}-{}-15".format(y, c), "%Y-%b-%d"))
            values.append(v)


    s_out = pd.Series(data=values, index=dates)


    # convert units (dam^3/month -> m^3/s)
    factor = s_out.index.map(lambda d: 1000.0 * 1. / (calendar.monthrange(d.year, d.month)[1] * 24 * 3600))
    s_out *= factor


    s_out.sort_index(inplace=True)
    print(s_out.head())
    print(s_out.tail())


    with open(str(out_file), "w") as of:
        of.write("Converted from: \'{}\' \n".format(in_file))
        of.write("-----------------------------------------\n")

    s_out.name = "Monthly flows [m3/s]"
    s_out.to_csv(str(out_file), date_format="%Y-%m-%d", float_format="%.1f", index_label="Date", header=True, mode="a")

    print("Conversion:\n {} -> {} \n Completed!".format(in_file, out_file))



if __name__ == '__main__':
    main()