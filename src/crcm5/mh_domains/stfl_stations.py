import calendar
import re
from datetime import datetime
from numbers import Number, Integral
from pathlib import Path

from crcm5.mh_domains.constants import M3PERS_IN_CFS
from data.cehq_station import Station, set_data_from_pandas_timeseries

import pandas as pd

import io

import matplotlib.pyplot as plt

import numpy as np

img_folder = Path("mh/engage_report/station_data_plots")



# create the folder for images
if not img_folder.exists():
    img_folder.mkdir()

def read_data_file_for_station(station: Station=None, data_file: Path=None) -> pd.DataFrame:


    with data_file.open() as f:
        save = False
        lines_of_interest = []
        for line in f:

            if not save:
                llower = line.strip().lower()
                print(llower)

            if save:
                lines_of_interest.append(line)

            # Parse everything below this line
            if llower.startswith("date") or llower.startswith("year"):
                save = True



    print(len(lines_of_interest))
    instream = io.StringIO("\n".join(lines_of_interest))
    df = pd.read_csv(instream, sep=",", usecols=(0, 1), infer_datetime_format=True, header=None)

    if station.source_data_units is not None:
        if station.source_data_units.lower() == "cfs":
            df[1] *= M3PERS_IN_CFS
        elif station.source_data_units.lower() == "dam3/year":
            ndays_per_year = np.array([365 + int(calendar.isleap(y)) for y in df[0]])
            df[1] *= 1000.0 / (ndays_per_year * 24 * 3600)

    return df



known_date_formats = [
    "%Y-%m-%d", "%Y", "%b %d %Y"
]

def load_stations_from_csv(index_file="mh/obs_data/streamflow_data_organized/station_index.txt",
                           selected_ids=None):


    res = []

    data_dir = Path(index_file).parent

    with open(index_file) as f:

        # skip the first line
        f.readline()

        for line in f:
            if line.strip() == "":
                continue

            toks = re.split("\s+", line)

            st_id = toks[0].strip()

            if (selected_ids is not None) and (st_id not in selected_ids):
                continue


            lon, lat, = [float(tok.strip()) for tok in toks[1:3]]
            st_da = None
            try:
                st_da = float(toks[3].strip())
            except Exception:
                pass

            st_name = " ".join(toks[5:]).split(",")[0]

            s = Station(st_id=st_id, lon=lon, lat=lat, name=st_name)
            s.source_data_units = toks[4].strip()
            s.drainage_km2 = st_da

            print(s)


            ts = read_data_file_for_station(s, data_file=Path(data_dir.joinpath("{}.csv".format(s.id))))

            ts.dropna(inplace=True)





            # if it is date do nothing
            if hasattr(ts.iloc[0, 0], "year"):
                pass
            # convert to dates if it is just a year
            elif isinstance(ts.iloc[0, 0], str):
                date_format = None
                # try different known date formats
                for the_date_format in known_date_formats:
                    try:
                        datetime.strptime(ts.iloc[0, 0], the_date_format)
                        date_format = the_date_format
                    except Exception:
                        pass

                if date_format is None:
                    raise Exception("Do not understand this date format: {}".format(ts.iloc[0, 0]))

                ts[0] = [datetime.strptime(t, date_format) for t in ts.iloc[:, 0]]

            elif float(ts.iloc[0, 0]).is_integer():  # in case we have only year values
                ts[0] = [datetime(int(y), 6, 15) for y in ts.iloc[:, 0]]

            else:
                print(ts.iloc[0, 0])
                raise Exception("Could not convert {} to a date".format(ts.iloc[0, 0]))

            print(ts.head())


            # start - plot for debug

            # fig = plt.figure()
            # ax = plt.gca()
            # ax.set_title(s.id)
            # ts.plot(ax=ax, x=0, y=1)
            # fig.autofmt_xdate()
            #
            # img_file = img_folder.joinpath("{}.png".format(s.id))
            # fig.savefig(str(img_file))

            # end - plot for debug



            set_data_from_pandas_timeseries(ts, s, date_col=0)

            res.append(s)

    return res






if __name__ == '__main__':
    load_stations_from_csv()

