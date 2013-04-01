__author__ = 'huziy'

import numpy as np


import pandas


def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"

class ModelPoint:
    def __init__(self):
        self.distance_to_station = None
        self.accumulation_area = None
        self.ix = None
        self.jy = None
        #2d mask, which is True for the grid points that enter self by rivers
        self.flow_in_mask = None

        self.data = None
        self.time = None



    def get_daily_climatology_for_complete_years(self, stamp_dates = None, years = None):
        assert stamp_dates is not None
        assert years is not None

        vals = []
        all_data = np.array(self.data)

        year_mask = np.array( [d.year in years for d in self.time] )

        for d in stamp_dates:
            mask = [1 if d.month == x.month and d.day == x.day else 0 for x in self.time ]

            x = np.array(mask) * all_data * year_mask
            vals.append( x[x > 0].mean() )

        return stamp_dates, vals

    def get_daily_climatology_for_complete_years_with_pandas(self, stamp_dates = None, years = None):
        assert stamp_dates is not None
        assert years is not None

        df = pandas.DataFrame(data=self.data, index=self.time, columns=["values",])
        df["year"] = df.index.map(lambda d: d.year)

        df = df[df["year"].isin(years)]
        daily_clim = df.groupby(by=lambda d: (d.month, d.day)).mean()

        print daily_clim.describe()


        vals = [daily_clim.ix[(d.month, d.day), "values"] for d in stamp_dates]



        return stamp_dates, vals