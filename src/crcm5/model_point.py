from datetime import datetime
from pandas.core.frame import DataFrame

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
        #Data manger for the simulation dataset corresponding the the data of this model point
        #some variables are averaged over upstream points sum(ai * xi)/sum(ai), where ai - are the areas of the gridcells
        self.data_manager = None
        #:type : pandas.DataFrame
        #for holding timeseries data for different variables
        self.climatology_data_frame = None

        self.distance_to_station = None

        self.accumulation_area = None
        self.ix = None
        self.jy = None
        #2d mask, which is True for the grid points that enter self by rivers
        self.flow_in_mask = None

        self.data = None


        self.time = None
        self.common_lake_fractions = None
        self.lake_fraction = None

        #fields for the diagnose points plots
        #Ts - means timeseries
        self.upstreamSurfRunoffTs = None     #m^3/s
        self.upstreamSubSurfRunoffTs = None  #m^3/s
        self.meanUpstreamTempTs = None       #degree Celsium
        self.upstreamSweTs = None            #mm
        self.upstreamPrecipTs = None         # mm/day

        #upstream fields
        self.upstreamSweFieldMonthly = None  #mm
        self.upstreamTempFieldMonthly = None #degrees Celsium



        self.continuous_data_years = None
        self.mean_upstream_lake_fraction = None




    def get_monthly_climatology_for_complete_years(self, varname = "STFL", stamp_dates = None):
        """
        Gets daily climatology using pandas.DataFrame that backs up this point
        """
        monthly_clim = self.climatology_data_frame.groupby(by=lambda d: d.month).mean()
        if stamp_dates is None:
            stamp_dates = [ datetime(1985, m, 15) for m in range(1,13) ]

        vals = [monthly_clim.ix[d.month, varname] for d in stamp_dates]
        return stamp_dates, np.array(vals)






    def get_daily_climatology_for_complete_years(self, varname = "STFL", stamp_dates = None):
        """
        Gets daily climatology using pandas.DataFrame that backs up this point
        """
        assert isinstance(self.climatology_data_frame, DataFrame)
        vals = np.array([self.climatology_data_frame.ix[(d.month, d.day), varname] for d in stamp_dates])
        return stamp_dates, vals

    def get_mask_of_cells_upstream(self):
        return self.data_manager.get_mask_for_cells_upstream(self.ix, self.jy)


    def get_daily_climatology_for_complete_years_with_pandas(self, stamp_dates = None, years = None, input_ts = None):
        assert stamp_dates is not None
        assert years is not None

        if input_ts is None:
            input_ts = self.data

        if years is None:
            years = self.continuous_data_years


        df = pandas.DataFrame(data=input_ts, index=self.time, columns=["values",])
        df["year"] = df.index.map(lambda d: d.year)

        df = df[df["year"].isin(years)]
        daily_clim = df.groupby(by=lambda d: (d.month, d.day)).mean()

        #print daily_clim.describe()


        vals = [daily_clim.ix[(d.month, d.day), "values"] for d in stamp_dates]



        return stamp_dates, vals