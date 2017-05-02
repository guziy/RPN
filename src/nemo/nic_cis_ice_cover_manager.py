import calendar
from collections import defaultdict

import xarray as xr
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

from nemo.nemo_commons import season_to_months_default


class CisNicIceManager(object):

    def __init__(self,  nc_file_path="/RESCUE/skynet3_rech1/huziy/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260/cis_nic_glerl_interpolated_lc.nc",
                 ice_varname="LC", time_varname="time"):
        """
        
        :param nc_file_path: (netcdf file created from NIC, CIS merged data by GLERL) 
        """


        self.ds = xr.open_dataset(nc_file_path)
        self.ice_vname = ice_varname
        self.time_vname = time_varname

        self.time_data = pd.Series(pd.to_datetime(self.ds[self.time_vname][:].values))


        # read the coordinates
        self.lons = self.ds["lon"][:].values
        self.lats = self.ds["lat"][:].values






        # assert isinstance(ice_darr, xr.DataArray)
        #
        # print(ds)
        #
        #
        # time = pd.to_datetime(ds["time"][:].values)
        #
        # print("ice conc. time ranges: {} ... {}".format(time[0], time[-1]))
        #
        # timerange = pd.date_range(start=time[0], end=time[-1], freq="D")
        #
        #
        # # area average time series
        # self.area_avg_ts = ice_darr.where((ice_darr >= 0) & (ice_darr <= 1)).dropna(dim="time", how="all")
        #
        # self.area_avg_ts = self.area_avg_ts.sel(time=timerange, method="nearest").mean(dim="x", skipna=True).mean(dim="y", skipna=True).to_pandas()
        #




    def get_annual_maxima_concentrations(self, mask=None, start_date=None, end_date=None):


        pass


    def get_area_avg_ts(self, mask=None, start_year=-np.Inf, end_year=np.Inf):

        data = self.ds[self.ice_vname]


        # select only data within the year range
        time_sel_vec = np.where(self.time_data.map(lambda d: (d.year >= start_year) and (d.year <= end_year)))[0]

        data = data[time_sel_vec, :, :]

        time_data = pd.to_datetime(data.time.values)

        # region for which the averaging will happen
        i_arr, j_arr = np.where(mask)

        data = np.ma.masked_where(np.isnan(data), data)
        data = data[:, i_arr, j_arr].mean(axis=1)

        return pd.Series(data=data, index=time_data)


    def get_seasonal_mean_climatologies(self, start_year, end_year, season_to_months:dict=None):
        """
        :return {season: (mean, std, nobs, selected_dates)}
        :param start_year: 
        :param end_year: 
        :param season_to_months: 
        """


        result = {}

        for season, months in season_to_months.items():

            season_yearly_data = []
            selected_dates = []

            nobs = 0

            for y in range(start_year, end_year + 1):
                sel_vec = np.where(self.time_data.map(lambda d: (d.month in months) and (d.year == y)))[0]


                if len(sel_vec) == 0:
                    print("Skipping {}, {}".format(season, y))
                    continue

                data = self.ds[self.ice_vname][sel_vec, :, :]
                data = data.where((data <= 1) & (data >= 0)).dropna(dim="time", how="all")
                selected_dates.extend(pd.to_datetime(data.coords["time"].values))

                if 0 in data.shape:
                    print("Skipping {}, {}".format(season, y))
                    continue


                # number of seasonal means used to calculate the climatology
                nobs += 1

                # print("{}, {}: data shape={}".format(season, y, data.shape))
                # print(data.coords["time"])

                data = data.mean(dim="time")


                season_yearly_data.append(data.values)

            season_yearly_data = np.ma.masked_where(np.isnan(season_yearly_data), season_yearly_data)
            if nobs > 0:
                result[season] = (season_yearly_data.mean(axis=0), season_yearly_data.std(axis=0), nobs, selected_dates)
                # print("{}: nobs={}".format(season, nobs))
                # print("selected dates:")
                # print(selected_dates)
                # print("--" * 10)


        return result


    def close(self):
        self.ds.close()




def main():
    manager = CisNicIceManager()

    season_to_months = {calendar.month_name[i]: [i, ] for i in range(1, 13)}

    print(season_to_months)


    # manager.get_seasonal_mean_climatologies(start_year=1980, end_year=2010, season_to_months=season_to_months)



    manager.close()


if __name__ == '__main__':
    main()
