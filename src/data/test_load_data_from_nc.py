

import numpy as np

def main():
    path = "/snow3/huziy/Daymet_daily_derivatives/daymet_spatial_agg_prcp_10x10/daymet_v3_prcp_1981_na.nc4"

    from netCDF4 import Dataset

    with Dataset(path) as ds:
        data = ds["prcp"][:]
        the_mean = data.mean(axis=0)
        the_mean = np.ma.masked_where(np.isnan(the_mean), the_mean)
        print(the_mean.mean(), the_mean.min(), the_mean.max(), the_mean.std(), the_mean.sum())




if __name__ == '__main__':
    main()