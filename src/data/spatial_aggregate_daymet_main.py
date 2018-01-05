from pathlib import Path

from data.highres_data_manager_xarray import spatial_aggregate_daymet_data


def main():
    source_dir = Path("/snow3/huziy/Daymet_daily")
    # source_dir = Path("/RECH/data/Validation/Daymet/Daily")
    # vname could be: tmin, tmax, tavg, swe, prcp
    spatial_aggregate_daymet_data(
        source_dir=source_dir, dest_dir=Path("/snow3/huziy/Daymet_daily_derivatives"), block_shape=(10, 10),
        vname="swe"
    )



if __name__ == '__main__':
    main()
