
from xarray import DataArray
from datetime import datetime, timedelta


def main():
    N = 48
    data = range(N)

    dates = [datetime(2001, 1, 1, 15) + timedelta(hours=i) for i in range(N)]

    # time variable is called the same as time dimension: works
    scheme1 = {
        "dims": "t",
        "coords": {"t": {"dims": "t", "data": dates}},
        "data": data
    }

    # time variable is called differently from the time dimension: does not work
    scheme2 = {
        "dims": "t",
        "coords": {"time": {"dims": "t", "data": dates}},
        "data": data
    }




    a1 = DataArray.from_dict(scheme1)
    print(a1)

    a1_daily = a1.resample("D", "t")
    print(a1_daily)
    assert isinstance(a1_daily, DataArray)

    print([t for t in a1_daily.coords["t"].values])


    print("=======================")



    a2 = DataArray.from_dict(scheme2)
    print(a2)
    print(a2.resample("D", "time"))






if __name__ == '__main__':
    main()
