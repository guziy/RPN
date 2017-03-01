

from data import cehq_station
from data.cehq_station import Station


def main():

    """

    :return:
    """


    s = """
    06DA002
    06CD002
    06EA002
    05FE004
    05EF001
    05BN012
    05CK004
    06AG006
    05AK001
    05QB003
    05LM006
    05KJ001
    05MD004
    05JU001
    """


    selected_ids = [tok.strip().upper() for tok in s.split("\n") if tok != ""]


    print(selected_ids)

    stations = cehq_station.load_from_hydat_db(
        province=None,
        selected_ids=selected_ids,
        skip_data_checks=True,
        natural=None
    )


    print(20 * "---")
    for station in stations:
        assert isinstance(station, Station)
        print("{}\t{:.4f}\t{:.4f}\t{}".format(station.id, station.longitude, station.latitude, station.drainage_km2))

        print(20 * "---")


if __name__ == '__main__':
    main()