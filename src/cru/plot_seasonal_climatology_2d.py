from collections import OrderedDict




from util.seasons_info import MonthPeriod






def main():

    seasons = OrderedDict([
        ("DJF", MonthPeriod(12, 3)),
        ("MAM", MonthPeriod(3, 3))
    ])





if __name__ == '__main__':
    main()