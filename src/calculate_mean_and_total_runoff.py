import os.path
__author__="huziy"
__date__ ="$Aug 9, 2011 2:17:06 PM$"

import application_properties
import rpn
import os
import numpy as np

from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_mean_for_day_of_year(stamp_dates, values):
    surfDict = {}
    for stamp_date, value in zip(stamp_dates, values):
        if stamp_date in surfDict:
            surfDict[stamp_date].append(value)
        else:
            surfDict[stamp_date] = [value]

    sortedDates = sorted(surfDict.keys())
    return sortedDates, [np.mean(surfDict[d]) for d in sortedDates]



def get_mean_time_series(path = 'data/CORDEX/NA'):
    forecastHourToValue = {}

    surface_runoff_name = 'TRAF'
    subsurface_runoff_name = 'TDRA'
    level_tdra = 5
    level_traf = 1
    for theFolder in os.listdir(path):

        folderPath = os.path.join(path, theFolder)
        if not os.path.isdir(folderPath):
            continue

        for theFile in os.listdir(folderPath):
            rpnObj = rpn.RPN(os.path.join(folderPath, theFile))
            # @type rpnObj RPN
            hours = rpnObj.get_current_validity_date()
            print('hours = ', hours)
            surf_runoff = rpnObj.get_first_record_for_name_and_level(surface_runoff_name, level_traf)
            subsurf_runoff = rpnObj.get_first_record_for_name_and_level(subsurface_runoff_name, level_tdra)

            surf_runoff = np.mean(surf_runoff)
            subsurf_runoff = np.mean(subsurf_runoff)

            forecastHourToValue[hours] = [surf_runoff, surf_runoff + subsurf_runoff]
            rpnObj.close()
    return forecastHourToValue


def toStampDates(hours, startDate, stamp_year = 2000):
    stamp_dates = []
    for hour in hours:
        d = startDate + timedelta(hours = hour)
        stamp_dates.append(datetime(stamp_year, d.month, d.day))
    return stamp_dates


def plot_timeseries(hourToValues, outfile = '', title = ''):
    startDate = datetime.strptime('1957090100', '%Y%m%d%H')

    hours = list(hourToValues.keys())
    hours.sort()
    stamp_dates = toStampDates(hours, startDate)

    t1, v1 = get_mean_for_day_of_year(stamp_dates, [ hourToValues[h][0] for h in hours])
    t2, v2 = get_mean_for_day_of_year(stamp_dates, [ hourToValues[h][1] for h in hours])

    plt.figure()
    plt.plot(t1, v1, label = 'surface runoff', lw = 3)
    plt.plot(t2, v2, label = 'total runoff', lw = 3)
    plt.title(title)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b'))
    plt.legend()
    plt.savefig(outfile)





def main():
    withFix = get_mean_time_series(path = 'data/CORDEX/NA_fix')
    withoutFix = get_mean_time_series(path = 'data/CORDEX/NA')
    plot_timeseries(withFix, 'runoff_with_fix.png', 'with fix')
    plot_timeseries(withoutFix, 'runoff_without_fix.png', 'without fix')


if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print("Hello World")
