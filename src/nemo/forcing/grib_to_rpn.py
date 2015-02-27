from datetime import datetime

__author__ = 'huziy'
import os
import pygrib
from rpn.rpn import RPN, data_types
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Calculate snowrate (mean during the last 3 hours)
# Snow accumulation field is zeroed every 12hr



def main(out_folder=""):
    out_folder = os.path.expanduser(out_folder)

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    path_template = "/RECH/data/Driving_data/Offline/ERA-Interim_0.75/Grib_files/ERA_Interim_0.75d_3h_snowfall_{}.grib"
    # ind = pygrib.index(path, "year", "month")
    # fcst_hour_list = [3, 6, 9, 12]
    fcst_hour_list = [6, 12]

    start_year = 1979
    end_year = 2012

    varname = "SN"

    dateo = datetime(1979, 1, 1)
    npas = 1
    for year in range(start_year, end_year + 1):
        path = path_template.format(year)
        ind = pygrib.index(path, "year", "month")


        for month in range(1, 13):
            fname = "ERAI_0.75_{}h_{}{:02d}.rpn".format(fcst_hour_list[0], year, month)
            fpath = os.path.join(out_folder, fname)
            if os.path.isfile(fpath):

                print "{} -- already exists, delete to regenerate".format(fpath)
                raise Exception("Unfortunately the program cannot be run in parallel for consistency of the data!"
                                " Please delete all previously generated files before proceeding.")



            r_out = RPN(fpath, "w")

            for i, grb_message in enumerate(ind.select(year=year, month=month)):

                if grb_message.startStep not in fcst_hour_list:
                    print "Start step is: ", grb_message.startStep
                    continue

                print grb_message
                # for k in grb_message.keys():
                #    print "{}: {}".format(k, grb_message[k])
                print grb_message.startStep
                print [grb_message[k] for k in ["year", "month", "day", "hour"]]
                print grb_message.validityDate, grb_message.validityTime

                if grb_message.startStep == fcst_hour_list[0]:
                    data_previous = 0

                # print np.sum((grb_message.values < data_previous) & (grb_message.values >= 0)), \
                #       np.sum((grb_message.values >= data_previous) & (data_previous >= 0))
                data = (grb_message.values - data_previous) / float(fcst_hour_list[0] * 3600)

                data[data <= 1.0e-16] = 0

                data_previous = grb_message.values[:, :]


                # add a column at the end
                data_ext = np.zeros((data.shape[0], data.shape[1] + 1))
                data_ext[:, :-1] = data
                data_ext[:, -1] = data[:, 0]
                data = data_ext

                data = np.flipud(data[:, :])

                print data.shape
                r_out.write_2D_field(name = varname,
                                     data = data.transpose(), ip = [0, npas * fcst_hour_list[0], 0],
                                     ig = [0] * 4,
                                     npas = npas, deet=3600 * fcst_hour_list[0], label="ERAI075", dateo = dateo,
                                     grid_type="B", typ_var="P",
                                     nbits = -32, data_type = data_types.compressed_floating_point)
                npas += 1
            r_out.close()

            # plt.figure()
            # plt.plot(np.mean(month_data, axis=1).mean(axis=1), marker="s")
            # plt.show()

        ind.close()

if __name__ == '__main__':
    main(out_folder="/home/huziy/skynet3_rech1/ERAI075_snowfall_rpn/6h")
