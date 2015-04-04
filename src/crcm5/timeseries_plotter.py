import pickle
from matplotlib.font_manager import FontProperties
from scipy.spatial.kdtree import KDTree
from crcm5.model_data import Crcm5ModelDataManager
from data.cell_manager import CellManager
from permafrost.active_layer_thickness import CRCMDataManager
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np

import matplotlib.pyplot as plt


class TimeseriesPlotter:

    def __init__(self, name_to_date_to_field, basemap, lons2d, lats2d,
                 ax = None, cell_area = None, cell_manager = None, data_manager = None):


        self.gwdi_mean_field = None
        self.traf_mean_field = None
        self.tdra_mean_field = None
        self.upin_mean_field = None


        self.basemap = basemap
        self.date_to_stfl_field = name_to_date_to_field["STFL"]
        self.date_to_traf_field = name_to_date_to_field["TRAF"]
        self.date_to_tdra_field = name_to_date_to_field["TDRA"]
        self.date_to_pr_field = name_to_date_to_field["PR"]
        self.date_to_swe_field = name_to_date_to_field["I5"]
        self.date_to_swst_field = name_to_date_to_field["SWST"]
        #self.date_to_imav_field = name_to_date_to_field["IMAV"]

        self.acc_area_km2 = name_to_date_to_field["FACC"]
        #:type : CellManager
        self.cell_manager = cell_manager
        assert isinstance(self.cell_manager, CellManager)
        self.cell_area = cell_area

        x, y, z = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
        self.kdtree = KDTree(list(zip(x,y,z)))
        ax.figure.canvas.mpl_connect("button_press_event", self)
        self.ax = ax
        self.lons2d = lons2d
        self.lats2d = lats2d
        self.data_manager = data_manager
        assert isinstance(self.data_manager, Crcm5ModelDataManager)


        self.x_pr, self.y_pr = basemap(lons2d, lats2d)

        self.lons_flat = lons2d.flatten()
        self.lats_flat = lats2d.flatten()
        self.dates_sorted = list( sorted(list(name_to_date_to_field.items())[0][1].keys()) )


        self.counter = 0

        self.date_to_swsr_field = name_to_date_to_field["SWSR"]
        self.date_to_swsl_field = name_to_date_to_field["SWSL"]
        #self.date_to_gwdi_field = name_to_date_to_field["GWDI"]
        self.date_to_upin_field = name_to_date_to_field["UPIN"]


        #static fields
        self.slope = name_to_date_to_field["SLOP"]
        self.channel_length = name_to_date_to_field["LENG"]
        self.lake_outlet = name_to_date_to_field["LKOU"]

        self.coef_bf = -np.ones(self.slope.shape)

        good_points = self.slope >= 0
        self.coef_bf[good_points] = (self.slope[good_points]) ** 0.5 / ((self.channel_length[good_points]) ** (4.0/3.0) * data_manager.manning_bf[good_points] )




    def _get_closest_ij(self, event):
        lon, lat = self.basemap(event.xdata, event.ydata, inverse = True)

        x0, y0, z0 = lat_lon.lon_lat_to_cartesian(lon, lat)

        dist, i = self.kdtree.query((x0,y0,z0))

        lon0, lat0 = self.lons_flat[i], self.lats_flat[i]

        ind = np.where((self.lons2d == lon0) & (self.lats2d == lat0))


        ix = ind[0][0]
        jy = ind[1][0]
        return ix, jy



    def __call__(self,event):
        if event.button != 3:
            return
        i,j = self._get_closest_ij( event )

        vals = [
            self.date_to_stfl_field[d][i,j] for d in self.dates_sorted
        ]
        plt.figure()
        plt.plot(self.dates_sorted, vals, label = "STFL")


        mask = self.cell_manager.get_mask_of_cells_connected_with(self.cell_manager.cells[i][j])


        print("sum(mask) = ", np.sum(mask))

        vals1 = [
           np.sum( self.date_to_traf_field[d][mask == 1] ) for d in self.dates_sorted
        ]
        plt.plot(self.dates_sorted, vals1, label = "TRAF")

        vals2 = [
           np.sum( self.date_to_tdra_field[d][mask == 1] ) for d in self.dates_sorted
        ]
        plt.plot(self.dates_sorted, vals2, label = "TDRA")

        vals3 = [
           np.sum( self.date_to_pr_field[d][mask == 1] ) for d in self.dates_sorted
        ]
        plt.plot(self.dates_sorted, vals3, label = "PR")


        #vals4 = [
        #   np.sum( self.date_to_gwdi_field[d][mask == 1] ) for d in self.dates_sorted
        #]
        #plt.plot(self.dates_sorted, vals4, label = "GWDI")

        vals5 = [
           np.sum( self.date_to_upin_field[d][i,j] ) for d in self.dates_sorted
        ]
        plt.plot(self.dates_sorted, vals5, label = "UPIN")





        if self.upin_mean_field is None:
            self.upin_mean_field = np.mean(list(self.date_to_upin_field.values()), axis = 0)


        plt.legend()

        plt.title("{0}: acc={1} km**2".format(self.counter, self.acc_area_km2[i, j]))



#        plt.figure()
#        ax1 = plt.gca()
#        to_plot_2d = np.ma.masked_where(mask < 0.5, self.upin_mean_field)
#        img = self.basemap.pcolormesh(self.x_pr, self.y_pr, to_plot_2d, ax = ax1)
#        plt.colorbar(img, ax = ax1)
#        self.basemap.drawcoastlines(ax = ax1)
#        plt.title("min-max: {0};{1}".format(to_plot_2d.min(), to_plot_2d.max()))
#
#        self.ax.annotate(str(self.counter), (event.xdata, event.ydata), font_properties =
#                FontProperties(size=10), bbox=dict(boxstyle="round", fc="w"))
#        self.ax.redraw_in_frame()
#
#        plt.figure()
#        ax1 = plt.gca()
#        to_plot_2d = np.ma.masked_where(mask < 0.5, self.data_manager.cbf)
#        img = self.basemap.pcolormesh(self.x_pr, self.y_pr, to_plot_2d, ax = ax1)
#        plt.colorbar(img, ax = ax1)
#        self.basemap.drawcoastlines(ax = ax1)
#
#
#        plt.title("CBF, {0:g}: v= {1}, min={2}, max={3}".format(self.counter, to_plot_2d[i,j], to_plot_2d.min(), to_plot_2d.max()))
#
#
#        plt.figure()
#        ax1 = plt.gca()
#        to_plot_2d = np.ma.masked_where(mask < 0.5, self.data_manager.bankfull_storage_m3)
#        img = self.basemap.pcolormesh(self.x_pr, self.y_pr, to_plot_2d, ax = ax1)
#        plt.colorbar(img, ax = ax1)
#        self.basemap.drawcoastlines(ax = ax1)
#        plt.title("STBM, {0}: v= {1}".format(self.counter, to_plot_2d[i,j]))
#
#
#        plt.figure()
#        ax1 = plt.gca()
#        mbf = self.data_manager.manning_bf
#        to_plot_2d = np.ma.masked_where(mask < 0.5, mbf)
#        img = self.basemap.pcolormesh(self.x_pr, self.y_pr, to_plot_2d, ax = ax1)
#        plt.colorbar(img, ax = ax1)
#        self.basemap.drawcoastlines(ax = ax1)
#        plt.title("MABF, {0}: v= {1}, min={2}, max={3}".format(self.counter, to_plot_2d[i,j], to_plot_2d.min(), to_plot_2d.max()))
#
#        plt.figure()
#        ax1 = plt.gca()
#        to_plot_2d = np.ma.masked_where(mask < 0.5, self.slope)
#        img = self.basemap.pcolormesh(self.x_pr, self.y_pr, to_plot_2d, ax = ax1)
#        plt.colorbar(img, ax = ax1)
#        self.basemap.drawcoastlines(ax = ax1)
#        plt.title("SLOPe, {0}: v= {1}, min={2}, max={3}".format(self.counter, to_plot_2d[i,j], to_plot_2d.min(), to_plot_2d.max()))
#
#
#
#
#        plt.figure()
#        ax1 = plt.gca()
#        to_plot_2d = np.ma.masked_where(mask < 0.5, self.data_manager.lake_area)
#        img = self.basemap.pcolormesh(self.x_pr, self.y_pr, to_plot_2d, ax = ax1)
#        plt.colorbar(img, ax = ax1)
#        self.basemap.drawcoastlines(ax = ax1)
#        plt.title("lake area, {0}: v= {1}".format(self.counter, to_plot_2d[i,j]))
#
#        plt.figure()
#        ax1 = plt.gca()
#        to_plot_2d = np.ma.masked_where(mask < 0.5, self.coef_bf)
#        img = self.basemap.pcolormesh(self.x_pr, self.y_pr, to_plot_2d, ax = ax1)
#        plt.colorbar(img, ax = ax1)
#        self.basemap.drawcoastlines(ax = ax1)
#        plt.title("coef_bf, {0}: v= {1:.1g}, min={2:.1g}, max={3:.1g}".format(self.counter, to_plot_2d[i,j], to_plot_2d.min(), to_plot_2d.max()))
#





        plt.figure()

        #snow
        vals6 = [
                   np.sum( self.date_to_swe_field[d][mask == 1] ) for d in self.dates_sorted
        ]
        plt.plot(self.dates_sorted, vals6, label = "SWE")


        vals4 = [
           np.sum( self.date_to_swe_field[d][i,j] ) for d in self.dates_sorted
        ]
        plt.plot(self.dates_sorted, vals4, label = "GWST")


        vals5 = [
           np.sum( self.date_to_swsr_field[d][i,j] ) for d in self.dates_sorted
        ]
        plt.plot(self.dates_sorted, vals5, label = "SWSR")

        vals5 = [
           np.sum( self.date_to_swsl_field[d][i,j] ) for d in self.dates_sorted
        ]
        plt.plot(self.dates_sorted, vals5, label = "SWSL")
        plt.legend()
        plt.title("{0}, lkfr = {1}".format(self.counter, self.data_manager.lake_fraction[i,j]))







        fName = "route_params_{0}_{1}.bin".format(i, j)
        info = {}

        #traf -> dict( date -> value in m**3/s )

        traf_dict = dict(list(zip(self.dates_sorted,
            [self.date_to_traf_field[d][i,j] for d in self.dates_sorted])) )

        traf_dict = {"TRAF": traf_dict}

        info.update(traf_dict)

        upin_dict = dict(list(zip(self.dates_sorted,
            [self.date_to_upin_field[d][i,j] for d in self.dates_sorted])) )
        upin_dict = {"UPIN": upin_dict}
        info.update(upin_dict)


#        gwdi_dict = dict(zip(self.dates_sorted,
#                    [self.date_to_gwdi_field[d][i,j] for d in self.dates_sorted]) )
#        gwdi_dict = {"GWDI": gwdi_dict}
#        info.update(gwdi_dict)

        swsr_dict = dict(list(zip(self.dates_sorted,
                            [self.date_to_swsr_field[d][i,j] for d in self.dates_sorted])) )
        swsr_dict = {"SWSR": swsr_dict }
        info.update(swsr_dict)

        swsl_dict = dict(list(zip(self.dates_sorted,
                                    [self.date_to_swsl_field[d][i,j] for d in self.dates_sorted])) )
        swsl_dict = {"SWSL": swsl_dict }
        info.update(swsl_dict)

        stfl_dict = dict(list(zip(self.dates_sorted,
                            [self.date_to_stfl_field[d][i,j] for d in self.dates_sorted])) )
        stfl_dict = {"STFL": stfl_dict }
        info.update(stfl_dict)

        swst_dict = dict(list(zip(self.dates_sorted,
                            [self.date_to_swst_field[d][i,j] for d in self.dates_sorted])) )
        swst_dict = {"SWST": swst_dict }
        info.update(swst_dict)


        info["SBFM"] = self.data_manager.bankfull_storage_m3[i,j]
        info["CBF"] = self.data_manager.cbf[i,j]
        info["LKFR"] = self.data_manager.lake_fraction[i,j]
        info["LKAR"] = self.data_manager.lake_area[i,j]
        info["LKOU"] = self.lake_outlet[i,j]


        pickle.dump(info, open(fName, mode="w"))
















        self.counter += 1
        plt.show()
        pass




def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print("Hello world")
  
