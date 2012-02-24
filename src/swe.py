from datetime import timedelta
from datetime import datetime
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import application_properties
from cru.temperature import CRUDataManager

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

__author__ = 'huziy'

class SweDataManager(CRUDataManager):
    def __init__(self, path = "data/swe_ross_brown/swe.nc", var_name = ""):
        CRUDataManager.__init__(self, path=path, var_name=var_name)
        print self.nc_dataset.variables.keys()
        pass

    def _init_fields(self, nc_dataset):
        print "init_fields"
        nc_vars = nc_dataset.variables
        times = nc_vars["time"][:]

        lons = nc_vars["longitude"][:]
        lats = nc_vars["latitude"][:]


        self.lons2d, self.lats2d = lons, lats




        time_units_s = nc_vars["time"].units

        step_s, start_date_s = map(lambda x: x.strip(), time_units_s.split("since"))
        print step_s, start_date_s

        start_date = datetime.strptime(start_date_s, "%Y-%m-%d %H:%M:%S")
        if step_s == "hours":
            self.times = map(lambda x: start_date + timedelta(minutes = int( x * 60) ), times )
        elif step_s == "days":
            self.times = map(lambda x: start_date + timedelta(minutes = int(x * 60 * 24)), times )

        print self.times[0]
        print self.times[-1]
        self.var_data = nc_vars[self.var_name][:]

        pass

def main():
    from permafrost import draw_regions
    dm = SweDataManager(var_name="SWE")

    b, lons2d, lats2d = draw_regions.get_basemap_and_coords()

    x, y = b(dm.lons2d, dm.lats2d)


    fig = plt.figure()

    start_year = 1981
    end_year = 1997


    levels = [10,] + range(20, 120, 20) + [150,200, 300,500,1000]
    cmap = mpl.cm.get_cmap(name="jet_r", lut = len(levels))
    norm = colors.BoundaryNorm(levels, cmap.N)


    gs = gridspec.GridSpec(1,2)
    ax = fig.add_subplot(gs[0,0])
    data = dm.get_mean(start_year, end_year, months = [3])
    img = b.contourf(x, y, data.copy(), ax = ax, cmap = cmap, norm = norm, levels = levels)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img,  cax = cax)
    b.drawcoastlines(ax = ax)
    ax.set_title("SWE (not interp.), \n DJF period: {0} - {1}".format(start_year, end_year))


    ax = fig.add_subplot(gs[0,1])
    data_projected = dm.interpolate_data_to(data, lons2d, lats2d, nneighbours=1)
    x, y = b(lons2d, lats2d)
    img = b.contourf(x, y, data_projected, ax = ax, levels = img.levels)

    #add pretty colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img,  cax = cax)

    b.drawcoastlines(ax = ax)
    ax.set_title("SWE ( interp.), \n DJF period: {0} - {1}".format(start_year, end_year))

    plt.savefig("swe_rb_djf.png")


    pass




if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  