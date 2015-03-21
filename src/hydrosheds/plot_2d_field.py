__author__ = "huziy"
__date__ = "$Apr 1, 2011 11:54:33 AM$"

from mpl_toolkits.basemap import Basemap

import numpy as np

import matplotlib.pyplot as plt

import application_properties

from math import *

from domains.map_parameters_amno import polar_stereographic
from netCDF4 import Dataset


xs = polar_stereographic.xs
ys = polar_stereographic.ys

lons = polar_stereographic.lons
lats = polar_stereographic.lats



inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = 400 * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]

font_size = 8
params = {
    'axes.labelsize': font_size,
    'font.size': font_size,
    'legend.fontsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'figure.figsize': fig_size
}

title_font_size = font_size

import pylab

pylab.rcParams.update(params)


def plot_drainage_areas(path="data/hydrosheds/test_using_splitting_amno.nc"):
    ds = Dataset(path)
    # basemap = polar_stereographic.basemap
    basemap = Basemap()
    lons = ds.variables["lon"][:]
    lats = ds.variables["lat"][:]
    channel_slope = ds.variables["slope"][:]

    lons[lons < 0] += 360

    x, y = basemap(lons, lats)

    acc_area = ds.variables["accumulation_area"][:]
    acc_area = np.log(acc_area)
    acc_area = np.ma.masked_where(channel_slope < 0, acc_area)

    basemap.pcolormesh(x, y, acc_area)
    basemap.drawcoastlines()
    plt.colorbar()
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.show()


def plot_source_flow_accumulation(sub=-1, vmin=0, vmax=10):
    path = 'data/hydrosheds/corresponding_DA.nc'
    ds = Dataset(path)
    data = ds.variables['DA_source'][:, :]

    lons = ds.variables['longitude'][:, :]
    lats = ds.variables['latitude'][:, :]

    basemap = polar_stereographic.basemap

    data = np.ma.masked_where(data <= 0, data)
    lons = np.ma.masked_where(lons == 0, lons)

    if sub < 0:
        plt.figure()
    x, y = basemap(lons, lats)
    # print np.log(data).shape
    print np.min(data), np.max(data)
    #    print np.min(np.log(data)), np.max(np.log(data))
    ax = basemap.pcolormesh(x, y, np.ma.log10(data), vmin=vmin, vmax=vmax)
    basemap.drawcoastlines()

    plt.colorbar()
    return ax


def plot_target_flow_accumulations(sub=-1, vmin=0, vmax=10):
    path = 'data/hydrosheds/corresponding_DA.nc'
    ds = Dataset(path)
    data = ds.variables['DA_target'][:, :]

    lons = ds.variables['longitude'][:, :]
    lats = ds.variables['latitude'][:, :]

    basemap = polar_stereographic.basemap

    data = np.ma.masked_where(data <= 0, data)
    lons = np.ma.masked_where(lons == 0, lons)
    if sub < 0:
        plt.figure()
    x, y = basemap(lons, lats)
    # print np.log(data).shape
    print np.min(data), np.max(data)
    #   print np.min(np.log(data)), np.max(np.log(data))
    ax = basemap.pcolormesh(x, y, np.ma.log10(data), vmin=vmin, vmax=vmax)
    basemap.drawcoastlines()

    plt.colorbar()

    return ax
    pass


def plot_scatter(path='data/hydrosheds/corresponding_DA.nc', margin=20):
    nc_da = Dataset(path)
    # margin = 30
    v1 = nc_da.variables['DA_source'][margin:-margin, margin:-margin]
    v2 = nc_da.variables['DA_target'][margin:-margin, margin:-margin]

    # basin_path = 'data/infocell/amno180x172_basins.nc'

    # nc = NetCDFFile(basin_path)
    # mask = None
    # for k, v in nc.variables.iteritems():
    #    if mask is None:
    #        mask = v.data[:,:].copy()
    #    else:
    #        mask += v.data

    condition = (v1 > 0) & (v2 > 0)
    v1 = v1[condition]
    v2 = v2[condition]

    v11 = v1[v1 / v2 < 3]
    v22 = v2[v1 / v2 < 3]
    v1 = v11
    v2 = v22

    print len(v1), v1.shape
    print len(v2), v2.shape

    plt.figure()
    plt.grid(True)
    plt.scatter(np.log10(v1), np.log10(v2), linewidth=0, s=10)

    plt.xlabel('hydrosheds, $\\log_{10}(DA_{max})$ ')
    plt.ylabel('upscaled, $\\log_{10}(DA_{sim})$')

    min_x = np.min(np.log10(v1))

    x = plt.xlim()
    plt.xlim(min_x, x[1])
    plt.ylim(min_x, x[1])
    plt.plot([min_x, x[1]], [min_x, x[1]], color='k')

    me = 1 - np.sum(np.power(v1 - v2, 2)) / np.sum(np.power(v1 - np.mean(v1), 2))
    plt.title('ME = {0:.4f}'.format(me))
    plt.tight_layout()
    plt.savefig('da_scatter.png')
    return me


def compare_drainages_2d():
    # ###plot
    plt.figure()
    plt.subplot(1, 2, 1)
    plot_source_flow_accumulation(sub=1)

    plt.subplot(1, 2, 2)
    plot_target_flow_accumulations(sub=1)


def main():
    # compare_drainages_2d()
    # plot_scatter(path="/home/huziy/skynet3_exec1/hydrosheds/corresponding_DA_af_0.44deg.nc")
    # plot_scatter(path="/home/huziy/skynet3_exec1/hydrosheds/corresponding_DA_af_0.44deg_2.nc")
    # plot_scatter(path="/home/huziy/skynet3_exec1/hydrosheds/corresponding_DA_qc_260x260_0.1deg_2.nc")
    # plot_scatter(path="/home/huziy/skynet3_exec1/hydrosheds/corresponding_DA_qc_0.5deg.nc")
    # plot_scatter(path="/home/huziy/skynet3_rech1/Netbeans Projects/Java/DDM/corresponding_DA_0.5deg_86x86.nc")

    plot_scatter(path="/home/huziy/skynet3_rech1/Netbeans Projects/Java/DDM/corresponding_DA_0.1deg_260x260.v3.nc",
                 margin=20)
    # plot_scatter(path="/home/huziy/skynet3_exec1/hydrosheds/corresponding_DA_qc_260x260_0.1deg_2.nc")
    plt.show()
    pass


def plot_me_from_margin():
    margins = list(range(20, 100, 5))
    mes = []
    for margin in margins:
        me = plot_scatter(
            path="/home/huziy/skynet3_rech1/Netbeans Projects/Java/DDM/corresponding_DA_0.1deg_260x260.v3.nc",
            margin=margin)
        mes.append(me)

    plt.figure()
    plt.plot(margins, mes)
    plt.xlabel("Domain margin in grid points")
    plt.ylabel("Modelling efficiency")
    plt.title("High resolution grid")
    plt.savefig("me_from_margin.png")
    # plt.show()



    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    # plot_drainage_areas()
    # plot_me_from_margin()
    print "Hello World"
