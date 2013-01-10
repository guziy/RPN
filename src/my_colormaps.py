from matplotlib.colors import ListedColormap

__author__="huziy"
__date__ ="$9-Mar-2011 11:55:16 AM$"
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def get_cmap_from_ncl_spec_file(path = "", ncolors = None):
    """
    if ncolors is ommited, the value is taken from the file
    """
    f = open(path)
    Nmax = 256.0
    all_numbers = []
    for line in f:
        line = line.strip()
        if line == "": continue
        if line.lower().startswith("ncolors") and ncolors is None:
            ncolors = int( line.split("=")[1].strip() )
            continue
        if not line.startswith("#") and not line.lower().startswith("ncolors"):
            vals = map( lambda x: float(x.strip())/Nmax, line.split())
            all_numbers.append(vals)

    all_numbers = np.array(all_numbers)
    red_numbers = all_numbers[:,0]
    green_numbers = all_numbers[:, 1]
    blue_numbers = all_numbers[:, 2]
    dx = 1.0 / float(len(red_numbers) - 1)
    reds = []
    greens = []
    blues = []
    x = 0
    for i in range(len(red_numbers)):

        if i == len(red_numbers) - 1:
            x = 1
        reds.append((x,red_numbers[i], red_numbers[i]))
        greens.append((x, green_numbers[i], green_numbers[i]))
        blues.append((x, blue_numbers[i], blue_numbers[i]))
        x += dx


    cdict = {
        'blue':  blues,
        'green': greens,
        'red':  reds
    }

    return mpl.colors.LinearSegmentedColormap('sig_colormap', cdict, ncolors)



    pass


def get_cmap_wo_red(ncolors = 1024):
    all_numbers = [
                    [0,    0.3,    1],
                    [0,    1,      1],
                    [0.5,  1,    0.5],
                    [0.6,    1,      0],
                    [0.8,    0.8,    0.1],
                    [1,    1,    0.0],

    ]

    all_numbers = np.array(all_numbers)
    red_numbers = all_numbers[:,0]
    green_numbers = all_numbers[:, 1]
    blue_numbers = all_numbers[:, 2]
    dx = 1.0 / float(len(red_numbers) - 1)
    reds = []
    greens = []
    blues = []
    x = 0
    for i in range(len(red_numbers)):

        if i == len(red_numbers) - 1:
            x = 1
        reds.append((x,red_numbers[i], red_numbers[i]))
        greens.append((x, green_numbers[i], green_numbers[i]))
        blues.append((x, blue_numbers[i], blue_numbers[i]))
        x += dx


    cdict = {
        'blue':  blues,
        'green': greens,
        'red':  reds
    }

    return mpl.colors.LinearSegmentedColormap('sig_colormap', cdict, ncolors)

def get_blue_colormap(ncolors = 1024):

    """
    :rtype : LinearSegmentedColormap
    """
    red_numbers = [
#        0.600000023841858,
#        0.723529458045960,
#        0.847058832645416,
#        1,
#        1,
#        1,
#        1,
#        1,
        0.833333313465118,
        0.694444417953491,
        0.555555522441864,
        0.416666656732559,
        0.277777761220932,
        0.138888880610466,
        0,
        0.0784313753247261
    ]

    green_numbers = [
#        0.200000002980232,
#        0.180392161011696,
#        0.160784319043159,
#        0,
#        0.187500000000000,
#        0.375000000000000,
#        0.562500000000000,
#        0.750000000000000,
        0.833333313465118,
        0.694444417953491,
        0.555555522441864,
        0.416666656732559,
        0.277777761220932,
        0.138888880610466,
        0,
        0.168627455830574
    ]

    blue_numbers = [
#        0,
#        0,
#        0,
#        0,
#        0.187500000000000,
#        0.375000000000000,
#        0.562500000000000,
#        0.750000000000000,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0.549019634723663
    ]
    dx = 1.0 / float(len(red_numbers) - 1)
    reds = []
    greens = []
    blues = []
    x = 0
    for i in range(len(red_numbers)):

        if i == len(red_numbers) - 1:
            x = 1
        reds.append((x,red_numbers[i], red_numbers[i]))
        greens.append((x, green_numbers[i], green_numbers[i]))
        blues.append((x, blue_numbers[i], blue_numbers[i]))
        x += dx


    cdict = {
        'blue':  blues,
        'green': greens,
        'red':  reds
    }

    return mpl.colors.LinearSegmentedColormap('diff_colormap', cdict, ncolors)


def get_red_colormap(ncolors = 1024):
    red_numbers = [
        0.600000023841858,
        0.723529458045960,
        0.847058832645416,
        1,
        1,
        1,
        1,
        1,
#        0.833333313465118,
#        0.694444417953491,
#        0.555555522441864,
#        0.416666656732559,
#        0.277777761220932,
#        0.138888880610466,
#        0,
#        0.0784313753247261
    ]

    green_numbers = [
        0.200000002980232,
        0.180392161011696,
        0.160784319043159,
        0,
        0.187500000000000,
        0.375000000000000,
        0.562500000000000,
        0.750000000000000,
#        0.833333313465118,
#        0.694444417953491,
#        0.555555522441864,
#        0.416666656732559,
#        0.277777761220932,
#        0.138888880610466,
#        0,
#        0.168627455830574
    ]

    blue_numbers = [
        0,
        0,
        0,
        0,
        0.187500000000000,
        0.375000000000000,
        0.562500000000000,
        0.750000000000000,
#        1,
#        1,
#        1,
#        1,
#        1,
#        1,
#        1,
#        0.549019634723663
    ]
    red_numbers.reverse()
    blue_numbers.reverse()
    green_numbers.reverse()

    dx = 1.0 / float(len(red_numbers) - 1)
    reds = []
    greens = []
    blues = []
    x = 0
    for i in range(len(red_numbers)):

        if i == len(red_numbers) - 1:
            x = 1
        reds.append((x,red_numbers[i], red_numbers[i]))
        greens.append((x, green_numbers[i], green_numbers[i]))
        blues.append((x, blue_numbers[i], blue_numbers[i]))
        x += dx


    cdict = {
        'blue':  blues,
        'green': greens,
        'red':  reds
    }

    return mpl.colors.LinearSegmentedColormap('diff_colormap', cdict, ncolors)


def get_lighter_jet_cmap(ncolors = 1024):
    all_numbers = [
                    [0,    0.3,    1],
                    [0,    1,      1],
                    [0.5,  1,    0.5],
                    [1,    1,      0],
                    [1,    0.8,    0.1],
                    [1,    0.0,    0],

    ]

    all_numbers = np.array(all_numbers)
    red_numbers = all_numbers[:,0]
    green_numbers = all_numbers[:, 1]
    blue_numbers = all_numbers[:, 2]
    dx = 1.0 / float(len(red_numbers) - 1)
    reds = []
    greens = []
    blues = []
    x = 0
    for i in range(len(red_numbers)):

        if i == len(red_numbers) - 1:
            x = 1
        reds.append((x,red_numbers[i], red_numbers[i]))
        greens.append((x, green_numbers[i], green_numbers[i]))
        blues.append((x, blue_numbers[i], blue_numbers[i]))
        x += dx


    cdict = {
        'blue':  blues,
        'green': greens,
        'red':  reds
    }

    return mpl.colors.LinearSegmentedColormap('sig_colormap', cdict, ncolors)

    pass

def get_sign_count_cmap(ncolors = 5):

    all_numbers = [
                    [0,    0.5,    1],
                    [0,    1,      1],
                    [0.5,  1,    0.5],
                    [1,    1,      0],
                    [1,    0.5,    0]
    ]

    all_numbers = np.array(all_numbers)
    red_numbers = all_numbers[:,0]
    green_numbers = all_numbers[:, 1]
    blue_numbers = all_numbers[:, 2]
    dx = 1.0 / float(len(red_numbers) - 1)
    reds = []
    greens = []
    blues = []
    x = 0
    for i in range(len(red_numbers)):

        if i == len(red_numbers) - 1:
            x = 1
        reds.append((x,red_numbers[i], red_numbers[i]))
        greens.append((x, green_numbers[i], green_numbers[i]))
        blues.append((x, blue_numbers[i], blue_numbers[i]))
        x += dx


    cdict = {
        'blue':  blues,
        'green': greens,
        'red':  reds
    }

    return mpl.colors.LinearSegmentedColormap('sig_colormap', cdict, ncolors)

    pass


def get_red_blue_colormap(ncolors = 1024, reversed = False):
    red_numbers = [
        0.600000023841858,
        0.723529458045960,
        0.847058832645416,
        1,
        1,
        1,
        1,
        1,
        0.833333313465118,
        0.694444417953491,
        0.555555522441864,
        0.416666656732559,
        0.277777761220932,
        0.138888880610466,
        0,
        0.0784313753247261
    ]

    green_numbers = [
        0.200000002980232,
        0.180392161011696,
        0.160784319043159,
        0,
        0.187500000000000,
        0.375000000000000,
        0.562500000000000,
        0.750000000000000,
        0.833333313465118,
        0.694444417953491,
        0.555555522441864,
        0.416666656732559,
        0.277777761220932,
        0.138888880610466,
        0,
        0.168627455830574
    ]

    blue_numbers = [
        0,
        0,
        0,
        0,
        0.187500000000000,
        0.375000000000000,
        0.562500000000000,
        0.750000000000000,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0.549019634723663
    ]

    if reversed:
        blue_numbers.reverse()
        red_numbers.reverse()
        green_numbers.reverse()

    dx = 1.0 / float(len(red_numbers) - 1)
    reds = []
    greens = []
    blues = []
    x = 0
    for i in range(len(red_numbers)):

        if i == len(red_numbers) - 1:
            x = 1
        reds.append((x,red_numbers[i], red_numbers[i]))
        greens.append((x, green_numbers[i], green_numbers[i]))
        blues.append((x, blue_numbers[i], blue_numbers[i]))
        x += dx


    cdict = {
        'blue':  blues,
        'green': greens,
        'red':  reds
    }

    return mpl.colors.LinearSegmentedColormap('diff_colormap', cdict, ncolors)





def get_red_white_blue_colormap(ncolors = 1024):

    red_numbers = [
                    0.600000023841858,
                    0.723529458045960,
                    0.847058832645416,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0.833333313465118,
                    0.666666686534882,
                    0.500000000000000,
                    0.333333343267441,
                    0.166666671633720,
                    0,
                    0.07843137532261
    ]

    green_numbers = [
                    0.200000002980232,
                    0.180392161011696,
                    0.160784319043159,
                    0,
                    0.250000000000000,
                    0.500000000000000,
                    0.750000000000000,
                    1,
                    1,
                    0.833333313465118,
                    0.666666686534882,
                    0.500000000000000,
                    0.333333343267441,
                    0.166666671633720,
                    0,
                    0.168627455830574
    ]

    blue_numbers = [
                    0,
                    0,
                    0,
                    0,
                    0.250000000000000,
                    0.500000000000000,
                    0.750000000000000,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0.549019634723663
    ]

    dx = 1.0 / float(len(red_numbers) - 1)
    reds = []
    greens = []
    blues = []
    x = 0
    for i in range(len(red_numbers)):

        if i == len(red_numbers) - 1:
            x = 1
        reds.append((x,red_numbers[i], red_numbers[i]))
        greens.append((x, green_numbers[i], green_numbers[i]))
        blues.append((x, blue_numbers[i], blue_numbers[i]))
        x += dx

    
    cdict = {
        'blue':  blues,
        'green': greens,
        'red':  reds
    }

    return mpl.colors.LinearSegmentedColormap('diff_colormap', cdict, ncolors)
    pass



    


def get_diff_colormap(ncolors = 1024):
    cdict = {

         'blue':  [(0.0,  0.0, 0.0),
                   (0.5,  0.0, 1.0),
                   (0.75, 0.5, 0.5),
                   (1.0,  0.4, 0.4)],
        
        'green': [ (0.0,  0.0, 0.0),
                   (0.45, 0.5, 0.5),
                   (0.5,  1.0, 1.0),
                   (0.55, 0.5, 0.5),
                   (1.0,  0.0, 0.0)],

         'red':  [( 0.0,  0.4, 0.4),
                   (0.25, 0.5, 0.5),
                   (0.5,  1.0, 0.0),
                   (1.0,  0.0, 0.0)]


    }
    
    return mpl.colors.LinearSegmentedColormap('diff_colormap', cdict, ncolors)

def get_diff_colormap1(ncolors = 1024):
    cdict = {

         'blue':  [(0.0,  0.0, 0.0),
                   (0.5,  0.0, 1.0),
                   (0.75, 0.9, 0.9),
                   (1.0,  0.7, 0.7)],

        'green': [ (0.0,  0.0, 0.0),
                   (0.45, 0.5, 0.5),
                   (0.5,  1.0, 1.0),
                   (0.55, 0.5, 0.5),
                   (1.0,  0.0, 0.0)],

         'red':  [( 0.0,  0.7, 0.7),
                   (0.25, 0.9, 0.9),
                   (0.5,  1.0, 0.0),
                   (1.0,  0.0, 0.0)]


    }

    return mpl.colors.LinearSegmentedColormap('diff_colormap', cdict, ncolors)


def cmap_map(function,cmap):
    """
    taken from:http://www.scipy.org/Cookbook/Matplotlib/ColormapTransformations
    Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous     points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red','green','blue'):         step_dict[key] = map(lambda x: x[0], cdict[key])
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(map( reduced_cmap, step_list))
    new_LUT = np.array(map( function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i,key in enumerate(('red','green','blue')):
        this_cdict = {}
        for j,step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j,i]
            elif new_LUT[j,i]!=old_LUT[j,i]:
                this_cdict[step] = new_LUT[j,i]
        colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector

    return mpl.colors.LinearSegmentedColormap('colormap',cdict,1024)


def test_ncl_map():

    import application_properties
    application_properties.set_current_directory()

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_axes([0.05, 0.65, 0.9, 0.15])

    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap = get_cmap_from_ncl_spec_file(path="colormap_files/topo_15lev.rgb",
                                            ncolors=  None),
                                   orientation='horizontal')
    plt.show()


def test():
    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_axes([0.05, 0.65, 0.9, 0.15])

    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap = get_cmap_wo_red(ncolors = 10),
                                   orientation='horizontal')
    plt.show()
if __name__ == "__main__":
    #test()
    test_ncl_map()
    print "Hello World"
