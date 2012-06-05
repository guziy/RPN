from matplotlib.ticker import MultipleLocator, LinearLocator
from numpy.lib.function_base import meshgrid
import application_properties
from permafrost import draw_regions

__author__ = 'huziy'

import numpy as np

from rpn.rpn import RPN
from matplotlib import cm
import matplotlib.pyplot as plt


class Cell:
    def __init__(self, i, j, lake_limit = 0.95):
        self.i = i
        self.j = j
        self.next = None
        self.previous = []
        self.lakefr = None
        self.lake_limit = 0.6


        pass


    def get_prev_lake_indexes(self):
        i_arr, j_arr = [self.i], [self.j]
        for p in self.previous:
            if p.lakefr < self.lake_limit:
                continue
            i1, j1 = p.get_prev_lake_indexes()
            i_arr.extend(i1)
            j_arr.extend(j1)
        return np.array(i_arr), np.array(j_arr)


    def get_lake_size_in_indexes(self):
        res = int(self.lakefr > self.lake_limit)
        for p in self.previous:
            if p.lakefr < self.lake_limit:
                continue
            res += p.get_lake_size_in_indexes()
        return res


    def get_acc_index(self):
        res = 1
        for p in self.previous:
            res += p.get_acc_index()
        return res

    def set_next(self, the_cell):
        self.next = the_cell
        if the_cell is not None:
            the_cell.previous.append(self)

def main():
    path = "/home/huziy/skynet3_exec1/from_guillimin/cell_area.rpn"
    r = RPN(path)
    tile = r.get_first_record_for_name("TILE")
    lkid = r.get_first_record_for_name("LKID")
    larea = r.get_first_record_for_name("AREA")
    r.close()



    #path = "/home/huziy/skynet3_exec1/from_guillimin/infocell.rpn"
    path = "infocell_260x260.rpn"
    rObj = RPN(path)
    dirs = rObj.get_first_record_for_name("FLDR")
    facc = rObj.get_first_record_for_name("FACC")
    lkfr = rObj.get_first_record_for_name("LKFR")
    lkou = rObj.get_first_record_for_name("LKOU")
    lons2d, lats2d = rObj.get_longitudes_and_latitudes()
    rObj.close()


    di_list = np.array([1,1,0,-1,-1,-1,0,1])
    dj_list = np.array([0,-1,-1,-1,0,1,1,1])

    delta_indices = np.log2(dirs[dirs > 0])
    delta_indices = delta_indices.astype(int)

    di = di_list[delta_indices].astype(float)
    dj = dj_list[delta_indices].astype(float)

    du = di / np.sqrt(di ** 2 + dj ** 2)
    dv = dj / np.sqrt(di ** 2 + dj ** 2)


    for i in xrange(100):
        print du[i], dv[i], np.log2(dirs[dirs > 0][i])

    du2d = np.ma.masked_all(dirs.shape)
    dv2d = np.ma.masked_all(dirs.shape)

    du2d[dirs>0] = du
    dv2d[dirs>0] = dv

    print du2d[161,145],dv2d[161,145],dirs[161,145]

    iv = xrange(dirs.shape[0])
    jv = xrange(dirs.shape[1])

    jv, iv = meshgrid(jv, iv)


    iv = iv.astype(float)
    jv = jv.astype(float)

    iv -= 0.5
    jv -= 0.5

    print iv.min(), iv.max()

    plt.figure()
    #field = np.ma.masked_where((field > 1e10) | (field < 1e7) , field)
    #tile1 = np.ma.masked_where((tile != 25) & (tile != 26), tile)
    plt.pcolormesh(iv,jv,tile,cmap = cm.get_cmap("jet", 36))
    plt.colorbar()
    plt.xlim(75, 100)
    plt.ylim(130,142)


    
    plt.figure()
    lkid = lkid.astype(int)
    print lkid[161,146], lkid[161,145],lkid[160,145:148]

    
    for i in xrange(1,121):
        x = np.ma.masked_where(lkid != i, lkid)
        if x.count() == 1 or x.count() == 0:
            print i, x.count()
    #lkid = np.ma.masked_where(~((lkid == 27)|(lkid == 28)) , lkid)

    lkid = np.ma.masked_where(lkid <= 0, lkid)
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(
        file_path=path,
        lon1=-68, lat1=52, lon2=16.65, lat2=0
    )
    x,y = b(lons2d, lats2d)
    img = b.pcolormesh(x,y,lkid,cmap = cm.get_cmap("jet", 100))
    plt.colorbar(img, ticks = MultipleLocator(base = 10))
    b.contour(x, y,tile, levels = xrange(48), colors = "k", linewidth = 0.5)
    b.drawcoastlines(linewidth=0.5)

    #plt.quiver(iv+0.5, jv+0.5, du2d, dv2d, scale = 30, width = 0.005, color="k", pivot="middle")
    #plt.xlim(30, 70)
    #plt.ylim(10,35)

    plt.figure()
    plt.pcolormesh(iv,jv,lkfr,cmap = cm.get_cmap("jet", 11))
    plt.colorbar()
    plt.xlim(155, 165)
    plt.ylim(140,150)


    

    plt.figure()
    d = np.ma.masked_all(dirs.shape)
    d[dirs > 0] = np.log2(dirs[dirs > 0]) 
    plt.pcolormesh(iv, jv, d,cmap = cm.get_cmap("jet", 8))
    plt.xlim(155, 165)
    plt.ylim(140,150)
    plt.colorbar(ticks = MultipleLocator(base = 1))
    plt.quiver(iv+0.5, jv+0.5, du2d, dv2d, scale = 4.5, width = 0.035, color="k", pivot="middle", units="inches")


    plt.figure()
    plt.title("Lake area")
    plt.pcolormesh(iv, jv, np.ma.masked_where( larea < 1.0e8, np.log(larea) ),cmap = cm.get_cmap("jet", 8))
    #plt.xlim(155, 165)
    #plt.ylim(140,150)
    print larea.min(), larea.max()
    plt.colorbar(ticks = LinearLocator(numticks = 10))
    #plt.quiver(iv+0.5, jv+0.5, du2d, dv2d, scale = 4.5, width = 0.035, color="k", pivot="middle", units="inches")



    plt.show()

    pass

def read_directions():
    #path = "/home/huziy/skynet3_exec1/from_guillimin/infocell.rpn"
    path = "infocell_260x260.rpn"

    i_start = 20
    j_start = 20
    i_end = i_start + 200
    j_end = j_start + 200
    rObj = RPN(path)
    dirs = rObj.get_first_record_for_name("FLDR")[i_start:i_end, j_start:j_end]
    facc = rObj.get_first_record_for_name("FACC")[i_start:i_end, j_start:j_end]
    lkfr = rObj.get_first_record_for_name("LKFR")[i_start:i_end, j_start:j_end]
    lkou = rObj.get_first_record_for_name("LKOU")[i_start:i_end, j_start:j_end]
    lons2d, lats2d = rObj.get_longitudes_and_latitudes()
    rObj.close()


    di_list = np.array([1,1,0,-1,-1,-1,0,1])
    dj_list = np.array([0,-1,-1,-1,0,1,1,1])



    delta_indices = np.log2(dirs[dirs > 0])
    delta_indices = delta_indices.astype(int)

    di = di_list[delta_indices].astype(float)
    dj = dj_list[delta_indices].astype(float)

    du = di / (di ** 2 + dj ** 2)
    dv = dj / (di ** 2 + dj ** 2)


    du_2d = np.ma.masked_all(dirs.shape)
    dv_2d = np.ma.masked_all(dirs.shape)

    du_2d[dirs > 0] = du
    dv_2d[dirs > 0] = dv


    assert isinstance(delta_indices, np.ndarray)
    print delta_indices.shape



    di_field = np.ma.masked_all(dirs.shape)
    dj_field = np.ma.masked_all(dirs.shape)


    di_field[dirs > 0] = di_list[delta_indices]
    dj_field[dirs > 0] = dj_list[delta_indices]


    di_field = di_field.astype(int)
    dj_field = dj_field.astype(int)

    nx, ny = dirs.shape
    i_p = xrange(nx)
    j_p = xrange(ny)

    j_p, i_p = meshgrid(j_p, i_p)
    i_n, j_n = np.zeros(i_p.shape), np.zeros(j_p.shape)

    i_n[dirs > 0] = i_p[dirs > 0] + di_field[dirs > 0]
    j_n[dirs > 0] = j_p[dirs > 0] + dj_field[dirs > 0]



    cells = []
    for i in xrange(nx):
        cells.append([])
        for j in xrange(ny):
            cells[i].append(Cell(i, j))

    for i in xrange(nx):
        for j in xrange(ny):
            cells[i][j].lakefr = lkfr[i,j]
            if dirs[i, j] <= 0:
                cells[i][j].set_next(None)
                continue

            di = di_field[i, j]
            dj = dj_field[i, j]

            cells[i][j].set_next(cells[i+di][j+dj])


    lkou = lkou.astype(int)
    iou, jou = np.where(lkou == 1)

    for i,j in zip(iou, jou):
        print(i,j, cells[i][j].get_lake_size_in_indexes(), cells[i][j].lakefr)

    print dirs[54, 150], dirs[53, 151]

    ii = 33
    jj = 96

    outlet = cells[ii][jj]
    i_lake, j_lake = outlet.get_prev_lake_indexes()

    save = di_field[i_lake, j_lake]
    di_field = np.ma.masked_all(di_field.shape)
    di_field[i_lake, j_lake] = save

    save = dj_field[i_lake, j_lake]
    dj_field = np.ma.masked_all(dj_field.shape)
    dj_field[i_lake, j_lake] = save




    plt.figure()
    #plt.contourf(dirs, levels = map( lambda x, p: x ** p, [2]*8, range(8)))
    #plt.pcolormesh(facc.transpose())




    plt.quiver(i_p,j_p, du_2d, dv_2d , scale = 0.5, width = 0.1 ,
        units = "xy", pivot = "middle", zorder = 5)

    plt.pcolormesh(i_p, j_p, dirs)
    plt.colorbar()

    print map( lambda x, p: x ** p, [2]*8, range(8))
    #plt.colorbar()



    #plot



    plt.show()

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    #read_directions()
    print "Hello world"
  
