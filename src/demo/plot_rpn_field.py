from numpy.lib.function_base import meshgrid
import application_properties

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
        self.lake_limit = 0.95


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
    path = "/home/huziy/skynet3_rech1/test/cell_area.rpn"
    r = RPN(path)
    field = r.get_first_record_for_name("AREA")
    r.close()

    plt.figure()
    field = np.ma.masked_where((field > 1e10) | (field < 1e7) , field)
    plt.pcolormesh(field.transpose(),cmap = cm.get_cmap("jet", 100))
    plt.colorbar()
    plt.show()

    pass

def read_directions():
    path = "/home/huziy/skynet3_rech1/test/infocell.rpn"


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


    dirs_m = np.ma.masked_where(dirs <= 0, dirs)
    delta_indices = np.log2(dirs[dirs > 0])

    assert isinstance(delta_indices, np.ndarray)
    print delta_indices.shape
    delta_indices = delta_indices.astype(int)


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



    du = lons2d[i_n, j_n] - lons2d[i_p, j_p]
    dv = lats2d[i_n, j_n] - lats2d[i_p, j_p]

    plt.quiver(lons2d,lats2d, du, dj_field.transpose(), scale = 1,
        units = "xy", pivot = "middle")

    print map( lambda x, p: x ** p, [2]*8, range(8))
    #plt.colorbar()



    #plot



    plt.show()

if __name__ == "__main__":
    application_properties.set_current_directory()
    #main()
    read_directions()
    print "Hello world"
  