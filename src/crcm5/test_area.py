from netCDF4 import Dataset
from matplotlib.figure import Figure
from rpn.rpn import RPN
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'huziy'

import unittest

class MyTestCase(unittest.TestCase):

    def test_something(self):
        self.assertEqual(True, True)

    def setUp(self):
        super(MyTestCase, self).setUp()
        import application_properties
        application_properties.set_current_directory()


    def test_area2nc(self):
        path1 = "/home/huziy/skynet3_rech1/Netbeans Projects/Java/DDM/test_area0.5_crcm5.nc"
        path2 = "/home/huziy/skynet3_rech1/Netbeans Projects/Java/DDM/test_area0.5_eckert_iv.nc"

        x1 = Dataset(path1).variables["cell_area"][:].flatten()
        x2 = Dataset(path2).variables["cell_area"][:].flatten()

        cond = x1 > 0
        x1 = x1[cond]
        x2 = x2[cond]

        plt.figure()
        plt.scatter(x1, x2, c = "b", linewidths=0)
        x_min = min(x1)
        x_max = max(x1)
        plt.plot([x_min, x_max], [x_min, x_max],"k")

        plt.xlabel("CRCM5")
        plt.ylabel("Eckert IV")
        plt.title( "dA/A = {0:.4f}".format( np.mean(x2 - x1)/ np.mean(x1)) )
        plt.savefig("eckert_vs_crcm5.png")



    def test_area(self):
        path_rpn = "/home/huziy/skynet3_exec1/from_guillimin/quebec_highres_spinup_12_month_without_lakes/pm1985010100_00000000p"
        path_nc = "/home/huziy/skynet3_rech1/Netbeans Projects/Java/DDM/test_area.nc"

        area_rpn = RPN(path=path_rpn).get_first_record_for_name("DX")
        area_nc = Dataset(path_nc).variables["cell_area"][:]

        fig = plt.figure()
        assert isinstance(fig, Figure)


        flat_nc_area = area_nc[20:-20,20:-20].flatten()
        flat_rpn_area = area_rpn.flatten() / 1.0e6


        flat_rpn_area = flat_rpn_area[flat_nc_area > 0]
        flat_nc_area = flat_nc_area[flat_nc_area > 0]


        plt.scatter(flat_rpn_area, flat_nc_area, linewidths=0)

        print flat_nc_area - flat_rpn_area
        x1 = min(flat_rpn_area)
        x2 = max(flat_rpn_area)
        plt.plot([x1, x2], [x1, x2],"k")

        plt.xlabel("CRCM5")
        plt.ylabel("Upscaler")
        #plt.show()
        pass




if __name__ == '__main__':
    unittest.main()
