


path = "/home/huziy/project-xgk-345-ab/Output/quebec_0.1_crcm5-hcd-rl-intfl_spinup/Samples/quebec_crcm5-hcd-rl-intfl_197902/pm1979010100_00016992p"
path0 = "/home/huziy/project-xgk-345-ab/Output/quebec_0.1_crcm5-hcd-rl-intfl_spinup/Samples/quebec_crcm5-hcd-rl-intfl_197901/pm1979010100_00000000p"

var_name = "STFL"


from rpn.rpn import RPN
import matplotlib.pyplot as plt
import numpy as np

rObj = RPN(path)

data = rObj.get_4d_field(var_name)
rObj.close()



#rObj = RPN(path0)

#lkfr = rObj.get_first_record_for_name("LF1")

#rObj.close()
#plt.pcolormesh(lkfr.transpose())
#plt.colorbar()
#plt.show()
#exit(0)

sorted_dates = sorted( data.keys() )



for d in sorted_dates:
    field2d = data[d].items()[0][1]
    field2d = np.ma.masked_where(field2d < 0.001, field2d)

    numrows, numcols = field2d.shape
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = field2d.transpose()[row,col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y)




    plt.figure()
    plt.pcolormesh(field2d.transpose())
     
    plt.gca().format_coord = format_coord
    plt.colorbar()
    plt.show()


