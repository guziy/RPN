


#path = "/home/huziy/project-xgk-345-ab/Output/quebec_0.1_crcm5-hcd-rl-intfl_spinup/Samples/quebec_crcm5-hcd-rl-intfl_197902/pm1979010100_00016992p"
#path0 = "/home/huziy/project-xgk-345-ab/Output/quebec_0.1_crcm5-hcd-rl-intfl_spinup/Samples/quebec_crcm5-hcd-rl-intfl_197901/pm1979010100_00000000p"
from util.format_coords import FormatCoords

path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_spinup1/pm1979010100_00008928p"
path0 = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_spinup1/pm1979010100_00000000p"
var_name = "GWDI"


from rpn.rpn import RPN
import matplotlib.pyplot as plt
import numpy as np
import application_properties


application_properties.set_current_directory()

rObj = RPN(path)

data = rObj.get_4d_field(var_name)
rObj.close()



rObj = RPN(path0)

coef_bf = rObj.get_first_record_for_name("CBF")
mabf = rObj.get_first_record_for_name("MABF")
#lkfr = rObj.get_first_record_for_name("LF1")

rObj.close()

the_fields = [coef_bf, mabf]
the_titles = ["CBF", "MABF"]
for the_field, the_title in zip( the_fields, the_titles ):
    plt.figure()
    #coef_bf = np.ma.masked_where(coef_bf <= 0, coef_bf)
    plt.pcolormesh(the_field.transpose() % 1)

    dec = the_field % 1
    print "dec-min,dec-max = {0};{1}".format(dec.min(), dec.max())
    print coef_bf.min(), coef_bf.max()
    numrows, numcols = the_field.shape

    plt.title(the_title)
    plt.gca().format_coord = FormatCoords(the_field)
    plt.colorbar()



#plt.show()
#exit(0)

sorted_dates = sorted( data.keys() )



for d in sorted_dates:
    field2d = data[d].items()[0][1]
    #field2d = np.ma.masked_where(field2d < 0.001, field2d)

    print "min={0}, max = {1}".format(field2d.min(), field2d.max())

    numrows, numcols = field2d.shape
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if 0 <= col < numcols and 0 <= row < numrows:
            z = field2d.transpose()[row,col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)




    plt.figure()
    plt.pcolormesh(field2d.transpose() % 1)

    dec = field2d % 1
    print "dec-min,dec-max = {0};{1}".format(dec.min(), dec.max())

     
    plt.gca().format_coord = format_coord
    plt.colorbar()
    plt.show()


