
from ctypes import c_float
from ctypes import POINTER
from ctypes import create_string_buffer

from ctypes import c_char_p
from ctypes import byref

__author__="huziy"
__date__ ="$Apr 5, 2011 12:26:05 PM$"

from ctypes import *
import application_properties
import numpy as np



import level_kinds
#TODO: maybe determine time step from file data
class RPN():
    def __init__(self, path = ''):
        try:
            self._dll = CDLL('rmnlib.so')
        except OSError:
            self._dll = CDLL('lib/rmnlib.so')

        self.VARNAME_DEFAULT = 8 * ' '
        self.VARTYPE_DEFAULT = 4 * ' '
        self.ETIKET_DEFAULT  = 16 * ' '
        self.GRIDTYPE_DEFAULT = 2 * ' '

        self._current_output_dt_seconds = -1

        self._current_info = None ##map containing info cocerning the last read record


        self.current_grid_type = 'Z' #self.GRIDTYPE_DEFAULT
        self.current_grid_reference = 'E' #TODO: maybe it can be determined somehow from the record header

        rpn_file_path = create_string_buffer(path)
        options = c_char_p('RND+R/O')
        dummy = c_int(0)

        self._dll.fnom_wrapper.argtypes = [POINTER(c_int), c_char_p, c_char_p, c_int]
        self._dll.fnom_wrapper.restype = c_int
        self._file_unit = c_int()
        self._dll.fnom_wrapper(byref(self._file_unit), rpn_file_path, options, dummy)
        print self._file_unit.value



        options.value = 'RND'
        self.nrecords = self._dll.fstouv_wrapper(self._file_unit, options)


        #set argument and return types of the library functions
        #fstinf
        self._dll.fstinf_wrapper.restype = c_int
        self._dll.fstinf_wrapper.argtypes = [c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int),
                                       c_int, c_char_p, c_int, c_int, c_int, c_char_p, c_char_p ]

        #ip1_all
        self._dll.ip1_all_wrapper.argtypes = [c_float, c_int]
        self._dll.ip1_all_wrapper.restype = c_int

        #fstluk
        self._dll.fstluk_wrapper.restype = c_int
        self._dll.fstluk_wrapper.argtypes = [POINTER(c_float), c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int)]

        #ezgdef_fmem
        self._dll.ezgdef_fmem_wrapper.restype = c_int
        self._dll.ezgdef_fmem_wrapper.argtypes = [c_int, c_int, c_char_p, c_char_p,
                                                  c_int, c_int, c_int, c_int,
                                                  POINTER(c_float), POINTER(c_float)
                                                  ]
        #gdll
        self._dll.gdll_wrapper.restype = c_int
        self._dll.gdll_wrapper.argtypes = [c_int, POINTER(c_float), POINTER(c_float)]

        #fstfrm
        self._dll.fstfrm_wrapper.restype = c_int
        self._dll.fstfrm_wrapper.argtypes = [c_int]

        #fclos
        self._dll.fclos_wrapper.restype = c_int
        self._dll.fclos_wrapper.argtypes = [c_int]

        #fstsui
        self._dll.fstsui_wrapper.restype = c_int
        self._dll.fstsui_wrapper.argtypes = [c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int)]

        #convip
        p_c_int = POINTER(c_int)
        self._dll.convip_wrapper.argtypes = [p_c_int, POINTER(c_float), p_c_int, p_c_int, c_char_p, p_c_int]

    def get_output_step_in_seconds(self):
        return self._current_output_dt_seconds

    def close(self):
        print  'close status: ', self._dll.fstfrm_wrapper(self._file_unit)
        self._dll.fclos_wrapper(self._file_unit)
 #       dlclose(self._dll._handle)
        del self._dll

    def get_number_of_records(self):
        return self._dll.fstnbr_wrapper(self._file_unit)

    def get_key_of_any_record(self):
        ni = c_int(0)
        nj = c_int(0)
        nk = c_int(0)
        datev = c_int(-1)
        etiket = create_string_buffer( self.ETIKET_DEFAULT )


        ip1 = c_int(-1)
        ip2 = c_int(-1)
        ip3 = c_int(-1)
        in_typvar = create_string_buffer(self.VARTYPE_DEFAULT)

        
        in_nomvar = create_string_buffer(self.VARNAME_DEFAULT)

        #int fstinf_wrapper(int iun, int *ni, int *nj, int *nk, int datev,char *in_etiket,
        #             int ip1, int ip2, int ip3, char *in_typvar, char *in_nomvar)

        key = self._dll.fstinf_wrapper(self._file_unit, byref(ni), byref(nj), byref(nk), datev, etiket,
                                 ip1, ip2, ip3, in_typvar, in_nomvar
                                )
        
        return key


    #get longitudes for the record
    def get_longitudes_and_latitudes(self):

        key = self.get_key_of_any_record()
        info = self._get_record_info(key, verbose = True) #sets grid type
        ig = info['ig']

        ni = c_int(0)
        nj = c_int(0)
        nk = c_int(0)
        datev = c_int(-1)

        
        ip1 = c_int(-1)
        ip2 = c_int(-1)
        ip3 = c_int(-1)


        print 'varname default = ', self.VARNAME_DEFAULT
       
        #read longitudes
        in_nomvar = '>>'
        in_nomvar = create_string_buffer(in_nomvar[:2])
        etiket = create_string_buffer(' ')
        in_typvar = create_string_buffer(' ')

        hor_key = self._dll.fstinf_wrapper(self._file_unit, byref(ni), byref(nj), byref(nk),
                                            datev, etiket, ip1, ip2, ip3, in_typvar, in_nomvar)
        print in_nomvar.value
        print in_typvar.value
        assert hor_key >= 0, 'hor_key = {0}'.format(hor_key)

        print 'hor_key = ', hor_key

        self._get_record_info(hor_key)

        hor_key = c_int(hor_key)
        data = np.zeros((ni.value,), dtype = np.float32)
        self._dll.fstluk_wrapper(data.ctypes.data_as(POINTER(c_float)), hor_key, ni, nj, nk)
        lons = data[:]

        #read latitudes
        in_nomvar = '^^'
        in_nomvar = create_string_buffer(in_nomvar[:2])
        ver_key = self._dll.fstinf_wrapper(self._file_unit, byref(ni), byref(nj), byref(nk),
                                            datev, etiket, ip1, ip2, ip3, in_typvar, in_nomvar)
        ver_key = c_int(ver_key)
        data = np.zeros((nj.value, ), dtype = np.float32)
        self._dll.fstluk_wrapper(data.ctypes.data_as(POINTER(c_float)), ver_key, ni, nj, nk)
        lats = data[:]

        print 'ver_key = ', ver_key.value

        n_lons = lons.shape[0]
        n_lats = lats.shape[0]

        print 'grid type: ', self.current_grid_type
        print 'grid ref: ', self.current_grid_reference

        info = self._get_record_info(hor_key, verbose = True)
        ig = info['ig']

        print self.current_grid_type
        grid_type = create_string_buffer(self.current_grid_type)
        grid_reference = create_string_buffer(self.current_grid_reference)

        ezgdef = self._dll.ezgdef_fmem_wrapper(c_int(n_lons), c_int(n_lats),
                grid_type, grid_reference,
                ig[0], ig[1], ig[2], ig[3],
                lons.ctypes.data_as(POINTER(c_float)),
                lats.ctypes.data_as(POINTER(c_float)))



        lons_2d = np.zeros((n_lats, n_lons), dtype = np.float32)
        lats_2d = np.zeros((n_lats, n_lons), dtype = np.float32)
        
        self._dll.gdll_wrapper(ezgdef, lats_2d.ctypes.data_as(POINTER(c_float)), 
                                       lons_2d.ctypes.data_as(POINTER(c_float)))

        
        return np.transpose(lons_2d), np.transpose(lats_2d)
        

  

    #returns first met record for the field varname
    def get_first_record_for_name(self, varname):
        return self.get_first_record_for_name_and_level(varname, -1)

    def get_first_record_for_name_and_level(self, varname = '', level = -1,
                                                  level_kind = level_kinds.ARBITRARY):

        ni = c_int(0)
        nj = c_int(0)
        nk = c_int(0)
        datev = c_int(-1)
        etiket = create_string_buffer( self.ETIKET_DEFAULT )


        if level == -1:
            ip1 = c_int(-1)
        else:
            ip1 = c_int(self._dll.ip1_all_wrapper(c_float(level), c_int(level_kind)))
        ip2 = c_int(-1)
        ip3 = c_int(-1)
        in_typvar = create_string_buffer(self.VARTYPE_DEFAULT)

        in_nomvar = self.VARNAME_DEFAULT
        in_nomvar = varname + in_nomvar
        in_nomvar = in_nomvar[:8]
        in_nomvar = create_string_buffer(in_nomvar)

        #int fstinf_wrapper(int iun, int *ni, int *nj, int *nk, int datev,char *in_etiket,
        #             int ip1, int ip2, int ip3, char *in_typvar, char *in_nomvar)


        key = self._dll.fstinf_wrapper(self._file_unit, byref(ni), byref(nj), byref(nk), datev, etiket,
                                 ip1, ip2, ip3, in_typvar, in_nomvar
                                )

        
        data = np.zeros((nk.value, nj.value, ni.value,), dtype = np.float32)
        
        self._current_info = self._get_record_info(key)

        #read the record
        print self._dll.fstluk_wrapper(data.ctypes.data_as(POINTER(c_float)), key, ni, nj, nk)
        data = np.transpose(data, (2, 1, 0))
        return data[:,:, 0]

    def _get_record_info(self, key, verbose = False):
        dateo = c_int()
        dt_seconds = c_int()
        npas = c_int()
        ni = c_int()
        nj = c_int()
        nk = c_int()

        nbits = c_int()
        datyp = c_int()

        ip1 = c_int()
        ip2 = c_int()
        ip3 = c_int()

        typvar = create_string_buffer(self.VARTYPE_DEFAULT)
        nomvar = create_string_buffer(self.VARNAME_DEFAULT)
        etiket = create_string_buffer(self.ETIKET_DEFAULT)
        grid_type = create_string_buffer(self.GRIDTYPE_DEFAULT)

        ig1 = c_int()
        ig2 = c_int()
        ig3 = c_int()
        ig4 = c_int()

        extra1 = c_int()
        extra2 = c_int()
        extra3 = c_int()

        swa = c_int()
        lng = c_int()
        dltf = c_int()
        ubc = c_int()


        self._dll.fstprm_wrapper(key,
             byref(dateo), byref(dt_seconds), byref(npas),
             byref(ni), byref(nj), byref(nk),
             byref(nbits), byref(datyp),
             byref(ip1), byref(ip2), byref(ip3),

             typvar, nomvar, etiket, grid_type,
             byref(ig1), byref(ig2), byref(ig3), byref(ig4),
             byref(swa), byref(lng),
             byref(dltf), byref(ubc),
             byref(extra1), byref(extra2), byref(extra3))

        if verbose:
            print 'ip ', [ip1.value, ip2.value, ip3.value]
            print 'grtype', grid_type.value
            print 'ig',[ig1.value, ig2.value, ig3.value, ig4.value]
            print 'varname: ', nomvar.value

        
        self._current_output_dt_seconds = dt_seconds.value * npas.value
        if '>>' in nomvar.value or '^^' in nomvar.value:
            self.current_grid_reference = grid_type.value
        else:
            self.current_grid_type = grid_type.value
        print 'current grid type ', self.current_grid_type

        result = {}
        result['ig'] = [ig1, ig2, ig3, ig4]
        result['ip'] = [ip1, ip2, ip3]
        result['shape'] = [ni, nj, nk]
        result['dateo'] = dateo
        result['dt_seconds'] = dt_seconds
        result['npas'] = npas
        result['varname'] = nomvar
        self._current_info = result #update info from the last read record
        return result


    ##returns None, if there is no next record satisfying the last search parameters
    def get_next_record(self):
        ni, nj, nk = self._current_info['shape']
        key = self._dll.fstsui_wrapper(self._file_unit, byref(ni), byref(nj), byref(nk))

 
        if key <= 0: return None
        data = np.zeros((nk.value, nj.value, ni.value,), dtype = np.float32)
        self._dll.fstluk_wrapper(data.ctypes.data_as(POINTER(c_float)), key, ni, nj, nk)
        data = np.transpose(data, (2, 1, 0))

        self._get_record_info(key)
        return data[:,:,0]

    def get_current_level(self, level_kind = level_kinds.ARBITRARY):
        ip1 = self._current_info['ip'][0]
        print 'ip1 = ', ip1
        level_value = c_float(-1)
        mode = c_int(-1) #from ip to real value
        kind = c_int(level_kind)
        flag = c_int(0)
        string = create_string_buffer(' ', 128)
        #print 'before convip'
        self._dll.convip_wrapper( byref(ip1), byref(level_value), byref( kind ) , byref(mode), string, byref(flag) )
        return level_value.value
        pass

    def get_current_validity_date(self):
        '''
        returns validity date in hours from the start
        '''

        ##have to search for data records in order to see dt and npas
        if self._current_info == None:
            key = self.get_key_of_any_record()
            self._get_record_info(key)

        while self._current_info['varname'].value.strip().lower() in ['>>','^^', 'hy']:
            self.get_next_record()


        print 'dateo  = ' , self._current_info['dateo']
        print 'dt_seconds = ', self._current_info['dt_seconds']
        print 'npas = ', self._current_info['npas']
        print 'varname = ', self._current_info['varname'].value
        return self._current_info['ip'][1] ##ip2
        pass

  
    def get_3D_field(self, name = 'SAND', level_kind = level_kinds.ARBITRARY):
        '''
        returns a map {level => 2d field}
        '''
        result = {}
        data1 = self.get_first_record_for_name(name)
        result[self.get_current_level(level_kind = level_kind)] = data1

        while data1 != None:
            data1 = self.get_next_record()
            if data1 != None:
                result[self.get_current_level(level_kind = level_kind)] = data1
        return result


def test():
    #path = 'data/geophys_africa'
    path = 'data/pm1989010100_00000003p'
    rpnObj = RPN(path)
    datev = rpnObj.get_current_validity_date()
    print 'validity date = ', datev.value
    print rpnObj.get_number_of_records()
    rpnObj.close()


if __name__ == "__main__":
    test()
    print "Hello World"
