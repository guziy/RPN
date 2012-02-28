import data_types

__author__="huziy"
__date__ ="$Apr 5, 2011 12:26:05 PM$"

from ctypes import *
import numpy as np

import os

import level_kinds

from datetime import datetime
from datetime import timedelta

class RPN():
    """
    Class for reading and writing rpn files
    usage:
        rObj = RPN(path = 'your path', mode = 'r')
        rObj = RPN(path = 'your path', mode = 'w')

    methods:
        get_first_record_for_name_and_level(self, varname = '', level = -1,
                                                  level_kind = level_kinds.ARBITRARY)
        get_3D_field(self, name = 'SAND', level_kind = level_kinds.ARBITRARY)
        get_current_level(self, level_kind = level_kinds.ARBITRARY)
        get_current_validity_date(self)
        get_first_record_for_name(self, varname)
        get_next_record(self)
        get_longitudes_and_latitudes(self)
        get_key_of_any_record(self)
        get_number_of_records(self)
        close(self)
        get_ip1_from_level(self, level, level_kind = level_kinds.ARBITRARY)
        write_2D_field(self, name = '', level = 1, level_kind = level_kinds.ARBITRARY, data = None )
    """
    def __init__(self, path = '', mode = 'r', start_century = 19):
        """
              start_century - is used for the calculation of the origin date, because
              of its ambiguous format MMDDYYR
        """
        if not os.path.isfile(path) and mode == 'r':
            raise Exception('{0} does not exist, or is not a file'.format(path))

        self.path = path
        try:
            self._dll = CDLL('rmnlib.so')
        except OSError:
            self._dll = CDLL('lib/rmnlib.so')

        self.VARNAME_DEFAULT = 8 * ' '
        self.VARTYPE_DEFAULT = 4 * ' '
        self.ETIKET_DEFAULT  = 16 * ' '
        self.GRIDTYPE_DEFAULT = 'Z'


        self.FROM_LEVEL_TO_IP1_MODE = 1
        self.FROM_IP1_TO_LEVEL_MODE = -1

        self._current_info = None ##map containing info concerning the last read record
        self.start_century = start_century

        self.current_grid_type = self.GRIDTYPE_DEFAULT
        self.current_grid_reference = 'E' 

        rpn_file_path = create_string_buffer(path)
        if mode == 'w':
            if os.path.isfile(path):
                os.remove(path)
            options = c_char_p('RND')
        else:
            options = c_char_p('RND+R/O')
        dummy = c_int(0)

        self._dll.fnom_wrapper.argtypes = [POINTER(c_int), c_char_p, c_char_p, c_int]
        self._dll.fnom_wrapper.restype = c_int
        self._file_unit = c_int()
        self._dll.fnom_wrapper(byref(self._file_unit), rpn_file_path, options, dummy)
        

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

        #fstecr
        self._dll.fstecr_wrapper.argtypes = [
            POINTER(c_float),
            c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
            c_int, c_int,
            c_char_p, c_char_p, c_char_p, c_char_p,
            c_int, c_int, c_int, c_int, c_int, c_int
        ]

        #cxgaig
        self._dll.cig_to_xg_wrapper.argtypes = [
            c_char_p,
            POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
            POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)
        ]

        #cigaxg
        self._dll.cxg_to_ig_wrapper.argtypes = [
            c_char_p,
            POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int),
            POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
        ]

        self._dateo_format = "%m%d%y%H"


    def get_output_step_in_seconds(self):
        raise Exception("Not yet implemented")

    def close(self):
        self._dll.fstfrm_wrapper(self._file_unit)
        self._dll.fclos_wrapper(self._file_unit)
        del self._dll

    def get_number_of_records(self):
        """
        returns number of records inside the rpn file
        """
        return self._dll.fstnbr_wrapper(self._file_unit)

    def get_key_of_any_record(self):
        """
        Returns the key of the first data record, i.e. not >>, ^^ or hy
        """
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

        if key < 0:
            raise Exception("key value is not valid {0}".format(key))

        return key

    def get_3D_record_for_name_and_level(self, varname = '', level = -1,
                                        level_kind = level_kinds.ARBITRARY):
        """
        TODO: add comments
        """
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

        if key < 0: raise Exception('varname = {0}, at level {1} is not found.'.format(varname, level))

        return self._get_data_by_key(key)


    def _get_data_by_key(self, record_key):
        """
        Get data record corresponding to the record_key

        :type record_key: c_int
        """
        self._get_record_info(record_key)
        the_type = self._get_current_data_type()
        ni, nj, nk = self._current_info["shape"]
        data = np.zeros((nk.value * nj.value * ni.value,), dtype = the_type)


        #read the record
        self._dll.fstluk_wrapper(data.ctypes.data_as(POINTER(c_float)), record_key, ni, nj, nk)

        data = np.reshape(data, (ni.value, nj.value, nk.value), order = 'F')
        return data


    def get_longitudes_and_latitudes_for_the_last_read_rec(self):

        """
        finds georeference
        """
        if self._current_info is None:
            raise Exception("No records has been read yet, or its metadata has not yet been saved.")

        ig = self._current_info['ig']


        ni = c_int(0)
        nj = c_int(0)
        nk = c_int(0)
        datev = c_int(-1)
        etiket = create_string_buffer( self.ETIKET_DEFAULT )

        print ig
        ip1, ip2, ip3 = ig[:3]
        in_typvar = create_string_buffer(self.VARTYPE_DEFAULT)


        in_nomvar = create_string_buffer(">>")

        key_hor = self._dll.fstinf_wrapper(self._file_unit, byref(ni), byref(nj), byref(nk), datev, etiket,
                                         ip1, ip2, ip3, in_typvar, in_nomvar
                                        )

        in_nomvar = create_string_buffer("^^")
        key_ver = self._dll.fstinf_wrapper(self._file_unit, byref(ni), byref(nj), byref(nk), datev, etiket,
                                                 ip1, ip2, ip3, in_typvar, in_nomvar
                                                )

        if key_hor < 0 or key_ver < 0:
            raise Exception("key value is not valid {0}".format(min(key_hor, key_ver)))


        lons = self._get_data_by_key(key_hor)[:, 0, 0]
        lats = self._get_data_by_key(key_ver)[0,:, 0]



        ig = self._current_info["ig"]
        n_lons = lons.shape[0]
        n_lats = lats.shape[0]

        grid_type = create_string_buffer(self.current_grid_type)
        grid_reference = create_string_buffer(self.current_grid_reference)

        ezgdef = self._dll.ezgdef_fmem_wrapper(c_int(n_lons), c_int(n_lats),
                            grid_type, grid_reference,
                            ig[0], ig[1], ig[2], ig[3],
                            lons.ctypes.data_as(POINTER(c_float)),
                            lats.ctypes.data_as(POINTER(c_float)))


        the_type = self._get_current_data_type()
        lons_2d = np.zeros((n_lats, n_lons), dtype = the_type)
        lats_2d = np.zeros((n_lats, n_lons), dtype = the_type)

        self._dll.gdll_wrapper(ezgdef, lats_2d.ctypes.data_as(POINTER(c_float)),
                                                   lons_2d.ctypes.data_as(POINTER(c_float)))


        return np.transpose(lons_2d), np.transpose(lats_2d)


        pass

    #get longitudes for the record
    def get_longitudes_and_latitudes(self):
        """
        get longitudes and latitudes of the fields in the rpn file,

        An rpn file can contain several
        """

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



        #read longitudes
        in_nomvar = '>>'
        in_nomvar = create_string_buffer(in_nomvar[:2])
        etiket = create_string_buffer(' ')
        in_typvar = create_string_buffer(' ')

        hor_key = self._dll.fstinf_wrapper(self._file_unit, byref(ni), byref(nj), byref(nk),
                                            datev, etiket, ip1, ip2, ip3, in_typvar, in_nomvar)
        #print in_nomvar.value
        #print in_typvar.value
        assert hor_key >= 0, 'hor_key = {0}'.format(hor_key)

        #print 'hor_key = ', hor_key

        hor_key = c_int(hor_key)
        lons = self._get_data_by_key(hor_key)[:,0,0]


        #read latitudes
        in_nomvar = '^^'
        in_nomvar = create_string_buffer(in_nomvar[:2])
        ver_key = self._dll.fstinf_wrapper(self._file_unit, byref(ni), byref(nj), byref(nk),
                                            datev, etiket, ip1, ip2, ip3, in_typvar, in_nomvar)
        ver_key = c_int(ver_key)
        lats = self._get_data_by_key(ver_key)[0,:,0]


        ig = self._current_info["ig"]
        n_lons = lons.shape[0]
        n_lats = lats.shape[0]

        #print 'grid type: ', self.current_grid_type
        #print 'grid ref: ', self.current_grid_reference


        #print 'grid type: ',  self.current_grid_type
        grid_type = create_string_buffer(self.current_grid_type)
        grid_reference = create_string_buffer(self.current_grid_reference)

        ezgdef = self._dll.ezgdef_fmem_wrapper(c_int(n_lons), c_int(n_lats),
                grid_type, grid_reference,
                ig[0], ig[1], ig[2], ig[3],
                lons.ctypes.data_as(POINTER(c_float)),
                lats.ctypes.data_as(POINTER(c_float)))


        the_type = self._get_current_data_type()
        lons_2d = np.zeros((n_lats, n_lons), dtype = the_type)
        lats_2d = np.zeros((n_lats, n_lons), dtype = the_type)
        
        self._dll.gdll_wrapper(ezgdef, lats_2d.ctypes.data_as(POINTER(c_float)), 
                                       lons_2d.ctypes.data_as(POINTER(c_float)))

        
        return np.transpose(lons_2d), np.transpose(lats_2d)
        

    def get_first_record_for_name(self, varname):
        """
        returns first met record for the field varname
        """
        return self.get_first_record_for_name_and_level(varname, -1)

    def get_first_record_for_name_and_level(self, varname = '', level = -1,
                                                  level_kind = level_kinds.ARBITRARY):
        """
        returns data of the first encountered record that satisfies the
        query, and returns the 2d field, if the rcord is 3d then it takes the 2d subset
        corresponding to the first 3rd dimension
        """
        return self.get_3D_record_for_name_and_level(varname = varname, level = level,
                            level_kind = level_kind)[:,:,0]


    def _get_record_info(self, key, verbose = False):
        """
        store the properties of the record with key, to the internal dictionary
        """
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

        

        if '>>' in nomvar.value or '^^' in nomvar.value:
            self.current_grid_reference = grid_type.value
        else:
            self.current_grid_type = grid_type.value
        #print 'current grid type ', self.current_grid_type

        try:
            dateo_s = "%09d" % dateo.value
            the_dateo = datetime.strptime(dateo_s, self._dateo_format + "{0}".format(dateo.value % 10))
        except Exception:
            print "dateo is corrupted using default: 010100000"
            the_dateo = datetime.strptime("01010000", self._dateo_format)

        if the_dateo.year // 100 != self.start_century:
            year = self.start_century * 100 + the_dateo.year % 100
            the_dateo = datetime( year, the_dateo.month, the_dateo.day, the_dateo.hour, the_dateo.minute)

        result = {'ig': [ig1, ig2, ig3, ig4],
                  'ip': [ip1, ip2, ip3],
                  'shape': [ni, nj, nk],
                  'dateo': the_dateo,
                  'dt_seconds': dt_seconds,
                  'npas': npas,
                  'varname': nomvar,
                  "var_type" : typvar,
                  "data_type" : datyp.value,
                  "nbits" : nbits.value
                  }
        self._current_info = result #update info from the last read record

        #self._set_origin_date(date_o=the_dateo)
        return result



    def _get_current_data_type(self):
        #determine datatype of the data inside the
        data_type = self._current_info["data_type"]
        nbits = self._current_info["nbits"]

        #print data_type, nbits
        if nbits == 32 or nbits == 16:
            return np.float32
        elif nbits == 64:
            return np.float64
        elif nbits == 16:
            return np.float16
        else:
            raise Exception("nbits is: {0}".format(nbits))


    def get_next_record(self):
        """
        returns None, if there is no next record satisfying the last search parameters
        """
        if self._current_info is None:
            key = self.get_key_of_any_record()
            self._get_record_info(key)
        
        [ni, nj, nk] = self._current_info['shape']
        key = self._dll.fstsui_wrapper(self._file_unit, byref(ni), byref(nj), byref(nk))

 
        if key <= 0: return None

        return self._get_data_by_key(key)[:,:,0]

    def get_current_level(self, level_kind = level_kinds.ARBITRARY):
        """
        returns level value for the last read record
        """
        ip1 = self._current_info['ip'][0]
        #print 'ip1 = ', ip1
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
        """
        returns validity date in hours from the simulation start, of the last read record
        return None if no data has been read yet
        """
        if self._current_info is None:
            return None
        return self._current_info['ip'][1].value ##ip2

    def get_dateo_of_last_read_record(self):
        """
        returns date of origin, the start date of the simulation
        """
        if self._current_info:
            return self._current_info["dateo"]
        else:
            raise Exception("No current info has been stored: please make sure you read some records first.")


    def get_datetime_for_the_last_read_record(self):
        """
        returns datetime object corresponding to the last read record
        """


        if not self._current_info is None :
            try:
                forecastHour = self.get_current_validity_date()
                return self._current_info["dateo"] + timedelta(hours = forecastHour)
            except Exception, exc:
                 print exc
                 raise Exception("problem when reading file {0}".format(self.path))
        else:
            raise Exception("No current info has been stored: please make sure you read some records first.")



    def get_4d_field(self, name = "", level_kind = level_kinds.ARBITRARY):
        """
        returns a map
        {t: {z: T(x, y)}}
        """
        result = {}
        data1 = self.get_first_record_for_name(name)

        while data1 is not None:
            level = self.get_current_level(level_kind = level_kind)
            time = self.get_datetime_for_the_last_read_record()

            if not result.has_key(time):
                result[time] = {}

            time_slice = result[time]
            time_slice[level] = data1

            data1 = self.get_next_record()

        return result

        pass


  
    def get_2D_field_on_all_levels(self, name = 'SAND', level_kind = level_kinds.ARBITRARY):
        """
        returns a map {level => 2d field}
        Use this method if you are sure that the field yu want to get has only one record per level
        """
        result = {}
        data1 = self.get_first_record_for_name(name)
        result[self.get_current_level(level_kind = level_kind)] = data1

        while data1 is not None:
            data1 = self.get_next_record()
            if data1 is not None:
                result[self.get_current_level(level_kind = level_kind)] = data1
        return result



    def get_ip1_from_level(self, level, level_kind = level_kinds.ARBITRARY):
        lev = c_float(level)
        lev_kind = c_int(level_kind)
        str_form = create_string_buffer('')
        ip1 = c_int(0)
#        self._dll.convip_wrapper(byref(ip1), byref(lev), byref(lev_kind),
#                            byref(c_int(self.FROM_LEVEL_TO_IP1_MODE)), str_form, byref(c_int(0)))

        return self._dll.ip1_all_wrapper(lev, lev_kind)
#        return int(ip1.value)



    def write_2D_field(self, name = '', level = 1, level_kind = level_kinds.ARBITRARY,
                             data = None, grid_type = 'Z', ig = None, ip = None, typ_var = "P",
                             date = 10160000, label = "soil temp",
                             lon1 = None, lat1 = None,
                             lon2 = None, lat2 = None
                             ):
        """
        Do not care about grid type just write data to the file
        int fstecr_wrapper(float* field, int bits_per_value, int iun,
                              int date, int deet, int npas,
                              int ni, int nj, int nk,
                              int ip1, int ip2, int ip3,
                              char *in_typvar, char *in_nomvar,
                              char *in_etiket, char *in_grtyp,
                              int ig1, int ig2, int ig3, int ig4,
                              int datyp, int rewrite)

        lon1, lat1, lon2, lat2 -are the parameters of the rotated lat lon (used only if grid_type = Z)
        Note: currently the datatypes of the input field is limited to the array of float32

        """
        theData = np.reshape(data, data.size, order = 'F')
        theData = np.array(theData, dtype = np.float32)

        
        nbits = c_int(-32)
        date_c = c_int(date)
        deet = c_int(0)
        npas = c_int(1)
        nk = c_int(1) if len(data.shape) == 2 else data.shape[2]

        if ip is None:
            ip1 = c_int(self.get_ip1_from_level(level, level_kind = level_kind))
            ip2 = c_int(0)
            ip3 = c_int(0)
        else:
            ip1, ip2, ip3 = map(c_int, ip)

        [ni, nj] = data.shape[:2]
        ni = c_int(ni)
        nj = c_int(nj)

        #figure out the ig values
        if None not in [lon1, lat1, lon2, lat2]:
            c_lon1, c_lat1, c_lon2, c_lat2 = map(c_float, [lon1, lat1, lon2, lat2])
            ig1, ig2, ig3, ig4 = map(c_int, [0, 0, 0, 0])
            self._dll.cxg_to_ig_wrapper(c_char_p(grid_type),
                byref(ig1), byref(ig2), byref(ig3), byref(ig4),
                byref(c_lat1), byref(c_lon1), byref(c_lat2), byref(c_lon2)
            )
            print ig1, ig2, ig3, ig4
        elif ig is not None:
            ig1, ig2, ig3, ig4 = map(c_int, ig)
        else:
            ig1, ig2, ig3, ig4 = map(c_int, [0, 0, 0, 0])

        typvar = create_string_buffer(typ_var)
        nomvar = create_string_buffer(name)
        etiket = create_string_buffer(label)
        grtyp = create_string_buffer(grid_type)
        datyp = c_int(data_types.IEEE_floating_point)
        rewrite = c_int(1)




        status = self._dll.fstecr_wrapper( theData.ctypes.data_as(POINTER(c_float)),
                                  nbits, self._file_unit, date_c, deet, npas,
                                  ni, nj, nk,
                                  ip1, ip2, ip3, typvar, nomvar,
                                  etiket, grtyp,
                                  ig1, ig2, ig3, ig4,
                                  datyp, rewrite
                                  )

        #set current info


        the_dateo = datetime.strptime("%08d" % (date // 10), self._dateo_format  )
        self._current_info = {'ig': [ig1, ig2, ig3, ig4],
                  'ip': [ip1, ip2, ip3],
                  'shape': [ni, nj, nk],
                  'dateo': the_dateo,
                  'dt_seconds': deet,
                  'npas': npas,
                  'varname': nomvar,
                  "var_type" : typvar,
                  "data_type" : datyp.value,
                  "nbits" : nbits.value
                  }

        #print 'write status: {0}'.format(status)


        pass


    def get_all_time_records_for_name(self, varname = "STFL"):
        """
        Created for retrieving the fields corresponding to
        different times,
        works as self.get_3D_field, but instead of the map
        {level: 2d record} it returns the map {date: 2d record}
        Use this methond only in the case if you are sure that
        the field you are trying to read containsonly one record for each time step
        """
        result = {}
        data1 = self.get_first_record_for_name(varname)
        result[self.get_datetime_for_the_last_read_record()] = data1

        while data1 is not None:
            data1 = self.get_next_record()
            if data1 is not None:
                result[self.get_datetime_for_the_last_read_record()] = data1
        return result

        pass

    def get_current_info(self):
        """
        return current info of just read record
        """
        return self._current_info


def test():
    #path = 'data/geophys_africa'
    path = 'data/pm1989010100_00000003p'
    rpnObj = RPN(path)
    precip = rpnObj.get_first_record_for_name("PR")

    import matplotlib.pyplot as plt

    plt.imshow(precip.transpose())
    plt.colorbar()
    plt.savefig("precip.png")


    os.system("r.diag ggstat {0}".format(path))

    print "Min = {0}, Mean = {1}, Max = {2}, Var = {3}".format(np.min(precip), np.mean(precip), np.max(precip), np.var(precip))



    datev = rpnObj.get_current_validity_date()
    print 'validity date = ', datev
    print rpnObj._current_info
    print rpnObj.get_number_of_records()
    rpnObj.get_longitudes_and_latitudes()
    rpnObj.close()


def test_get_all_records_for_name():
    path = "data/CORDEX/na/e1/pmNorthAmerica_0.44deg_CanHistoE1_A1950-1954-djf"
    rpnObj = RPN(path=path)
    date_to_field = rpnObj.get_4d_field(name="I0")

    lons2d1, lats2d1 = rpnObj.get_longitudes_and_latitudes_for_the_last_read_rec()
    lons2d2, lats2d2 = rpnObj.get_longitudes_and_latitudes()

    rpnObj.close()

    print np.mean(lons2d1), np.mean(lats2d1)
    print np.mean(lons2d2), np.mean(lats2d2)

    for t, v in date_to_field.iteritems():
        for z, v1 in v.iteritems():
            print t, z, v1.min(), v1.mean(), v1.max()



import application_properties
if __name__ == "__main__":
    application_properties.set_current_directory()
    test()
    test_get_all_records_for_name()
    print "Hello World"
