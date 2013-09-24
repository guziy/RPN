__author__ = 'huziy'

import tables as tb

#table for storing projection parameters
projection_table_scheme = {
    "name": tb.StringCol(256),
    "value": tb.FloatCol()
}

