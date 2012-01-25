__author__ = 'huziy'

import numpy as np

class TimeSeries:
    def __init__(self, data = None, time = None):

        if data is not None and time is not None:
            assert len(data) == len(time)
        else:
            assert [time, data] == [None, None]
        self.data = data
        self.time = time
        pass

    def get_mean(self, months = xrange(1,13)):
        """
        returns mean over the months speciifed in the
        months parameter
        """
        bool_vector = map(lambda x: x.month in months, self.time)
        indices = np.where(bool_vector)[0]
        return np.mean(self.data[indices])



class DateValuePair:
    def __init__(self, date = None, value = None):
        """
        Object for holding corresponding date and value
        pairs
        """
        self.date = date
        self.value = value
        pass


def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  