from nose.tools import ok_

from rpn.rpn import RPN

__author__ = 'huziy'



##TODO: learn how to test properly


class TestRpn(RPN):
    def __init__(self):
        path = "/home/huziy/skynet3_rech1/test/snw_LImon_NA_CRCM5_CanESM2_historical_r1i1p1_185001-200512.rpn"
        RPN.__init__(self, path=path)

    def test_dateo(self):
        """
         Test dateo calculation

        """
        field = self.get_first_record_for_name("I5")
        print(self.get_dateo_of_last_read_record())
        print(self._dateo_to_string(-1274695862))


    def test2(self):
        """
            Test if teardown is not called after each method
        """
        field1 = self.get_first_record_for_name("I5")
        lons, lats = self.get_longitudes_and_latitudes_for_the_last_read_rec()


    def test_get_varnames(self):
        """
            Test if the var names are retreived correctly
        """
        the_names = self.get_list_of_varnames()
        # I know that the current test file should contain the following
        ok_("I5" in the_names)
        ok_(">>" in the_names)
        ok_("^^" in the_names)



    def setup(self):
        print "setting up the object"

    def teardown(self):
        self.close()
        print "tearing down the object"


def setup():
    import application_properties

    application_properties.set_current_directory()
    print "setting up the test suite"


def test_get_records_for_foreacst_hour():
    path = "/home/huziy/skynet3_rech1/test/snw_LImon_NA_CRCM5_CanESM2_historical_r1i1p1_185001-200512.rpn"

    rObj = RPN(path)
    nRecords = rObj.get_number_of_records()

    print nRecords

    #res = rObj.get_records_for_foreacst_hour(var_name="I5", forecast_hour=0)

    ok_(nRecords == 1874, msg="The number of records is not what I've expected")

    #assert_(len(res) == 1, msg="Only one record in the file for the forecast_hour = 0")

    rObj.close()


def teardown():
    print "tearing down the test suite"


