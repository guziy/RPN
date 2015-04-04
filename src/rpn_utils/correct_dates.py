from datetime import datetime, timedelta

__author__ = 'huziy'

"""
correct dates based on npas, timestep and start date, by correcting the date of origin and ip2
"""

from rpn.rpn import RPN


def fix_file(path="/RESCUE/skynet3_rech1/huziy/test_export_to_hdf/test/pm1950010100_15802559p",
             leap_year=False, start_date=datetime(1950, 1, 1)):
    r_in = RPN(path=path)
    r_out = RPN(path=path + ".fixed", mode="w")

    data = []
    i = 0
    while data is not None:
        data = r_in.get_next_record()
        if data is None:
            break
        info = r_in.get_current_info()

        nbits = info["nbits"]
        data_type = info["data_type"]

        if nbits > 0:
            nbits = -nbits

        print("nbits = {0}, data_type = {1}".format(nbits, data_type))

        # ips = map(lambda x: x.value, info["ip"])
        ips = info["ip"]

        if leap_year:
            ips[2] = int(info["npas"] * info["dt_seconds"] / 3600)
            new_start_date = start_date
        else:
            # get the start of the current month
            hours_total = int(info["npas"] * info["dt_seconds"] / 3600)
            year = start_date.year + hours_total // (365 * 24)
            print(year)
            d_temp = datetime(2001, 1, 1) + timedelta(days=hours_total % (365 * 24), hours=hours_total % 24)

            new_start_date = datetime(year, d_temp.month, d_temp.day, d_temp.hour)

        r_out.write_2D_field(name=info["varname"],
                             data=data, ip=info["ip"],
                             ig=info["ig"],
                             npas=info["npas"], deet=info["dt_seconds"],
                             label="CORR_DATE", dateo=new_start_date,
                             grid_type=info["grid_type"], typ_var=info["var_type"],
                             nbits=nbits, data_type=data_type)
        i += 1


    # check that all fields were copied
    nrecs_in = r_in.get_number_of_records()
    assert i == nrecs_in, "copied {0} records, but should be {1}".format(i, nrecs_in)

    r_in.close()
    r_out.close()



def main():
    fix_file()

if __name__ == '__main__':
    main()