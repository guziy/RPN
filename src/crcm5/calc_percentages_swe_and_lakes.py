__author__ = 'huziy'


def main():
    da_m = 34451.42
    da_o = 33998.0

    dqdt_m_r = 4522067018.18
    dqdt_m_rl = 591028708.454


    d_prod_swe_and_area = 2540759972.76
    gamma = dqdt_m_rl / d_prod_swe_and_area

    print "Gamma = {:.2f}".format(gamma)


    print "Lake influence: {:.1f} %".format((dqdt_m_r - d_prod_swe_and_area) / dqdt_m_r * 100)
    print "SWE influence: {:.1f} %".format(d_prod_swe_and_area / dqdt_m_r * 100)


    print "Lake influence: {:.1f} %".format((dqdt_m_r - dqdt_m_rl) / dqdt_m_r * 100)

if __name__ == "__main__":
    main()