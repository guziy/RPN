def deg_min_sec_to_deg(d, m, s, hem="N"):

    if hem in ["S", "W"]:
        mul = -1
    else:
        mul = 1

    return mul * (d + m / 60 + s / 3600)