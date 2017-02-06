
from rpn.domains.rotated_lat_lon import RotatedLatLon
from domains.grid_config import GridConfig

# The projection used in the last 2 papers of Huziy's thesis
qc_rll_projection_huziy_thesis = RotatedLatLon(lon1=360 - 68, lat1=52, lon2=16.65, lat2=0)
qc_rll_domain_huziy_thesis = GridConfig(rll=qc_rll_projection_huziy_thesis, iref=142, jref=122, dx=0.1, dy=0.1, xref=180, yref=0, ni=260, nj=260)

qc_rll_domain_huziy_thesis_04 = qc_rll_domain_huziy_thesis.decrease_resolution_keep_free_domain_same(4)


if __name__ == "__main__":
    print(qc_rll_domain_huziy_thesis_04)




