
# convert from cfs to m3/s
from pathlib import Path

M3PERS_IN_CFS = 0.028316847


selected_station_ids_for_streamflow_validation = [
    "05LM006",
    "05AG006",
    "05AK001",
    "05BN012",
    "05CK004",
    "05EF001",
    "06CD002",
    "06EA002",
    "06DA002",
    "05TE002",
    "05TG003",
    "05RB003",
    "05TD001",
    "05JU001",
    "05LC001",
    "05LH005",
    "05102500",
    "05QB003",
    # annual  "05KJ001",
    # annual "05MD004"
]



stations_to_greyout = [
    "05BN012",
    "05AK001",
    "05CK004",
    "05AG006",
    "05LM006",
    "05LH005",
]


MH_UPSTREAM_STATION_BASINS_FOLDER = Path("mh/engage_report/upstream_stations_areas")

upstream_station_boundaries_shp_path = {
    "mh_0.44": MH_UPSTREAM_STATION_BASINS_FOLDER / "mh_0.44" / "basin_boundaries_derived.shp"
}