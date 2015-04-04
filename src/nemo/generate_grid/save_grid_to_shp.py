from rpn.domains.rotated_lat_lon import RotatedLatLon

__author__ = 'huziy'

from . import nemo_domain_properties

from shapely.geometry import mapping, Polygon
import fiona
from fiona.crs import from_string


def main():
    the_grid = nemo_domain_properties.known_domains["GLK_210x130_0.1deg"]

    polys = []
    dx, dy = the_grid.dx, the_grid.dy
    for i in range(the_grid.nx):
        poly_row = []
        for j in range(the_grid.ny):
            lon0, lat0 = (i - the_grid.iref + 1) * the_grid.dx + the_grid.lonref, \
                         (j - the_grid.jref + 1) * the_grid.dy + the_grid.latref

            poly_row.append(
                Polygon([(lon0 - dx / 2.0, lat0 - dy / 2.0),
                         (lon0 - dx / 2.0, lat0 + dy / 2.0),
                         (lon0 + dx / 2.0, lat0 + dy / 2.0),
                         (lon0 + dx / 2.0, lat0 - dy / 2.0)]))

        polys.append(poly_row)

    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'str'},
    }

    rll = the_grid.rll
    assert isinstance(rll, RotatedLatLon)

    b = rll.get_basemap_params(-20, -20, 20, 20)

    print(dir(b))

    # Write a new Shapefile
    with fiona.open('my_shp2.shp', 'w', 'ESRI Shapefile', schema) as c:
        # If there are multiple geometries, put the "for" loop here
        for row, prow in enumerate(polys):
            for col, poly in enumerate(prow):
                c.write({
                    'geometry': mapping(poly),
                    'properties': {'id': "({},{})".format(row, col)},
                })



if __name__ == '__main__':
    main()
