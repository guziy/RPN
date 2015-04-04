from GChartWrapper.GChart import LineXY

__author__ = 'huziy'

from data import cehq_station
import numpy as np
from GChartWrapper import Line
import application_properties
#adapted from here http://www.portailsig.org/content/python-creation-de-cartes-dynamiques-google-maps-avec-google-charts-kml
def create_kml_file_for_level_stations(data_path = "data/cehq_levels",
                                       kml_file_name = "mon.kml",
                                       title = "Water levels in meters",
                                       icon_color = "ffffccee",
                                       icon_link = "http://dl.dropbox.com/u/4629759/blue-L.png",
                                       data_url_format = "", plot_daily_normals = False, plot_monthly_normals = False
                                       ):
    stations = cehq_station.read_station_data(folder=data_path)

    width = 250
    height = 100
    kmlBody = ("")

    for s in stations:
        assert isinstance(s, cehq_station.Station)
        print(s.id)


        ##Monthly normals
        if plot_monthly_normals:
            values_monthly = s.get_monthly_normals()
            if values_monthly is None:
                print("Skipping {0} since the data series is not continuous enough".format(s.id))
                continue # skip stations with incomplete data
            low = min(values_monthly)
            up = max(values_monthly)
            xy_monthly = Line((values_monthly - low) / (up - low) * 100.0)
            xy_monthly.axes.type("xyx")
            xy_monthly.size(width, height)

            xy_monthly.axes.range(0, 1,12)
            xy_monthly.axes.range(1, low, up)
            xy_monthly.axes.label(2, None, "Month")

        #Daily normals
        if plot_daily_normals:
            times, values_daily = s.get_daily_normals()
            if values_daily is None:
                print("Skipping {0} since the data series is not continuous enough".format(s.id))
                continue
            low = min(values_daily)
            up = max(values_daily)
            xy_daily = Line((values_daily - low) / (up - low) * 100.0)
            xy_daily.axes.type("xyx")
            xy_daily.size(width, height)

            xy_daily.axes.range(0, 1,365)
            xy_daily.axes.range(1, low, up)
            xy_daily.axes.label(2, None, "Day")



        kml = (

        """
            <Placemark>\n
            <name>%s</name>\n
            <Style>
                 <IconStyle>
                     <color>%s</color>
                     <Icon>
                        <href>%s</href>
                     </Icon>
                </IconStyle>


            </Style>

            <description>\n
            <![CDATA[\n
            <p> <b> %s </b>  </p>
            <p> Flow acc. area is %.1f km<sup>2<sup> </p>
            ]]>\n
            </description>\n

            <Point>\n
               <coordinates>%f, %f</coordinates>\n
            </Point>\n
            </Placemark>\n"""
        ) % ( s.id, icon_color, icon_link, title, s.drainage_km2,
               s.longitude, s.latitude)

        kmlBody += kml

    #"morceaux" du fichier KML
    kmlHeader = ('<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n'
                 '<kml xmlns=\"http://earth.google.com/kml/2.2\">\n'
                 '<Document>\n')

    kmlFooter = ('</Document>\n'
                 '</kml>\n')

    kmlFull = kmlHeader + kmlBody + kmlFooter

    open(kml_file_name,'wb').write(kmlFull)

def main():
    create_kml_file_for_level_stations(kml_file_name="level_stations.kml")
    create_kml_file_for_level_stations(data_path="data/cehq_measure_data",
                                       icon_color= "ff11ff22", icon_link= "http://dl.dropbox.com/u/4629759/red-Q.png",
                                       kml_file_name="streamflow_stations.kml",
                                       title="Streamflow m<sup>3</sup>/s"

    )
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print("Hello world")
  