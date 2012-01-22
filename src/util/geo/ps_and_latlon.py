__author__="huziy"
__date__ ="$15 sept. 2010 19:05:55$"

from math import *

def psxy2latlon(x, y):
    '''
    Convert ps indices to lat lon, x and y
    are relative indices to Pole
    '''
    C1 = 180.0 / pi;
    TRUE_LAT = 60;
    RE = 6.371e6 * (1 + sin(TRUE_LAT/C1));
    RE2 = pow(RE, 2);
    XAXIS = 25; D60 = 45000.0;
    SOUTHERN_HEM = 2;
    NORTHERN_HEM = 1;
    XHEM = NORTHERN_HEM;
    x = float(x)
    y = float(y)
    if x == 0 and y == 0:
        lat = 90;
        lon = 0;
    else:
        if x == 0:
            lon = y / abs(y) * 90.0;
        elif x > 0:
            lon = atan(y/x)*C1;
        elif x < 0:
            lon = atan(y/x)*C1 + y / abs(y) * 180.0;

        lon = lon - XAXIS;       #   /*Longitude*/
##        if lon < 0:
##           lon = lon + 360;

        R2 = pow(x*D60,2)+pow(y*D60,2);  #/*Latitude*/
        ADLAT= (RE2-R2)/(RE2+R2);        #/*Latitude*/
        lat = asin(ADLAT)*C1;            #/*Latitude*/

    if XHEM == SOUTHERN_HEM:
        lat = -lat;
        lon = -lon;
    return lat, lon


##% /*LL2IJ give the position of a point  in the PS-GRID in relative units     */
##% /*The position is calculated from the North Pole (0,0)                     */
##%
##% 49.65698 N, 96.99443 W
def latlon2psxy(lat, lon):
    '''
    Converts lat lon to ps indices relative to the Pole
    '''
    TRUE_LAT = 60.0;
    RE = 6.371e6 * (1 + sin(TRUE_LAT/180.0 * pi));
    D60 = 45000.0;
    XAXIS = 25;
    ANG = pi * ( 90 - lat ) / 360.0;
    R = RE * tan( ANG ) / D60;
    ANG = pi*(-lon -XAXIS) / 180.0;
    x = R*cos(ANG);
    y = -R*sin(ANG);
    return x,y



if __name__ == "__main__":
    d = 45000.0
    lat, lon = psxy2latlon(10.0 / d,0.0 / d)
    
    print lon , lat

    x, y = latlon2psxy(lat, lon)
    print x * d, y * d


    print "Hello World"
