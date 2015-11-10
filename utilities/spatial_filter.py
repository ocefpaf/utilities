import os
import numpy as np
from pyproj import Proj
from matplotlib.path import Path

string = ("+proj=merc +a=6378137 +b=6378137 "
          "+lat_ts=0.0 +lon_0=0.0 +x_0=0.0 "
          "+y_0=0 +k=1.0 +units=m +nadgrids=@null +no_defs")

merc = Proj(string)

rootpath = os.path.split(__file__)[0]
fname = os.path.join(rootpath, "data", "secoora_polygon.csv")
points = np.loadtxt(fname, delimiter=',')
secoora_polygon = Path(points)


def _in_polygon(polygon, xp, yp, transform=None, radius=0.0):
    """
    Check is points `xp` and `yp` are inside the `polygon`.
    Polygon is a `matplotlib.path.Path` object.

    http://stackoverflow.com/questions/21328854/shapely-and-matplotlib-point-in-polygon-not-accurate-with-geolocation

    Examples
    --------
    >>> polygon = Path([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    >>> x1, y1 = 0.5, 0.5
    >>> x2, y2 = 1, 1
    >>> x3, y3 = 0, 1.5
    >>> _in_polygon(polygon, [x1, x2, x3], [y1, y2, y3])
    array([ True, False, False], dtype=bool)


    """
    return polygon.contains_points(np.atleast_2d([xp, yp]).T,
                                   transform=None, radius=0.0)


def in_secoora(lon, lat, transform=None, radius=0.0):
    """
    Examples
    --------
    >>> lons = -84, -80,
    >>> lats = 32, 28
    >>> in_secoora(lons, lats, transform=None, radius=0.0)
    array([ True, False], dtype=bool)

    """
    x, y = merc(lon, lat)
    return _in_polygon(secoora_polygon, x, y, transform=None, radius=0.0)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

"""
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(9, 13),
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax


fig, ax = make_map(projection=ccrs.PlateCarree())
ax.set_extent([-88, -73, 23, 37])
ax.coastlines(resolution='10m')
ax.plot(points[:, 0], points[:, 1])

points = np.array([(-87.88864894047779, 36.90376569037656),
                   (-76.65231475232825, 36.88352004319071),
                   (-76.97624510730193, 36.21541368605749),
                   (-77.48238628694831, 35.54730732892427),
                   (-78.04926440815225, 34.98042920772033),
                   (-78.85909029558645, 34.37305979214468),
                   (-79.46645971116210, 33.92765555405587),
                   (-80.13456606829531, 33.30004049129436),
                   (-80.94439195572951, 32.77365366446213),
                   (-81.57200701849102, 31.96382777702793),
                   (-81.79470913753543, 31.21473883115130),
                   (-81.69348090160615, 30.42515859090295),
                   (-81.57200701849102, 29.65582399784046),
                   (-81.28856795788905, 29.25091105412336),
                   (-80.98488325010123, 28.29936563638817),
                   (-80.70144418949925, 27.77297880955594),
                   (-80.49898771764071, 27.28708327709542),
                   (-80.35726818733972, 26.67971386151977),
                   (-80.33702254015386, 26.19381832905925),
                   (-80.57997030638412, 25.70792279659873),
                   (-80.96463760291537, 25.68767714941287),
                   (-81.51127007693345, 26.41652044810365),
                   (-81.83520043190714, 27.20610068835200),
                   (-82.09839384532325, 27.73248751518423),
                   (-82.32109596436765, 28.31961128357403),
                   (-82.42232420029694, 28.96747199352139),
                   (-82.74625455527062, 29.43312187879605),
                   (-83.13092185180185, 29.81778917532730),
                   (-83.51558914833310, 30.20245647185854),
                   (-83.98123903360777, 30.42515859090295),
                   (-84.50762586044000, 30.30368470778782),
                   (-84.91253880415710, 30.38466729653124),
                   (-85.31745174787420, 30.60736941557564),
                   (-85.86408422189229, 30.83007153462005),
                   (-86.53219057902551, 30.93129977054933),
                   (-87.05857740585774, 30.95154541773518),
                   (-87.58496423268997, 30.95154541773518),
                   (-87.80766635173438, 31.05277365366446),
                   (-87.84815764610608, 34.08962073154271),
                   (-87.88864894047779, 36.90376569037656)])
"""
