from __future__ import division, absolute_import

# Standard Library.
import os
import signal
import subprocess
from contextlib import contextmanager
try:
    from urllib import urlopen
    from urlparse import urlparse
except ImportError:
    from urllib.request import urlopen
    from urllib.parse import urlparse

# Scientific stack.
import numpy as np
import numpy.ma as ma
from pandas import read_csv
from netCDF4 import Dataset, date2index, num2date


import lxml.html


rootpath = os.path.split(__file__)[0]
df = read_csv(os.path.join(rootpath, 'data', 'climatology_data_sources.csv'))
style = os.path.join(rootpath, 'data', 'style.css')

__all__ = ['rot2d',
           'shrink',
           'get_roms',
           'css_styles',
           'to_html',
           'make_map',
           'inline_map',
           'embed_html',
           'get_coordinates',
           'parse_url',
           'url_lister',
           'time_limit',
           'TimeoutException',
           'make_qr',
           'nbviewer_link']


# ROMS.
def rot2d(x, y, ang):
    """
    Rotate vectors by geometric angle.

    Examples
    --------
    >>> import numpy as np
    >>> x, y = rot2d(1, 0, np.deg2rad(90))
    >>> np.allclose([0, 1], [x, y])
    True

    """
    xr = x * np.cos(ang) - y * np.sin(ang)
    yr = x * np.sin(ang) + y * np.cos(ang)
    return xr, yr


def shrink(a, b):
    """Return array shrunk to fit a specified shape by trimming or averaging.

    a = shrink(array, shape)

    array is an numpy ndarray, and shape is a tuple (e.g., from
    array.shape).  `a` is the input array shrunk such that its maximum
    dimensions are given by shape. If shape has more dimensions than
    array, the last dimensions of shape are fit.

    as, bs = shrink(a, b)

    If the second argument is also an array, both a and b are shrunk to
    the dimensions of each other. The input arrays must have the same
    number of dimensions, and the resulting arrays will have the same
    shape.

    Examples
    --------
    >>> import numpy as np
    >>> rand = np.random.rand
    >>> shrink(rand(10, 10), (5, 9, 18)).shape
    (9, 10)
    >>> map(np.shape, shrink(rand(10, 10, 10), rand(5, 9, 18)))
    [(5, 9, 10), (5, 9, 10)]

    """

    if isinstance(b, np.ndarray):
        if not len(a.shape) == len(b.shape):
            raise Exception('Input arrays must have the same number of'
                            'dimensions')
        a = shrink(a, b.shape)
        b = shrink(b, a.shape)
        return (a, b)

    if isinstance(b, int):
        b = (b,)

    if len(a.shape) == 1:  # 1D array is a special case
        dim = b[-1]
        while a.shape[0] > dim:  # Only shrink a.
            if (dim - a.shape[0]) >= 2:  # Trim off edges evenly.
                a = a[1:-1]
            else:  # Or average adjacent cells.
                a = 0.5*(a[1:] + a[:-1])
    else:
        for dim_idx in range(-(len(a.shape)), 0):
            dim = b[dim_idx]
            a = a.swapaxes(0, dim_idx)  # Put working dim first
            while a.shape[0] > dim:  # Only shrink a
                if (a.shape[0] - dim) >= 2:  # trim off edges evenly
                    a = a[1:-1, :]
                if (a.shape[0] - dim) == 1:  # Or average adjacent cells.
                    a = 0.5*(a[1:, :] + a[:-1, :])
            a = a.swapaxes(0, dim_idx)  # Swap working dim back.
    return a


def get_roms(url, time_slice, n=3):
    url = parse_url(url)
    with Dataset(url) as nc:
        ncv = nc.variables
        time = ncv['ocean_time']
        tidx = date2index(time_slice, time, select='nearest')
        time = num2date(time[tidx], time.units, time.calendar)

        mask = ncv['mask_rho'][:]
        lon_rho = ncv['lon_rho'][:]
        lat_rho = ncv['lat_rho'][:]
        anglev = ncv['angle'][:]

        u = ncv['u'][tidx, -1, ...]
        v = ncv['v'][tidx, -1, ...]

        u = shrink(u, mask[1:-1, 1:-1].shape)
        v = shrink(v, mask[1:-1, 1:-1].shape)

        u, v = rot2d(u, v, anglev[1:-1, 1:-1])

        lon = lon_rho[1:-1, 1:-1]
        lat = lat_rho[1:-1, 1:-1]

        u, v = u[::n, ::n], v[::n, ::n]
        lon, lat = lon[::n, ::n], lat[::n, ::n]

        u = ma.masked_invalid(u)
        v = ma.masked_invalid(v)
    return dict(lon=lon, lat=lat, u=u, v=v, time=time)


# IPython display.
def css_styles(css=style):
    """
    Load css styles.

    Examples
    --------
    >>> from IPython.display import HTML
    >>> html = css_styles()
    >>> isinstance(html, HTML)
    True

    """
    from IPython.display import HTML
    with open(css) as f:
        styles = f.read()
    return HTML('<style>{}</style>'.format(styles))


def to_html(df, css=style):
    """
    Return a pandas table HTML representation with the datagrid css.

    Examples
    --------
    >>> from IPython.display import HTML
    >>> from pandas import DataFrame
    >>> df = DataFrame(np.empty((5, 5)))
    >>> html = to_html(df)
    >>> isinstance(html, HTML)
    True

    """
    from IPython.display import HTML
    with open(css, 'r') as f:
        style = """<style>{}</style>""".format(f.read())
    table = dict(style=style, table=df.to_html())
    return HTML('{style}<div class="datagrid">{table}</div>'.format(**table))


# Mapping
def make_map(bbox, **kw):
    """
    Creates a folium map instance for SECOORA.

    Examples
    --------
    >>> from folium import Map
    >>> bbox = [-87.40, 24.25, -74.70, 36.70]
    >>> m = make_map(bbox)
    >>> isinstance(m, Map)
    True

    """
    from folium import Map

    line = kw.pop('line', True)
    states = kw.pop('states', True)
    layers = kw.pop('layers', True)
    hf_radar = kw.pop('hf_radar', True)
    zoom_start = kw.pop('zoom_start', 5)
    secoora_stations = kw.pop('secoora_stations', True)

    lon, lat = np.array(bbox).reshape(2, 2).mean(axis=0)
    m = Map(width='100%', height='100%',
            location=[lat, lon], zoom_start=zoom_start)

    if hf_radar:
        url = "http://hfrnet.ucsd.edu/thredds/wms/HFRNet/USEGC/6km/hourly/RTV"
        m.add_wms_layer(wms_name="HF Radar",
                        wms_url=url,
                        wms_format="image/png",
                        wms_layers='surface_sea_water_velocity')

    if layers:
        add = 'MapServer/tile/{z}/{y}/{x}'
        base = 'http://services.arcgisonline.com/arcgis/rest/services'
        ESRI = dict(Imagery='World_Imagery/MapServer',
                    Ocean_Base='Ocean/World_Ocean_Base',
                    Topo_Map='World_Topo_Map/MapServer',
                    Street_Map='World_Street_Map/MapServer',
                    Physical_Map='World_Physical_Map/MapServer',
                    Terrain_Base='World_Terrain_Base/MapServer',
                    NatGeo_World_Map='NatGeo_World_Map/MapServer',
                    Shaded_Relief='World_Shaded_Relief/MapServer',
                    Ocean_Reference='Ocean/World_Ocean_Reference',
                    Navigation_Charts='Specialty/World_Navigation_Charts')
        for tile_name, url in ESRI.items():
            tile_url = '{}/{}/{}'.format(base, url, add)
            m.add_tile_layer(tile_name=tile_name,
                             tile_url=tile_url)

    m.add_layers_to_map()
    if line:
        # Create the map and add the bounding box line.
        kw = dict(line_color='#FF0000', line_weight=2)
        m.line(get_coordinates(bbox), **kw)
    if states:
        path = 'https://raw.githubusercontent.com/ocefpaf/secoora/factor_map/'
        path += 'notebooks/secoora.json'
        m.geo_json(geo_path=path,
                   fill_color='none', line_color='Orange')
    if secoora_stations:
        for x, y, name in zip(df['lon'], df['lat'], df['ID']):
            if not np.isnan(x) and not np.isnan(y):
                location = y, x
                popup = '<b>{}</b>'.format(name)
                kw = dict(radius=500, fill_color='#3186cc', popup=popup,
                          fill_opacity=0.2)
                m.circle_marker(location=location, **kw)
    return m


def embed_html(path="mapa.html", width=750, height=500):
    from IPython.display import HTML
    """
    Avoid in-lining the source HTMl into the notebook by adding just a link.
    CAVEAT: All links must be relative!

    Examples
    --------
    >>> html = embed_html(path="./mapa.html")
    >>> isinstance(html, HTML)

    """
    html = ('<iframe src="files/{path}" '
            'style="width: {width}px; height: {height}px;'
            'border: none"></iframe>').format
    return HTML(html(path=path, width=width, height=height))


def inline_map(m):
    """
    Takes a folium instance or a html path and load into an iframe.

    Examples
    --------
    >>> import os
    >>> from IPython.display import HTML, IFrame
    >>> bbox = [-87.40, 24.25, -74.70, 36.70]
    >>> m = make_map(bbox)
    >>> html = inline_map(m)
    >>> isinstance(html, HTML)
    True
    >>> fname = os.path.join('data', 'mapa.html')
    >>> html = inline_map(fname)
    >>> isinstance(html, IFrame)
    True

    """
    from folium import Map
    from IPython.display import HTML, IFrame
    if isinstance(m, Map):
        m._build_map()
        srcdoc = m.HTML.replace('"', '&quot;')
        embed = HTML('<iframe srcdoc="{srcdoc}" '
                     'style="width: 100%; height: 500px; '
                     'border: none"></iframe>'.format(srcdoc=srcdoc))
    elif isinstance(m, str):
        embed = IFrame(m, width=750, height=500)
    return embed


def get_coordinates(bbox):
    """
    Create bounding box coordinates for the map.  It takes flat or
    nested list/numpy.array and returns 5 points that closes square
    around the borders.

    Examples
    --------
    >>> bbox = [-87.40, 24.25, -74.70, 36.70]
    >>> len(get_coordinates(bbox))
    5

    """
    bbox = np.asanyarray(bbox).ravel()
    if bbox.size == 4:
        bbox = bbox.reshape(2, 2)
        coordinates = []
        coordinates.append([bbox[0][1], bbox[0][0]])
        coordinates.append([bbox[0][1], bbox[1][0]])
        coordinates.append([bbox[1][1], bbox[1][0]])
        coordinates.append([bbox[1][1], bbox[0][0]])
        coordinates.append([bbox[0][1], bbox[0][0]])
    else:
        raise ValueError('Wrong number corners.'
                         '  Expected 4 got {}'.format(bbox.size))
    return coordinates


# Web-parsing.
def parse_url(url):
    """
    This will preserve any given scheme but will add http if none is
    provided.

    Examples
    --------
    >>> parse_url('www.google.com')
    'http://www.google.com'
    >>> parse_url('https://www.google.com')
    'https://www.google.com'

    """
    if not urlparse(url).scheme:
        url = "http://{}".format(url)
    return url


def url_lister(url):
    """
    Extract all href links from a given URL.

    """
    urls = []
    connection = urlopen(url)
    dom = lxml.html.fromstring(connection.read())
    for link in dom.xpath('//a/@href'):
        urls.append(link)
    return urls


def nbviewer_link(notebook):
    """
    Return a nbviewer link for a given notebook in the current
    repository.

    """
    # User and repository names.
    out = subprocess.Popen(['git', 'remote', 'show', 'origin', '-n'],
                           stdout=subprocess.PIPE).stdout.read().decode()
    out = out.split('\n')
    out = [l.strip().split(':')[-1] for l in out if
           l.strip().startswith('Fetch')]
    user, repo = out[0].split('/')
    repo = repo.split('.git')[0]
    # Branch name.
    out = subprocess.Popen(['git', 'branch'],
                           stdout=subprocess.PIPE).stdout.read().decode()
    out = out.split('\n')
    branch = [l.split()[-1] for l in out if l.strip().startswith('*')][0]
    # Path
    path = os.path.abspath(notebook)
    path = ''.join(path.split(repo, 1)[-1])
    # URL.
    params = dict(user=user,
                  repo=repo,
                  branch=branch,
                  path=path)
    url = ('http://nbviewer.ipython.org/github/'
           '{user}/{repo}/blob/{branch}{path}').format
    return url(**params)


# Misc.
@contextmanager
def time_limit(seconds=10):
    """
    Raise a TimeoutException after n `seconds`.

    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class TimeoutException(Exception):
    """
    Timeout Exception.

    Example
    -------
    >>> def long_function_call():
    ...     import time
    ...     sec = 0
    ...     while True:
    ...         sec += 1
    ...         time.sleep(1)
    >>> try:
    ...     with time_limit(3):
    ...         long_function_call()
    ... except TimeoutException as msg:
    ...     print('{!r}'.format(msg))
    TimeoutException('Timed out!',)
    """
    pass


def make_qr(text):
    import qrcode
    qr = qrcode.QRCode(version=1,
                       error_correction=qrcode.constants.ERROR_CORRECT_L,
                       box_size=10, border=4)
    qr.add_data(text)
    qr.make(fit=True)
    return qr.make_image()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
