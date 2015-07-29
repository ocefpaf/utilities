from __future__ import division, absolute_import

# Standard Library.
import os
import fnmatch
import warnings
from glob import glob
from io import BytesIO
from datetime import datetime, timedelta
try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen

# Scientific stack.
import pytz
import numpy as np
from owslib import fes
from owslib.swe.sensor.sml import SensorML
from pandas import Panel, DataFrame, read_csv, concat
from netCDF4 import MFDataset, date2index, num2date

import iris
from iris.pandas import as_data_frame

import requests
from lxml import etree
from bs4 import BeautifulSoup

# Local.
from .pytools import url_lister, parse_url


iris.FUTURE.netcdf_promote = True
iris.FUTURE.cell_datetime_objects = True

__all__ = ['get_model_name',
           'secoora2df',
           'secoora_buoys',
           'load_secoora_ncs',
           'fes_date_filter',
           'service_urls',
           'sos_request',
           'get_ndbc_longname',
           'get_coops_metadata',
           'coops2df',
           'ndbc2df',
           'nc2df',
           'CF_names',
           'titles',
           'fix_url']


salinity = ['sea_water_salinity',
            'sea_surface_salinity',
            'sea_water_absolute_salinity',
            'sea_water_practical_salinity']

temperature = ['sea_water_temperature',
               'sea_surface_temperature',
               'sea_water_potential_temperature',
               'equivalent_potential_temperature',
               'sea_water_conservative_temperature',
               'pseudo_equivalent_potential_temperature']

water_level = ['sea_surface_height',
               'sea_surface_elevation',
               'sea_surface_height_above_geoid',
               'sea_surface_height_above_sea_level',
               'water_surface_height_above_reference_datum',
               'sea_surface_height_above_reference_ellipsoid']

speed_direction = ['sea_water_speed', 'direction_of_sea_water_velocity']

u = ['surface_eastward_sea_water_velocity',
     'eastward_sea_water_velocity',
     'sea_water_x_velocity',
     'x_sea_water_velocity',
     'eastward_transformed_eulerian_mean_velocity',
     'eastward_sea_water_velocity_assuming_no_tide']

v = ['northward_sea_water_velocity',
     'surface_northward_sea_water_velocity',
     'sea_water_y_velocity',
     'y_sea_water_velocity',
     'northward_transformed_eulerian_mean_velocity',
     'northward_sea_water_velocity_assuming_no_tide']

"""
'surface_geostrophic_sea_water_x_velocity',
'surface_geostrophic_sea_water_y_velocity'
'surface_geostrophic_eastward_sea_water_velocity',
'surface_geostrophic_northward_sea_water_velocity',
'baroclinic_eastward_sea_water_velocity',
'baroclinic_northward_sea_water_velocity',
'barotropic_eastward_sea_water_velocity',
'barotropic_northward_sea_water_velocity',
'barotropic_sea_water_x_velocity',
'barotropic_sea_water_y_velocity',
'bolus_eastward_sea_water_velocity',
'bolus_northward_sea_water_velocity',
'bolus_sea_water_x_velocity',
'bolus_sea_water_y_velocity',
'surface_eastward_geostrophic_sea_water_velocity',
'surface_northward_geostrophic_sea_water_velocity',
'surface_geostrophic_sea_water_x_velocity_assuming_sea_level_for_geoid',
'surface_geostrophic_sea_water_y_velocity_assuming_sea_level_for_geoid',
'surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid',
'surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid',
'surface_eastward_geostrophic_sea_water_velocity_assuming_sea_level_for_geoid',
'surface_northward_geostrophic_sea_water_velocity_assuming_sea_level_for_geoid'
"""

CF_names = dict({'salinity': salinity,
                 'sea_water_temperature': temperature,
                 'currents': dict(u=u, v=v, speed_direction=speed_direction),
                 'water_surface_height_above_reference_datum': water_level})

CSW = {'COMT':
       'comt.sura.org:8000',
       'NGDC Geoportal':
       'http://www.ngdc.noaa.gov/geoportal/csw',
       'USGS WHSC Geoportal':
       'http://geoport.whoi.edu/geoportal/csw',
       'NODC Geoportal: granule level':
       'http://www.nodc.noaa.gov/geoportal/csw',
       'NODC Geoportal: collection level':
       'http://data.nodc.noaa.gov/geoportal/csw',
       'NRCAN CUSTOM':
       'http://geodiscover.cgdi.ca/wes/serviceManagerCSW/csw',
       'USGS Woods Hole GI_CAT':
       'http://geoport.whoi.edu/gi-cat/services/cswiso',
       'USGS CIDA Geonetwork':
       'http://cida.usgs.gov/gdp/geonetwork/srv/en/csw',
       'USGS Coastal and Marine Program':
       'http://cmgds.marine.usgs.gov/geonetwork/srv/en/csw',
       'USGS Woods Hole Geoportal':
       'http://geoport.whoi.edu/geoportal/csw',
       'CKAN testing site for new Data.gov':
       'http://geo.gov.ckan.org/csw',
       'EPA':
       'https://edg.epa.gov/metadata/csw',
       'CWIC':
       'http://cwic.csiss.gmu.edu/cwicv1/discovery'}

titles = dict(SABGOM='http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/'
              'sabgom/SABGOM_Forecast_Model_Run_Collection_best.ncd',
              SABGOM_ARCHIVE='http://omgarch1.meas.ncsu.edu:8080/thredds/'
              'dodsC/fmrc/sabgom/'
              'SABGOM_Forecast_Model_Run_Collection_best.ncd',
              USEAST='http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/'
              'us_east/US_East_Forecast_Model_Run_Collection_best.ncd',
              COAWST_4='http://geoport.whoi.edu/thredds/dodsC/coawst_4/use/'
              'fmrc/coawst_4_use_best.ncd',
              ESPRESSO='http://tds.marine.rutgers.edu/thredds/dodsC/roms/'
              'espresso/2013_da/his_Best/'
              'ESPRESSO_Real-Time_v2_History_Best_Available_best.ncd',
              BTMPB='http://oos.soest.hawaii.edu/thredds/dodsC/hioos/tide_pac',
              TBOFS='http://opendap.co-ops.nos.noaa.gov/thredds/dodsC/TBOFS/'
              'fmrc/Aggregated_7_day_TBOFS_Fields_Forecast_best.ncd',
              HYCOM='http://oos.soest.hawaii.edu/thredds/dodsC/pacioos/hycom/'
              'global',
              CBOFS='http://opendap.co-ops.nos.noaa.gov/thredds/dodsC/CBOFS/'
              'fmrc/Aggregated_7_day_CBOFS_Fields_Forecast_best.ncd',
              ESTOFS='http://geoport-dev.whoi.edu/thredds/dodsC/estofs/'
              'atlantic',
              NECOFS_GOM3_FVCOM='http://www.smast.umassd.edu:8080/thredds/'
              'dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc',
              NECOFS_GOM3_WAVE='http://www.smast.umassd.edu:8080/thredds/dodsC'
              '/FVCOM/NECOFS/Forecasts/NECOFS_WAVE_FORECAST.nc',
              USF_ROMS='http://crow.marine.usf.edu:8080/thredds/dodsC/'
              'WFS_ROMS_NF_model/USF_Ocean_Circulation_Group_West_Florida_'
              'Shelf_Daily_ROMS_Nowcast_Forecast_Model_Data_best.ncd',
              USF_SWAN='http://crow.marine.usf.edu:8080/thredds/dodsC/'
              'WFS_SWAN_NF_model/USF_Ocean_Circulation_Group_West_Florida_'
              'Shelf_Daily_SWAN_Nowcast_Forecast_Wave_Model_Data_best.ncd',
              USF_FVCOM='http://crow.marine.usf.edu:8080/thredds/dodsC/'
              'FVCOM-Nowcast-Agg.nc')


def fix_url(start, url):
    """
    If dates are older than 30 days switch URL prefix to archive.
    NOTE: start must be non-naive datetime object.

    Examples
    --------
    >>> from datetime import datetime
    >>> import pytz
    >>> start = datetime(2010, 1, 1).replace(tzinfo=pytz.utc)
    >>> url = ('http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/'
    ...        'sabgom/SABGOM_Forecast_Model_Run_Collection_best.ncd')
    >>> new_url = fix_url(start, url)
    >>> new_url.split('/')[2]
    'omgarch1.meas.ncsu.edu:8080'

    """
    diff = (datetime.utcnow().replace(tzinfo=pytz.utc)) - start
    if diff > timedelta(days=30):
        url = url.replace('omgsrv1', 'omgarch1')
    return url


def _remove_parenthesis(word):
    """
    Examples
    --------
    >>> _remove_parenthesis("(ROMS)")
    'ROMS'

    """
    try:
        return word[word.index("(") + 1:word.rindex(")")]
    except ValueError:
        return word


def _guess_name(model_full_name):
    """
    Examples
    --------
    >>> some_names = ['USF FVCOM - Nowcast Aggregation',
    ...               'ROMS/TOMS 3.0 - New Floria Shelf Application',
    ...               'COAWST Forecast System : USGS : US East Coast and Gulf'
    ...               'of Mexico (Experimental)',
    ...               'ROMS/TOMS 3.0 - South-Atlantic Bight and Gulf of'
    ...               'Mexico',
    ...               'HYbrid Coordinate Ocean Model (HYCOM): Global',
    ...               'ROMS ESPRESSO Real-Time Operational IS4DVAR Forecast'
    ...               'System Version 2 (NEW) 2013-present FMRC History'
    ...               '(Best)']
    >>> [_guess_name(model_full_name) for model_full_name in some_names]
    ['USF_FVCOM', 'ROMS/TOMS', 'COAWST_USGS', 'ROMS/TOMS', 'HYCOM', \
'ROMS_ESPRESSO']

    """
    words = []
    for word in model_full_name.split():
        if word.isupper():
            words.append(_remove_parenthesis(word))
    mod_name = ' '.join(words)
    if not mod_name:
        mod_name = ''.join([c for c in model_full_name.split('(')[0]
                            if c.isupper()])
    if len(mod_name.split()) > 1:
        mod_name = '_'.join(mod_name.split()[:2])
    return mod_name


def _sanitize(name):
    """
    Examples
    --------
    >>> _sanitize('ROMS/TOMS')
    'ROMS_TOMS'
    >>> _sanitize('USEAST model')
    'USEAST_model'
    >>> _sanitize('GG1SST, SST')
    'GG1SST_SST'

    """
    name = name.replace(', ', '_')
    name = name.replace('/', '_')
    name = name.replace(' ', '_')
    name = name.replace(',', '_')
    return name


def get_model_name(cube, url):
    """
    Return a model short and long name from a cube.

    Examples
    --------
    >>> import iris
    >>> import warnings
    >>> url = ('http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/sabgom/'
    ...        'SABGOM_Forecast_Model_Run_Collection_best.ncd')
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")  # Suppress iris warnings.
    ...     cube = iris.load_cube(url, "sea_water_potential_temperature")
    >>> get_model_name(cube, url)
    ('SABGOM', 'ROMS/TOMS 3.0 - South-Atlantic Bight and Gulf of Mexico')

    """
    url = parse_url(url)
    # [model_full_name]: if there is no title assign the URL.
    try:
        model_full_name = cube.attributes['title']
    except AttributeError:
        model_full_name = url
    # [mod_name]: first searches the titles dictionary, if not try to guess.
    for mod_name, uri in titles.items():
        if url == uri:
            return mod_name, model_full_name
    warnings.warn('Model %s not in the list.  Guessing' % url)
    mod_name = _guess_name(model_full_name)
    mod_name = _sanitize(mod_name)
    return mod_name, model_full_name


def _extract_columns(name, cube):
    """
    Workaround to extract data from a cube and create a dataframe
    following SOS boilerplate.

    """
    try:
        station = cube.attributes['abstract']
    except KeyError:
        station = name.replace('.', '_')
    sensor = 'NA'
    lon = cube.coord(axis='X').points[0]
    lat = cube.coord(axis='Y').points[0]
    time = cube.coord(axis='T')
    time = time.units.num2date(cube.coord(axis='T').points)[0]
    date_time = time.strftime('%Y-%M-%dT%H:%M:%SZ')
    data = cube.data.mean()
    return station, sensor, lat, lon, date_time, data


def secoora2df(buoys, varname):
    """
    This function assumes a global cube object.
    FIXME: Consider removing from the packages and add directly in the
    notebook for clarity.

    """
    secoora_obs = dict()
    for station, cube in buoys.items():
        secoora_obs.update({station: _extract_columns(station, cube)})

    df = DataFrame.from_dict(secoora_obs, orient='index')
    df.reset_index(inplace=True)
    columns = {'index': 'station',
               0: 'name',
               1: 'sensor',
               2: 'lat',
               3: 'lon',
               4: 'date_time',
               5: varname}

    df.rename(columns=columns, inplace=True)
    df.set_index('name', inplace=True)
    return df


def secoora_buoys():
    """
    Returns a generator with secoora catalog_platforms URLs.

    Examples
    ---------
    >>> import types
    >>> from urlparse import urlparse
    >>> buoys = secoora_buoys()
    >>> isinstance(buoys, types.GeneratorType)
    True
    >>> url = list(buoys)[0]
    >>> bool(urlparse(url).scheme)
    True

    """
    thredds = "http://129.252.139.124/thredds/catalog_platforms.html"
    urls = url_lister(thredds)
    base_url = "http://129.252.139.124/thredds/dodsC"
    for buoy in urls:
        if (("?dataset=" in buoy) and
           ('archive' not in buoy) and
           ('usf.c12.weatherpak' not in buoy) and
           ('cormp.ocp1.buoy' not in buoy)):
            try:
                buoy = buoy.split('id_')[1]
            except IndexError:
                buoy = buoy.split('=')[1]
            if buoy.endswith('.nc'):
                buoy = buoy[:-3]
            url = '{}/{}.nc'.format(base_url, buoy)
            yield url


def _secoora_buoys():
    """
    TODO: BeautifulSoup alternative.

    """
    from bs4 import BeautifulSoup
    thredds = "http://129.252.139.124/thredds/catalog_platforms.html"
    connection = urlopen(thredds)
    page = connection.read()
    connection.close()
    soup = BeautifulSoup(page)
    base_url = "http://129.252.139.124/thredds/dodsC"
    for a in soup.find_all("a"):
        href = a.get('href')
        if "?dataset=" in href:
            buoy = a.next_element.string
            url = '{}/{}.nc.html'.format(base_url, buoy)
            yield url


def load_secoora_ncs(run_name):
    """
    Loads local files using the run_name date.
    NOTE: Consider moving this inside the notebook.
    """
    fname = '{}-{}.nc'.format
    OBS_DATA = nc2df(os.path.join(run_name,
                                  fname(run_name, 'OBS_DATA')))
    SECOORA_OBS_DATA = nc2df(os.path.join(run_name,
                                          fname(run_name, 'SECOORA_OBS_DATA')))

    ALL_OBS_DATA = concat([OBS_DATA, SECOORA_OBS_DATA], axis=1)
    index = ALL_OBS_DATA.index

    dfs = dict(OBS_DATA=ALL_OBS_DATA)
    for fname in glob(os.path.join(run_name, "*.nc")):
        if 'OBS_DATA' in fname:
            continue
        else:
            model = fname.split('.')[0].split('-')[-1]
            df = nc2df(fname)
            # FIXME: Horrible work around duplicate times.
            if len(df.index.values) != len(np.unique(df.index.values)):
                kw = dict(cols='index', take_last=True)
                df = df.reset_index().drop_duplicates(**kw).set_index('index')
            kw = dict(method='time', limit=30)
            df = df.reindex(index).interpolate(**kw).ix[index]
            dfs.update({model: df})

    return Panel.fromDict(dfs).swapaxes(0, 2)


def fes_date_filter(start, stop, constraint='overlaps'):
    """
    Take datetime-like objects and returns a fes filter for date range.
    NOTE: Truncates the minutes!!!

    FIXME: Not sure if this is working as expected.
    @rsignell-usgs what are the expected values for within and overlaps?

    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> stop = datetime(2010, 1, 1, 12, 30, 59).replace(tzinfo=pytz.utc)
    >>> start = stop - timedelta(days=7)
    >>> begin, end = fes_date_filter(start, stop, constraint='overlaps')
    >>> begin.literal, end.literal
    ('2010-01-01 12:00', '2009-12-25 12:00')
    >>> begin.propertyoperator, end.propertyoperator
    ('ogc:PropertyIsLessThanOrEqualTo', 'ogc:PropertyIsGreaterThanOrEqualTo')
    >>> begin, end = fes_date_filter(start, stop, constraint='within')
    >>> begin.literal, end.literal
    ('2009-12-25 12:00', '2010-01-01 12:00')
    >>> begin.propertyoperator, end.propertyoperator
    ('ogc:PropertyIsGreaterThanOrEqualTo', 'ogc:PropertyIsLessThanOrEqualTo')

    """
    start = start.strftime('%Y-%m-%d %H:00')
    stop = stop.strftime('%Y-%m-%d %H:00')
    if constraint == 'overlaps':
        propertyname = 'apiso:TempExtent_begin'
        begin = fes.PropertyIsLessThanOrEqualTo(propertyname=propertyname,
                                                literal=stop)
        propertyname = 'apiso:TempExtent_end'
        end = fes.PropertyIsGreaterThanOrEqualTo(propertyname=propertyname,
                                                 literal=start)
    elif constraint == 'within':
        propertyname = 'apiso:TempExtent_begin'
        begin = fes.PropertyIsGreaterThanOrEqualTo(propertyname=propertyname,
                                                   literal=start)
        propertyname = 'apiso:TempExtent_end'
        end = fes.PropertyIsLessThanOrEqualTo(propertyname=propertyname,
                                              literal=stop)
    else:
        raise NameError('Unrecognized constraint {}'.format(constraint))
    return begin, end


def service_urls(records, service='odp:url'):
    """
    Extract service_urls of a specific type (DAP, SOS) from csw records.

    """
    service_string = 'urn:x-esri:specification:ServiceType:' + service
    urls = []
    for key, rec in records.items():
        # Create a generator object, and iterate through it until the match is
        # found if not found, gets the default value (here "none").
        url = next((d['url'] for d in rec.references if
                    d['scheme'] == service_string), None)
        if url is not None:
            urls.append(url)
    urls = sorted(set(urls))
    return urls


def sos_request(url='opendap.co-ops.nos.noaa.gov/ioos-dif-sos/SOS', **kw):
    """
    Examples
    --------
    >>> from urlparse import urlparse
    >>> from datetime import date, datetime, timedelta
    >>> today = date.today().strftime("%Y-%m-%d")
    >>> start = datetime.strptime(today, "%Y-%m-%d") - timedelta(7)
    >>> bbox = [-87.40, 24.25, -74.70, 36.70]
    >>> sos_name = 'water_surface_height_above_reference_datum'
    >>> offering='urn:ioos:network:NOAA.NOS.CO-OPS:WaterLevelActive'
    >>> params = dict(observedProperty=sos_name,
    ...               eventTime=start.strftime('%Y-%m-%dT%H:%M:%SZ'),
    ...               featureOfInterest='BBOX:{0},{1},{2},{3}'.format(*bbox),
    ...               offering=offering)
    >>> uri = 'http://opendap.co-ops.nos.noaa.gov/ioos-dif-sos/SOS'
    >>> url = sos_request(uri, **params)
    >>> bool(urlparse(url).scheme)
    True
    """
    url = parse_url(url)
    offering = 'urn:ioos:network:NOAA.NOS.CO-OPS:CurrentsActive'
    params = dict(service='SOS',
                  request='GetObservation',
                  version='1.0.0',
                  offering=offering,
                  responseFormat='text/csv')
    params.update(kw)
    r = requests.get(url, params=params)
    r.raise_for_status()
    content = r.headers['Content-Type']
    if 'excel' in content or 'csv' in content:
        return r.url
    else:
        raise TypeError('Bad url {}'.format(r.url))


def get_ndbc_longname(station):
    """
    Get long_name for specific station from NOAA NDBC.

    Examples
    --------
    >>> get_ndbc_longname(31005)
    u'Sw Extension'
    >>> get_ndbc_longname(44013)
    u'Boston 16 Nm East Of Boston'

    """
    url = "http://www.ndbc.noaa.gov/station_page.php"
    params = dict(station=station)
    r = requests.get(url, params=params)
    r.raise_for_status()
    soup = BeautifulSoup(r.content)
    # NOTE: Should be only one!
    long_name = soup.findAll("h1")[0]
    long_name = long_name.text.split(' - ')[1].strip()
    long_name = long_name.split(',')[0].strip()
    return long_name.title()


def _get_value(sensor, name='longName'):
    value = None
    sml = sensor.get(name, None)
    if sml:
        value = sml.value
    return value


def get_coops_metadata(station):
    """
    Get longName and sensorName for specific station from COOPS SOS using
    DescribeSensor and owslib.swe.sensor.sml.SensorML.

    Examples
    --------
    >>> long_name, station_id = get_coops_metadata(8651370)
    >>> long_name
    'Duck, NC'
    >>> station_id
    'urn:ioos:station:NOAA.NOS.CO-OPS:8651370'

    """
    url = ('opendap.co-ops.nos.noaa.gov/ioos-dif-sos/SOS?service=SOS&'
           'request=DescribeSensor&version=1.0.0&'
           'outputFormat=text/xml;subtype="sensorML/1.0.1"&'
           'procedure=urn:ioos:station:NOAA.NOS.CO-OPS:%s') % station
    url = parse_url(url)
    xml = etree.parse(urlopen(url))
    root = SensorML(xml)
    if len(root.members) > 1:
        msg = "Expected 1 member, got {}".format
        raise ValueError(msg(len(root.members)))
    system = root.members[0]

    # NOTE:  Some metadata of interest.
    # system.description
    # short_name = _get_value(system.identifiers, name='shortName')
    # [c.values() for c in system.components]

    long_name = _get_value(system.identifiers, name='longName')
    station_id = _get_value(system.identifiers, name='stationID')

    return long_name, station_id


def coops2df(collector, coops_id):
    """
    Request CSV response from SOS and convert to Pandas dataframe.

    """
    collector.features = [coops_id]
    long_name, station_id = get_coops_metadata(coops_id)
    response = collector.raw(responseFormat="text/csv")
    kw = dict(parse_dates=True, index_col='date_time')
    data_df = read_csv(BytesIO(response.encode('utf-8')), **kw)
    data_df.name = long_name
    return data_df


def ndbc2df(collector, ndbc_id):
    """
    Request CSV response from NDBC and convert to Pandas dataframe.

    """
    uri = 'http://dods.ndbc.noaa.gov/thredds/dodsC/data/adcp'
    url = ('%s/%s/' % (uri, ndbc_id))
    urls = url_lister(url)

    filetype = "*.nc"
    file_list = [filename for filename in fnmatch.filter(urls, filetype)]
    files = [fname.split('/')[-1] for fname in file_list]
    urls = ['%s/%s/%s' % (uri, ndbc_id, fname) for fname in files]
    if not urls:
        raise Exception("Cannot find data at {!r}".format(url))
    nc = MFDataset(urls)

    kw = dict(calendar='gregorian', select='nearest')
    time_dim = nc.variables['time']
    dates = num2date(time_dim[:], units=time_dim.units,
                     calendar=kw['calendar'])

    idx_start = date2index(collector.start_time.replace(tzinfo=None),
                           time_dim, **kw)
    idx_stop = date2index(collector.end_time.replace(tzinfo=None),
                          time_dim, **kw)
    if idx_start == idx_stop:
        raise Exception("No data within time range"
                        " {!r} and {!r}".format(collector.start_time,
                                                collector.end_time))
    dir_dim = nc.variables['water_dir'][idx_start:idx_stop, ...].squeeze()
    speed_dim = nc.variables['water_spd'][idx_start:idx_stop, ...].squeeze()
    if dir_dim.ndim != 1:
        dir_dim = dir_dim[:, 0]
        speed_dim = speed_dim[:, 0]
    time_dim = nc.variables['time']
    dates = dates[idx_start:idx_stop].squeeze()
    data = dict()
    data['sea_water_speed (cm/s)'] = speed_dim
    col = 'direction_of_sea_water_velocity (degree)'
    data[col] = dir_dim
    time = dates
    columns = ['sea_water_speed (cm/s)',
               'direction_of_sea_water_velocity (degree)']
    return DataFrame(data=data, index=time, columns=columns)


def nc2df(fname):
    """
    Load a netCDF timeSeries file as a dataframe.

    """
    cube = iris.load_cube(fname)
    for coord in cube.coords(dimensions=[0]):
        name = coord.name()
        if name != 'time':
            cube.remove_coord(name)
    for coord in cube.coords(dimensions=[1]):
        name = coord.name()
        if name != 'station name':
            cube.remove_coord(name)
    df = as_data_frame(cube)
    if cube.ndim == 1:  # Horrible work around iris.
        station = cube.coord('station name').points[0]
        df.columns = [station]
    return df


if __name__ == '__main__':
    import doctest
    doctest.testmod()
