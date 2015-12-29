from __future__ import division, absolute_import

# Standard Library.
import os
import copy
import fnmatch
import warnings
from glob import glob
from io import BytesIO
try:
    from HTMLParser import HTMLParser
except ImportError:
    from html.parser import HTMLParser
from datetime import datetime, timedelta
try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen

# Scientific stack.
import pytz
import numpy as np
from owslib import fes
from owslib.ows import ExceptionReport
from owslib.swe.sensor.sml import SensorML
from pandas import Panel, DataFrame, read_csv, concat
from netCDF4 import Dataset, MFDataset, date2index, num2date

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
           'collector2table',
           'sos_request',
           'get_ndbc_longname',
           'get_coops_metadata',
           'pyoos2df',
           'ndbc2df',
           'nc2df',
           'CF_names',
           'titles',
           'fix_url',
           'fetch_range',
           'start_log',
           'is_station']


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
        model_full_name = cube.attributes.get('title', url)
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
    station = cube.attributes.get('abstract', None)
    if not station:
        station = name.replace('.', '_')

    parser = HTMLParser()
    station = parser.unescape(station)

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


def is_station(url):
    """
    Return True is cdm_data_type exists and is equal to 'station;

    Examples
    --------
    >>> url = ('http://thredds.cdip.ucsd.edu/thredds/dodsC/'
    ...        'cdip/archive/144p1/144p1_historic.nc')
    >>> is_station(url)
    True
    >>> url = ("http://comt.sura.org/thredds/dodsC/data/comt_1_archive/"
    ...        "inundation_tropical/VIMS_SELFE/"
    ...        "Hurricane_Ike_2D_final_run_without_waves")
    >>> is_station(url)
    False

    """
    nc = Dataset(url)
    station = False
    if hasattr(nc, 'cdm_data_type'):
        if nc.cdm_data_type.lower() == 'station':
            station = True
    return station


def _load_nc(nc):
    if isinstance(nc, Dataset):
        return nc
    else:
        return Dataset(nc)


def source_of_data(nc, coverage_content_type='modelResult'):
    """
    Check if the `coverage_content_type` of the cude.
    The `coverage_content_type` is an ISO 19115-1 code to indicating the
    source of the data types and can be one of the following:

    image, thematicClassification, physicalMeasurement, auxiliaryInformation,
    qualityInformation, referenceInformation, modelResult, coordinate

    Examples
    --------
    >>> url = ('http://comt.sura.org/thredds/dodsC/data/comt_1_archive/'
    ...        'inundation_tropical/VIMS_SELFE/'
    ...        'Hurricane_Ike_2D_final_run_without_waves')
    >>> nc = Dataset(url)
    >>> source_of_data(nc)
    True
    >>> url = ('http://thredds.axiomdatascience.com/thredds/'
    ...        'dodsC/G1_SST_GLOBAL.nc')
    >>> source_of_data(url)  # False positive!
    True

    OBS: `source_of_data` assumes that the presence of one
    coverage_content_type` variable means the whole Dataset **is** the same
    `coverage_content_type`!

    """
    nc = _load_nc(nc)
    if nc.get_variables_by_attributes(coverage_content_type=coverage_content_type):  # noqa
        return True
    return False


def is_model(nc):
    """
    Heuristic way to find if a netCDF4 object is "modelResult" or not.
    WARNING: This function may return False positives and False
    negatives!!!

    Examples
    --------
    >>> models = ['http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/sabgom/SABGOM_Forecast_Model_Run_Collection_best.ncd',
    ...           'http://crow.marine.usf.edu:8080/thredds/dodsC/FVCOM-Nowcast-Agg.nc',
    ...           'http://geoport.whoi.edu/thredds/dodsC/coawst_4/use/fmrc/coawst_4_use_best.ncd',
    ...           'http://oos.soest.hawaii.edu/thredds/dodsC/hioos/tide_pac',
    ...           'http://opendap.co-ops.nos.noaa.gov/thredds/dodsC/TBOFS/fmrc/Aggregated_7_day_TBOFS_Fields_Forecast_best.ncd',
    ...           'http://oos.soest.hawaii.edu/thredds/dodsC/pacioos/hycom/global',
    ...           'http://opendap.co-ops.nos.noaa.gov/thredds/dodsC/CBOFS/fmrc/Aggregated_7_day_CBOFS_Fields_Forecast_best.ncd',
    ...           'http://geoport-dev.whoi.edu/thredds/dodsC/estofs/atlantic',
    ...           'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_WAVE_FORECAST.nc',
    ...           'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc']
    >>> all([is_model(url) for url in models])
    True
    >>> not_model = ['http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/043p1/043p1_d17.nc',
    ...              'http://thredds.axiomalaska.com/thredds/dodsC/Aquarius_V3_SSS_Daily.nc',
    ...              'http://thredds.axiomalaska.com/thredds/dodsC/Aquarius_V3_scat_wind_speed_Weekly.nc',
    ...              'http://thredds.axiomdatascience.com/thredds/dodsC/G1_SST_GLOBAL.nc']
    >>> any([is_model(url) for url in not_model])
    False

    """
    nc = _load_nc(nc)

    # First criteria (Strong): `UGRID/SGRID`
    if hasattr(nc, 'Conventions'):
        if 'ugrid' in nc.Conventions.lower():
            return True
    if hasattr(nc, 'Conventions'):
        if 'sgrid' in nc.Conventions.lower():
            return True
    # Second criteria (Strong): dimensionless coords are present.
    vs = nc.get_variables_by_attributes(formula_terms=lambda v: v is not None)
    if vs:
        return True
    # Third criteria (weak): Assumes that all "GRID" attribute are models.
    cdm_data_type = nc.getncattr('cdm_data_type') if hasattr(nc, 'cdm_data_type') else ''  # noqa
    feature_type = nc.getncattr('featureType') if hasattr(nc, 'featureType') else ''  # noqa

    grid, keyword, title = False, False, False

    grid = any([info.lower() == 'grid' for info in [cdm_data_type, feature_type]])  # noqa

    words = ['pom', 'hycom', 'fvcom', 'roms', 'numerical',
             'simulation', 'Circulation Models']
    if hasattr(nc, 'keywords'):
        keyword = any(word in nc.getncattr('keywords') for word in words)
    if hasattr(nc, 'title'):
        title = any(word in nc.getncattr('title') for word in words)

    if any([title, keyword]) and grid:
        return True
    return False


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
    soup = BeautifulSoup(page, "lxml")
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
                kw = dict(subset='index', keep='last')
                df = df.reset_index().drop_duplicates(**kw).set_index('index')
            kw = dict(method='time', limit=30)
            df = df.reindex(index).interpolate(**kw).ix[index]
            dfs.update({model: df})

    return Panel.fromDict(dfs).swapaxes(0, 2)


def fes_date_filter(start, stop, constraint='overlaps'):
    """
    Take datetime-like objects and returns a fes filter for date range
    (begin and end inclusive).
    NOTE: Truncates the minutes!!!

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


def collector2table(collector):
    """
    collector2table return a station table as a DataFrame.
    columns are station, sensor, lon, lat, and the index is the station
    number.

    This is a substitute for `sos_request`.

    """
    # This accepts only 1-day request, but since we only want the
    # stations available we try again with end=start.
    c = copy.copy(collector)
    try:
        response = c.raw(responseFormat="text/csv")
    except ExceptionReport:
        response = c.filter(end=c.start_time).raw(responseFormat="text/csv")
    df = read_csv(BytesIO(response.encode('utf-8')),
                  parse_dates=True)
    columns = {'sensor_id': 'sensor',
               'station_id': 'station',
               'latitude (degree)': 'lat',
               'longitude (degree)': 'lon'}
    df.rename(columns=columns, inplace=True)
    df['sensor'] = [s.split(':')[-1] for s in df['sensor']]
    df['station'] = [s.split(':')[-1] for s in df['station']]

    df = df[['station', 'sensor', 'lon', 'lat']]
    g = df.groupby('station')
    df = dict()
    for station in g.groups.keys():
        df.update({station: g.get_group(station).iloc[0]})
    return DataFrame.from_dict(df).T


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
    soup = BeautifulSoup(r.content, "lxml")
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
    url = ('opendap.co-ops.nos.noaa.gov/ioos-dif-sos/SOS?'
           'service=SOS&'
           'request=DescribeSensor&version=1.0.0&'
           'outputFormat=text/xml;'
           'subtype="sensorML/1.0.1/profiles/ioos_sos/1.0"&'
           'procedure=urn:ioos:station:NOAA.NOS.CO-OPS:%s') % station
    url = parse_url(url)
    xml = etree.parse(urlopen(url))
    root = SensorML(xml)
    if not root.members or len(root.members) > 1:
        msg = "Expected 1 member, got {}".format
        raise ValueError(msg(len(root.members)))
    system = root.members[0]

    # NOTE: Some metadata of interest.
    # system.description
    # short_name = _get_value(system.identifiers, name='shortName')
    # [c.values() for c in system.components]

    long_name = _get_value(system.identifiers, name='longName')
    # FIXME: The new CO-OPS standards sucks!
    long_name = long_name.split('station, ')[-1].strip()
    station_id = _get_value(system.identifiers, name='stationID')

    return long_name, station_id


def pyoos2df(collector, station_id, df_name=None):
    """
    Request CSV response from SOS and convert to Pandas dataframe.

    """
    collector.features = [station_id]
    try:
        response = collector.raw(responseFormat="text/csv")
        kw = dict(parse_dates=True, index_col='date_time')
        df = read_csv(BytesIO(response.encode('utf-8')), **kw)
    except requests.exceptions.ReadTimeout:
        df = ndbc2df(collector, station_id)
    # FIXME: Workaround to get only 1 sensor.
    df = df.reset_index()
    df = df.drop_duplicates(cols='date_time').set_index('date_time')
    if df_name:
        df.name = df_name
    return df


def ndbc2df(collector, ndbc_id):
    """
    Ugly hack because `collector.raw(responseFormat="text/csv")`
    Usually times out.

    """
    # FIXME: Only sea_water_temperature for now.
    if len(collector.variables) > 1:
        msg = "Expected only 1 variables to download, got {}".format
        raise ValueError(msg(collector.variables))
    if collector.variables[0] == 'sea_water_temperature':
        columns = 'sea_water_temperature (C)'
        ncvar = 'sea_surface_temperature'
        data_type = 'stdmet'
        # adcp, adcp2, cwind, dart, mmbcur, ocean, oceansites, pwind,
        # swden, tao-ctd, wlevel, z-hycom
    else:
        msg = "Do not know how to download {}".format
        raise ValueError(msg(collector.variables))

    uri = 'http://dods.ndbc.noaa.gov/thredds/dodsC/data/{}'.format(data_type)
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
    time = num2date(time_dim[:], units=time_dim.units,
                    calendar=kw['calendar'])

    idx_start = date2index(collector.start_time.replace(tzinfo=None),
                           time_dim, **kw)
    idx_stop = date2index(collector.end_time.replace(tzinfo=None),
                          time_dim, **kw)
    if idx_start == idx_stop:
        raise Exception("No data within time range"
                        " {!r} and {!r}".format(collector.start_time,
                                                collector.end_time))
    data = nc.variables[ncvar][idx_start:idx_stop, ...].squeeze()

    time_dim = nc.variables['time']
    time = time[idx_start:idx_stop].squeeze()
    df = DataFrame(data=data, index=time, columns=[columns])
    df.index.name = 'date_time'
    return df


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


def fetch_range(start=datetime(2014, 7, 1, 12), days=6, tzinfo=pytz.utc):
    """
    For hurricane Arthur week use `start=datetime(2014, 7, 0, 12)`.

    """
    start = start.replace(tzinfo=tzinfo)
    stop = start + timedelta(days=days)
    return start, stop


def _reload_log():
    """IPython workaround."""
    import imp
    import logging as log
    imp.reload(log)
    return log


def start_log(start, stop, bbox):
    log = _reload_log()
    import os
    import pyoos
    import owslib

    run_name = '{:%Y-%m-%d}'.format(stop)

    if not os.path.exists(run_name):
        os.makedirs(run_name)
        msg = 'Saving data inside directory {}'.format(run_name)
    else:
        msg = 'Overwriting the data inside directory {}'.format(run_name)

    fmt = '{:*^64}'.format
    log.captureWarnings(True)
    LOG_FILENAME = 'log.txt'
    LOG_FILENAME = os.path.join(run_name, LOG_FILENAME)
    log.basicConfig(filename=LOG_FILENAME,
                    filemode='w',
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%I:%M:%S',
                    level=log.INFO,
                    stream=None)

    log.info(fmt(msg))
    log.info(fmt(' Run information '))
    log.info('Run date: {:%Y-%m-%d %H:%M:%S}'.format(datetime.utcnow()))
    log.info('Download start: {:%Y-%m-%d %H:%M:%S}'.format(start))
    log.info('Download stop: {:%Y-%m-%d %H:%M:%S}'.format(stop))
    log.info('Bounding box: {0:3.2f}, {1:3.2f},'
             '{2:3.2f}, {3:3.2f}'.format(*bbox))
    log.info(fmt(' Software version '))
    log.info('Iris version: {}'.format(iris.__version__))
    log.info('owslib version: {}'.format(owslib.__version__))
    log.info('pyoos version: {}'.format(pyoos.__version__))
    return log

if __name__ == '__main__':
    import doctest
    doctest.testmod()
