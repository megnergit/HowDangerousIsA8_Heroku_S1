import importlib
import re
import os
import pandas as pd
from pathlib import Path
import requests
import polyline
import plotly.graph_objs as go
from tqdm import tqdm
from datetime import datetime
from geopandas.tools import geocode
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import great_circle
import numpy as np
import leafmap.foliumap as folium
from folium.plugins import HeatMap
from folium.features import DivIcon
from folium.map import Marker
import streamlit as st
import yaml
import config
import pdb
#import pretty_errors
# from geopy.point import Point
# import os
# from ast import literal_eval
# from folium import Marker
# import plotly.graph_objects
# import folium
# import geopandas as gpd
# from folium import Circle,  Marker, GeoJson
# import json
# from a8u.a8u import *
# import geopy
# from geopy import distance
importlib.reload(config)
# ==============================================
# house keeping
# ==============================================


def show_all() -> None:
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 99
    pd.options.display.expand_frame_repr = True


def get_lat_lon(landmark='Marienplatz'):
    # Marienplatz
    # (11.57540 48.13714)
    # centroid Munich
    #  11.525549293315658, 48.1548901
    results_geocode = geocode(landmark)
    lon = results_geocode['geometry'].x.values[0]
    lat = results_geocode['geometry'].y.values[0]
    return lat, lon


def coord_o2f(x):
    # object (string) to float

    x = x.replace('(', '')
    x = x.replace(')', '')
    x = x.replace('[', '')
    x = x.replace(']', '')
    c1, c2 = x.split(',')

    return (np.float64(c1), np.float64(c2))

# ==============================================
# pd.Grouper(key='created_at', freq=freq)).count()['confidence_digit']


def measure_xmax(df, df_comp):

    x1 = df['dist_interp_up'].groupby(pd.cut(df['dist_interp_up'], np.arange(
        0, df['dist_interp_up'].max() + 500, 500))).count().max()

    x2 = df_comp['dist_interp_up'].groupby(pd.cut(df_comp['dist_interp_up'], np.arange(
        0, df_comp['dist_interp_up'].max() + 500, 500))).count().max()

    xmax = np.max([x1, x2])

    return xmax


# ==============================================


def load_data(DATA_DIR):
    # unfall data
    df = pd.read_csv(DATA_DIR/'a8u_unfall.csv',
                     converters={'coord': coord_o2f})
    # route data up/down
    df_a8_route_up = pd.read_csv(DATA_DIR/'a8u_route_up.csv',
                                 converters={'coord': coord_o2f})
    df_a8_route_down = pd.read_csv(DATA_DIR/'a8u_route_down.csv',
                                   converters={'coord': coord_o2f})
    # anschlussstelle data
    df_a8_as = pd.read_csv(DATA_DIR/'a8u_as.csv',
                           converters={'coord': coord_o2f,
                                       'coord_up': coord_o2f,
                                       'coord_down': coord_o2f})
    return df, df_a8_route_up, df_a8_route_down, df_a8_as

# ===============================================
# prepare unfall data
# ==============================================


def get_unfall_data():

    CSV_DIR = config.CSV_DIR
    DATA_DIR = config.DATA_DIR
    unfall_list = config.UNFALL_LIST

    # limit the saerch area
    N = config.RECTANGLE['N']
    S = config.RECTANGLE['S']
    W = config.RECTANGLE['W']
    E = config.RECTANGLE['E']

    locator = Nominatim(user_agent='myGeocoer', timeout=60)  # minutes
    rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.002)

    for u in unfall_list:
        df = pd.read_csv(CSV_DIR/u,
                         decimal=',',
                         engine='c',
                         sep=";")

        if 'UIDENTSTLAE' in df.columns:
            df['UIDENTSTLAE'] = df['UIDENTSTLAE'].astype(np.int64)

        if 'UIDENTSTLA' in df.columns:
            df['UIDENTSTLA'] = df['UIDENTSTLA'].astype(np.int64)

        df = df[(df['YGCSWGS84'] >= S) & (df['YGCSWGS84'] <= N) &
                (df['XGCSWGS84'] >= W) & (df['XGCSWGS84'] <= E)].reset_index(drop=True)

        # ----------------------------------
        # this much of data we have near A8
        # ----------------------------------
        print(f'len(df): {len(df)}')

        # -------------------------------------------------------------------
        # Create coordinate, but careful on the order. it is (lat, lon)
        # -------------------------------------------------------------------
        df['coord'] = df['YGCSWGS84'].map(
            str) + ', ' + df['XGCSWGS84'].map(str)

        # -------------------------------------------------------------------
        # use reverse geocoding
        # -------------------------------------------------------------------
        t1 = datetime.now()
        tqdm.pandas(desc='my bar')
        df['location_dict'] = [rgeocode(loc).raw for loc in tqdm(df['coord'])]
        t2 = datetime.now()
        print((t2-t1).total_seconds())
        # set address

        # -------------------------------------------------------------------
        # set address / load  and store then in yeared file
        # -------------------------------------------------------------------
        df['address'] = [loc['display_name'] for loc in df['location_dict']]

        df['road'] = [(loc['address'])['road'] if 'road' in loc['address'].keys()
                      else None for loc in df['location_dict']]

        year = re.findall('[0-9][0-9][0-9][0-9]', u)[0]
        df.to_csv(DATA_DIR/f'a8u_locations_{year:4}.csv', index=False)

    # # convert locally
    # a2u_unfall_list = list(Path(DATA_DIR).glob('a8u_locations_*'))
    # a2u_unfall_list.sort()

    # for a in a2u_unfall_list:
    #     df = pd.read_csv(a)
    #     x = [literal_eval(loc) for loc in df['location_dict']]
    #     df['location_dict'] = x

    #     df['address'] = [loc['display_name'] for loc in df['location_dict']]

    #     df['road'] = [(loc['address'])['road'] if 'road' in loc['address'].keys()
    #                   else None for loc in df['location_dict']]

    #     df.to_csv(a, index=False)

# ------------------------------------------------------------------


def custom_cleaning():
    DATA_DIR = config.DATA_DIR

    a2u_unfall_list = list(Path(DATA_DIR).glob('a8u_locations_*'))
    a2u_unfall_list.sort()

    df_a8_list = []
    for a in a2u_unfall_list:
        print(f'\033[33m{a}\033[0m')
        df = pd.read_csv(a)
        if 'UIDENTSTLA' in df.columns:
            df['UIDENTSTLAE'] = df['UIDENTSTLA'].copy()
            df.drop(['UIDENTSTLA'], axis=1, inplace=True, errors='ignore')

        if 'UIDENTSTLAE' not in df.columns:
            df['UIDENTSTLAE'] = 0

        if 'OBJECTID_1' in df.columns:
            df['OBJECTID'] = df['OBJECTID_1'].copy()
            df.drop(['OBJECTID_1'], axis=1, inplace=True, errors='ignore')

        if 'FID' in df.columns:
            df.drop(['FID'], axis=1, inplace=True, errors='ignore')

        if 'LICHT' in df.columns:
            df['ULICHTVERH'] = df['LICHT'].copy()
            df.drop(['LICHT'], axis=1, inplace=True, errors='ignore')

        if 'IstStrasse' in df.columns:
            df['STRZUSTAND'] = df['IstStrasse'].copy()
            df.drop(['IstStrasse'], axis=1, inplace=True, errors='ignore')

        if 'IstSonstig' in df.columns:
            df['IstSonstige'] = df['IstSonstig'].copy()
            df.drop(['IstSonstig'], axis=1, inplace=True, errors='ignore')

        if 'IstGkfz' not in df.columns:
            df['IstGkfz'] = df['IstSonstige'].copy()
            df.drop(['IstSonstig'], axis=1, inplace=True, errors='ignore')

        df_a8_p1 = df[df['road'].str.contains('A 8', na=False)]
        df_a8_list.append(df_a8_p1)
        df.info()

    df_a8 = pd.concat(df_a8_list, axis=0,
                      ignore_index=True).reset_index(drop=True)
    df_a8.to_csv(DATA_DIR/'a8u_locations.csv', index=False)


# ===============================================
# get the route along A8 [route up]
# ==============================================
def get_a8_route(start_place, end_place, add_address):
    # get the array of coordinates on A8
    # OSRM_URL = "http://router.project-osrm.org/route/v1/foot/"

    OSRM_URL = config.OSRM_URL
    coord_start = np.flip(get_lat_lon(start_place))  # A8 START
    coord_end = np.flip(get_lat_lon(end_place))  # A8 Start

    a = ",".join([str(coord_start[0]), str(coord_start[1])])
    b = ",".join([str(coord_end[0]), str(coord_end[1])])

    QUERY = ";".join([a, b])+"?"
    payload = {'steps': 'true', 'annotations': 'true',
               'overview': 'full', 'geometries': 'polyline6'}
    target = OSRM_URL+QUERY
    # ------------------------------------------------------------
    # acquire array of coordinates
    # ------------------------------------------------------------
    r = requests.get(target, payload)
    a8_coord = polyline.decode(r.json()['routes'][0]
                               ['geometry'], precision=6)

    # ------------------------------------------------------------
    # extract interval -> cummulate to distance
    # ------------------------------------------------------------
    a8_interval = []
    a8_interval.append(0.0)
    a8_interval.extend(r.json()['routes'][0]['legs']
                       [0]['annotation']['distance'])

    a8_dist = np.cumsum(np.array(a8_interval))
    # ------------------------------------------------------------
    # store it in DataFrame
    # ------------------------------------------------------------
    df_a8_route = pd.DataFrame()
    df_a8_route['coord'] = a8_coord
    df_a8_route['interval'] = a8_interval
    df_a8_route['distance'] = a8_dist

    # ------------------------------------------------------------
    # getting address of each point
    # original idea is to automatically extract Anschlusstellen
    # did not work, but left as it is
    # ------------------------------------------------------------

    if add_address:
        try:
            locator = Nominatim(user_agent='myGeocoer', timeout=60)  # minutes
            rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.002)

            t1 = datetime.now()
            tqdm.pandas(desc='reverse geocoding...')
            df_a8_route['location_dict'] = [
                rgeocode(loc).raw for loc in tqdm(df_a8_route['coord'])]
            t2 = datetime.now()
            print((t2-t1).total_seconds())

            # store address
            df_a8_route['address'] = [loc['display_name']
                                      for loc in df_a8_route['location_dict']]

            # store road names. Most of the case, A8
            df_a8_route['road'] = [(loc['address'])['road'] if 'road' in loc['address'].keys()
                                   else None for loc in df_a8_route['location_dict']]

            # store PostLeitZahl
            df_a8_route['PLZ'] = [(loc['address'])['postcode'] if 'postcode'
                                  in loc['address'].keys()
                                  else None for loc in df_a8_route['location_dict']]

        except:
            pass
            # Fine elevation and store it

#        finally:

    elevation = [get_elevation(*loc)
                 for loc in tqdm(df_a8_route['coord'])]
    df_a8_route['elevation'] = elevation
    return df_a8_route

# ==============================================
# annotate
# ----------------------------------------------


def get_elevation(lat, lon):
    #    OPEN_ELEVATION_URL = ('https://api.open-elevation.com/api/v1/lookup'
    #                          f'?locations={lat},{lon}')
    OPEN_ELEVATION_URL = config.OPEN_ELEVATION_URL

    QUERY = f'?locations={lat},{lon}'
    # json object, various ways you can extract value
    r = requests.get(OPEN_ELEVATION_URL+QUERY).json()
    # one approach is to use pandas json functionality:
    elevation = pd.json_normalize(r, 'results')['elevation'].values[0]
    return elevation


# =========================================================
# get the route along A8 [route down]
# =========================================================

def get_dist_interp(u, df_route):

    try:
        dist = np.array(
            [great_circle(u, r).m for r in df_route['coord']])
        ix = dist.argmin()
        if dist[ix-1] < dist[ix+1]:
            ix = ix-1
    #    print(u, ix, ix+1)
        a = great_circle(
            df_route.loc[ix:ix, 'coord'], df_route.loc[ix+1:ix+1, 'coord']).m
        b = great_circle(
            df_route.loc[ix:ix, 'coord'], u).m
        c = great_circle(
            df_route.loc[ix+1:ix+1, 'coord'], u).m

        # squared
        x2 = (b * c / a)**2 * \
            (1.0 - ((b**2 + c**2 - a**2)/(2.0 * b * c))**2)

        # calculate (linearly) interpolated coordinate
        p = (a ** 2 + b**2 - c**2)/(2.0 * a ** 2)  # divide p : 1-p
        q = (a ** 2 + c**2 - b**2)/(2.0 * a ** 2)  # divide p : 1-p

        # coordin

        k1 = df_route.loc[ix:ix, 'coord'].values[0]
        k2 = df_route.loc[ix+1:ix+1, 'coord'].values[0]
        ki = np.array(k1) * p + np.array(k2) * q

    #        print(f'\033[33m{ki_up}\033[0m')
        # calculate (linearly) interpolated distance
        interv = df_route.loc[ix+1:ix+1, 'interval'].values[0]
        dist_prev = df_route.loc[ix:ix, 'distance'].values[0]
        dist_interp = dist_prev + interv * p
    except:
        x2 = None
        ki = None
        dist_interp = None

    # ki : interpolated coordinate
    return x2, ki, dist_interp

# -----------------------------------------------
# return df 'lane', 'coord_interp', 'dist_interp' added


def get_up_down(df, df_up, df_down):
    # ser : df['coord'] or df_a8_as['coord_up']
    lane_list = []
    coord_interp_list = []
    dist_interp_list = []
    dist_interp_up_list = []

    # for u in df['coord']:
    for u in tqdm(df['coord']):
        try:
            # -----------------------------------------------
            # calculate distance to up lane
            # -----------------------------------------------
            x2_up, ki_up, dist_interp_up = get_dist_interp(u, df_up)
            x2_down, ki_down, dist_interp_down = get_dist_interp(u, df_down)

            # ==========================================================
            if x2_up < x2_down:
                lane_list.append('up')
                coord_interp_list.append(ki_up)
                dist_interp_list.append(dist_interp_up)
            else:
                lane_list.append('down')
                coord_interp_list.append(ki_down)
                dist_interp_list.append(dist_interp_down)

            dist_interp_up_list.append(dist_interp_up)
        # ==========================================================
        except:
            lane_list.append(None)
            coord_interp_list.append(None)
            dist_interp_list.append(None)
            dist_interp_up_list.append(None)

    df['lane'] = lane_list
    df['coord_interp'] = coord_interp_list
    df['dist_interp'] = dist_interp_list
    df['dist_interp_up'] = dist_interp_up_list

    return df

# ----------------------------------------------------------
# get anschlussstelle
# ----------------------------------------------------------


def get_a8_as(df_route_up, df_route_down):
    # get anschlussstellen
    DATA_DIR = config.DATA_DIR

    a8_as_list = [['AS 69', 'Burgau', [48.41285, 10.43660],
                   [48.412583593503996, 10.434699552669077], [48.413063837527076, 10.43929773367299]],
                  ['AS 70', 'Zusmarshausen', [48.41000, 10.59717],
                   [48.409861255302694, 10.595931520379006], [48.41013093158816, 10.597929323437466]],
                  ['AS 71a', 'Adelsried', [48.41700, 10.72184],
                   [48.41687447317483, 10.721312289091346], [48.41709181141857, 10.722500076758413]],
                  ['AS 71b', 'Neusäß', [48.41723, 10.83731],
                   [48.417179039129785, 10.836058919133345], [48.41738788028743, 10.837107696698311]],
                  ['AS 72', 'Kreuz Augsburg West', [48.41465, 10.87165],
                   [48.414763411971926, 10.864951498869072], [48.41461556554195, 10.878866314158396]],
                  ['AS 73', 'Augsburg Ost', [48.40986, 10.92021],
                   [48.410740952344064, 10.91391227783138], [48.40889229304123, 10.926903545046596]],
                  ['AS 74a', 'Friedberg', [48.40454, 10.95238],
                   [48.40453883630113, 10.951346147552963], [48.404502031903895, 10.953523815060995]],
                  ['AS 74b', 'Dasing', [48.39336, 11.06657],
                   [48.39256220138096, 11.067762474319794], [48.393793640108974, 11.065996251613223]],
                  ['AS 75', 'Adelzhausen', [48.35209, 11.13426],
                   [48.350157450298916, 11.13613233549958], [48.35083989523587, 11.13587455554466]],
                  ['AS 76', 'Odelzhausen', [48.30878, 11.20594],
                   [48.30962681104632, 11.204408190314677], [48.30796975069872, 11.207412047727908]],
                  ['AS 77', 'Sulzemoos', [48.28080, 11.26999],
                   [48.279503971396984, 11.27233491307027], [48.28154026236177, 11.268778346752253]],
                  ['AS 78', 'Dachau/Füstenfeldbruck', [48.23339, 11.34849],
                   [48.23638267038759, 11.342899002222627], [48.23400745549067, 11.347771009922837]],
                  #              ['AS 79', 'Dreieck Muecnchen Echenried', [48.20774, 11.39199],
                  ['AS 79', 'Dreieck München', [48.20774, 11.39199],
                   [48.21015496157735, 11.387529918279794], [48.207769659945974, 11.392298988301357]],
                  ['AS 80', 'München Langwied', [48.19540, 11.41112],
                   [48.195690779082504, 11.41035980864779], [48.19606995155409, 11.410437968153259]],
                  ['AS 81', 'Kreuz München West', [48.18128, 11.43264],
                   [48.182521912960844, 11.430314727169304], [48.17992527338592, 11.43504032292742]],
                  ['AS 82', 'München Obermenzing', [48.16698436731309, 11.454057501367094],
                   [48.16674572114822, 11.45378123415353], [48.16718033520537, 11.454464287280349]]]

    zeichen, name, coord_middle, coord_up, coord_down = zip(*a8_as_list)
    a8_as = dict(ID=zeichen, name=name, coord=coord_middle,
                 coord_up=coord_up, coord_down=coord_down)

    df_a8_as = pd.DataFrame(a8_as)

    # ----------------------------------------------------------
    x2, ki, dist_interp = zip(*[get_dist_interp(
        u, df_route_up) for u in df_a8_as['coord_up']])
    df_a8_as['dist_up'] = dist_interp

    x2, ki, dist_interp = zip(*[get_dist_interp(
        u, df_route_down) for u in df_a8_as['coord_down']])
    df_a8_as['dist_down'] = dist_interp

#   save anschlussstelle data
    df_a8_as.to_csv(DATA_DIR/'df_a8_as.csv', index=False)

    return df_a8_as

# ----------------------------------------------------------
# visualize histogram in plotly
# ----------------------------------------------------------


def visualize_heatmap(df, df_a8_as):

    DATA_DIR = config.DATA_DIR

    zoom = 9.5
    tiles = 'openstreetmap'
# tiles = 'openstreetmap'
# tiles = 'Stamen Terrain'
# tiles = 'cartodbpositron'
# tiles = 'mapquestopen'
    offset = [-0.05, -0.04]

    lat_list, lon_list = zip(*df['coord'])
    center = [np.mean(np.float64(lat_list)), np.mean(np.float64(lon_list))]
    center = (np.array(center) + np.array(offset)).tolist()
    m_1 = folium.Map(location=center,
                     tiles=tiles,
                     zoom_start=zoom)

    dump = [Marker(c,
                   icon=DivIcon(
                       icon_size=(256, 32),
                       icon_anchor=(0, 36),
                       html=f'<div style="font-size:12pt; font-weight:900; color:gray">{n}</div>',)
                   ).add_to(m_1) for c, n in zip(df_a8_as['coord'], df_a8_as['name'])]

    HeatMap(data=df[['YGCSWGS84', 'XGCSWGS84']],
            min_opacity=0.1,
            radius=15).add_to(m_1)

    outfile = './html/m_1.html'
    m_1.save(outfile)
#    os.system('open '+outfile)

    return m_1


# visualize_heatmap(df, df_a8_as)

# ----------------------------------------------------------


def visualize_strecke(df, df_a8_as, df_elev, size, lane):

    height = size * 3
    width = size * 0.5

    y1 = -0.1
    y2 = 76

    if lane == 'up':
        name = 'up_route'  # name of plot
        lane = 'up'       # use only accident that happnes in up lane
        color = 'coral'   # color
        dist_unfall = 'dist_interp'  # distance to autobahn anschlussstelle
        dist_as = 'dist_up'  # distance to autobahn anschlussstelle
        y_range = [y2, y1]

    elif lane == 'down':
        name = 'down_route'
        lane = 'down'
        color = 'slateblue'
        dist_unfall = 'dist_interp'  # distance to autobahn anschlussstelle
        dist_as = 'dist_down'
        y_range = [y1, y2]

    elif lane == 'down at up':  # to show unfaelle in down lane in
        # the coordinate of up lane
        name = 'down_route'
        lane = 'down'
        color = 'purple'
        dist_unfall = 'dist_interp_up'  # distance to autobahn anschlussstelle
        dist_as = 'dist_up'
        y_range = [y2, y1]

    else:
        name = 'up_route'
        lane = 'up'
        color = 'coral'
        dist_unfall = 'dist_interp'  # distance to autobahn anschlussstelle
        dist_as = 'dist_up'
        y_range = [y2, y1]

    # -------------------------------------------------------------------

    trace1 = go.Scatter(name='dummy',
                        x=df_elev['elevation'],
                        y=df_elev['distance'] / 1000,
                        line=dict(color='darkseagreen', width=0.5),
                        fill='tozerox',
                        fillcolor='rgba(143,188,143, 0.4)',
                        #                        fillcolor='darkseagreen',  # lightsteelblue, #lavender
                        opacity=0.1,
                        orientation='h',
                        xaxis='x',
                        yaxis='y')

    trace2 = go.Histogram(name=name,
                          y=df.loc[df['lane'] == lane, dist_unfall] / 1000,
                          #                     ybins=dict(size=500),
                          marker=dict(color=color),
                          ybins=dict(size=0.5),
                          orientation='h',
                          opacity=0.4,
                          xaxis='x2',
                          yaxis='y2')

# "none" | "tozeroy" | "tozerox" | "tonexty" | "tonextx" | "toself" | "tonext"

    xaxis1 = dict(autorange=False,
                  range=[440, 640],
                  side='top',
                  #                  overlaying='x2',
                  title='Elevation [m]')

    yaxis1 = dict(autorange=False,
                  range=y_range,
                  position=0,
                  #                  overlaying='y',
                  title='[km]',
                  tickfont=dict(size=22),
                  ticksuffix=" ")

    xaxis2 = dict(autorange=False,
                  title='Number of Accidents',
                  overlaying='x',
                  range=[0, 25])

    yaxis2 = dict(autorange=False,
                  range=y_range,
                  anchor='free',
                  side='left',
                  position=0.95,
                  overlaying='y',
                  tickfont=dict(size=24,
                                family='Arial Bold',
                                color='darkgray'),
                  tickmode='array',
                  tickvals=df_a8_as[dist_as] / 1000,
                  ticktext="\u25C0 " + df_a8_as['name'])

    layout = go.Layout(height=height, width=width,
                       showlegend=False,
                       xaxis=xaxis1,
                       xaxis2=xaxis2,
                       yaxis=yaxis1,
                       yaxis2=yaxis2,
                       font=dict(size=20),
                       bargap=0.1,
                       margin=dict(l=0, r=0, t=0, b=0))

    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    return fig
# ----------------------------------------------------------


def visualize_annual(df, df_comp, size):

    y = df.groupby('UJAHR').count()['dist_interp_up']
    x = df.groupby('UJAHR').count().index.to_series()
    trace1 = go.Bar(name='Reference',
                    y=y, x=x,
                    orientation='v',
                    marker_color='coral',
                    opacity=0.4,
                    hoverlabel=dict(font=dict(size=20)))

    y_comp = df_comp.groupby('UJAHR').count()['dist_interp_up']
    x_comp = df_comp.groupby('UJAHR').count().index.to_series()
    trace2 = go.Bar(name='Comparison',
                    y=y_comp, x=x_comp,
                    opacity=0.4,
                    orientation='v',
                    marker_color='slateblue',
                    hoverlabel=dict(font=dict(size=20)))

    data = [trace1, trace2]

    layout = go.Layout(
        legend=dict(orientation='h',
                    x=1, y=1.02,
                    yanchor='bottom', xanchor='right'),

        font=dict(size=20),
        width=size,
        height=size * 0.25,  # width=width,
        margin=dict(l=0, r=0, t=0, b=0),)

    fig = go.Figure(data=data, layout=layout)
    return fig


# ----------------------------------------------------------


def visualize_strecke_comp(df, df_a8_as, df_elev, size, comp, x_range):

    height = size * 2.1
    width = size * 0.5

    y1 = -0.1
    y2 = 76
    name = 'reference'  # name of plot
#    lane = 'up'       # use only accident that happnes in up lane
    dist_unfall = 'dist_interp_up'  # distance to autobahn anschlussstelle
    dist_as = 'dist_up'  # distance to autobahn anschlussstelle
    y_range = [y2, y1]

    if comp:
        ytitle = ''
        color = 'slateblue'  # teal, darkcyan
        showticklabels = False
    else:
        ytitle = '[km]'
        color = 'coral'   # color
        showticklabels = True
        # -------------------------------------------------------------------

    trace1 = go.Scatter(name='dummy',
                        x=df_elev['elevation'],
                        y=df_elev['distance'] / 1000,
                        line=dict(color='darkseagreen', width=0.5),
                        fill='tozerox',
                        fillcolor='rgba(143,188,143, 0.4)',
                        #                        fillcolor='darkseagreen',  # lightsteelblue, #lavender
                        opacity=0.1,
                        orientation='h',
                        xaxis='x',
                        yaxis='y')

    trace2 = go.Histogram(name=name,
                          #                          y=df.loc[df['lane'] == lane, dist_unfall] / 1000,
                          y=df[dist_unfall] / 1000,
                          #                     ybins=dict(size=500),
                          marker=dict(color=color),
                          ybins=dict(size=0.5),
                          orientation='h',
                          opacity=0.4,
                          xaxis='x2',
                          yaxis='y2')

# "none" | "tozeroy" | "tozerox" | "tonexty" | "tonextx" | "toself" | "tonext"

    xaxis1 = dict(autorange=False,
                  range=[440, 640],
                  side='bottom',
                  #                  overlaying='x2',
                  tickfont=dict(size=14),
                  title='Elevation [m]')

    yaxis1 = dict(autorange=False,
                  range=y_range,
                  position=0,
                  #                  overlaying='y',
                  title=ytitle,
                  showticklabels=showticklabels,
                  tickfont=dict(size=22),
                  ticksuffix="  ")

    xaxis2 = dict(autorange=False,
                  side='top',
                  title='Number of Accidents',
                  overlaying='x',
                  range=x_range)
#                  range=[0, 25])

    yaxis2 = dict(autorange=False,
                  range=y_range,
                  anchor='free',
                  side='left',
                  position=0.97,
                  overlaying='y',
                  tickfont=dict(size=24,
                                family='Arial Bold',
                                color='darkgray'),
                  tickmode='array',
                  tickvals=df_a8_as[dist_as] / 1000,
                  ticktext="\u25C0 " + df_a8_as['name'])

    layout = go.Layout(height=height, width=width,
                       showlegend=False,
                       xaxis=xaxis1,
                       xaxis2=xaxis2,
                       yaxis=yaxis1,
                       yaxis2=yaxis2,
                       font=dict(size=20),
                       bargap=0.1,
                       margin=dict(l=0, r=0, t=0, b=0))

    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    return fig

# # ==============================================


def visualize_strecke_overlay(df, df_a8_as, df_elev, size):

    height = size * 3
    width = size * 0.5

    y1 = -0.1
    y2 = 76

    x1 = 0
    x2 = 30

    name = 'up_route'  # name of plot
#    lane = 'up'       # use only accident that happnes in up lane
    color_up = 'coral'   # color
    color_up = 'darkorange'   # chocolate, orange, gold
    color_down = 'slateblue'  # teal, darkcyan
#    color_down = 'cadetblue'
#    color_up = 'orange'   # color
#    color_down = 'slateblue'
#    color_down = 'blue'
    dist_unfall = 'dist_interp'  # distance to autobahn anschlussstelle
    dist_as = 'dist_up'  # distance to autobahn anschlussstelle
    y_range = [y2, y1]
    # -------------------------------------------------------------------
    # background elenvation
    trace1 = go.Scatter(name='dummy',
                        x=df_elev['elevation'],
                        y=df_elev['distance'] / 1000,
                        line=dict(color='darkseagreen', width=0.5),
                        fill='tozerox',
                        fillcolor='rgba(143,188,143, 0.4)',
                        opacity=0.1,
                        orientation='h',
                        xaxis='x',
                        yaxis='y')

    # accidents histogram for up lane
    trace2 = go.Histogram(name=name,
                          y=df.loc[df['lane'] == 'up', dist_unfall] / 1000,
                          marker=dict(color=color_up),
                          ybins=dict(size=0.5),
                          orientation='h',
                          opacity=0.6,
                          xaxis='x2',
                          yaxis='y2')

    # accidents histogram for down lane
    trace3 = go.Histogram(name=name,
                          y=df.loc[df['lane'] == 'down', dist_unfall] / 1000,
                          marker=dict(color=color_down),
                          ybins=dict(size=0.5),
                          orientation='h',
                          opacity=0.6,
                          xaxis='x2',
                          yaxis='y2')

# "none" | "tozeroy" | "tozerox" | "tonexty" | "tonextx" | "toself" | "tonext"

    xaxis1 = dict(autorange=False,
                  range=[440, 640],
                  side='top',
                  #                  overlaying='x2',
                  title='Elevation [m]')

    yaxis1 = dict(autorange=False,
                  range=y_range,
                  position=0,
                  #                  overlaying='y',
                  title='[km]',
                  tickfont=dict(size=22),
                  ticksuffix=" ")

    xaxis2 = dict(autorange=False,
                  title='Number of Accidents',
                  overlaying='x',
                  range=[x1, x2])

    yaxis2 = dict(autorange=False,
                  range=y_range,
                  anchor='free',
                  side='left',
                  position=0.95,
                  overlaying='y',
                  tickfont=dict(size=24,
                                family='Arial Bold',
                                color='darkgray'),
                  tickmode='array',
                  tickvals=df_a8_as[dist_as] / 1000,
                  ticktext="\u25C0 " + df_a8_as['name'])

    layout = go.Layout(height=height, width=width,
                       showlegend=False,
                       xaxis=xaxis1,
                       xaxis2=xaxis2,
                       yaxis=yaxis1,
                       yaxis2=yaxis2,
                       font=dict(size=20),
                       bargap=0.1,
                       barmode='overlay',
                       margin=dict(l=0, r=0, t=0, b=0))

    data = [trace1, trace2, trace3]
    fig = go.Figure(data=data, layout=layout)
    return fig

# =========================================================
# streamlit functions
# =========================================================
# def current_state():
#     st.write(st.session_state['c_up_down'])
#     st.write(st.session_state['c_season'])
#     st.write(st.session_state['c_day_of_week'])
#     st.write(st.session_state['c_hour_of_day'])
#     st.write(st.session_state['c_daylight'])
#     st.write(st.session_state['c_road'])
#     st.write(st.session_state['c_lkw'])


def current_state():
    st.write(st.session_state['up_down'])
    st.write(st.session_state['year'])
    st.write(st.session_state['season'])
    st.write(st.session_state['day_of_week'])
    st.write(st.session_state['hour_of_day'])
    st.write(st.session_state['daylight'])
    st.write(st.session_state['road'])
    st.write(st.session_state['lkw'])
    st.write(st.session_state['up_down'])

# def update_c_up_down():
#     pass
#    st.write('update c_up_down')
#    st.write(st.session_state['up_down'])


def initialize_state():
    if 'year' not in st.session_state:
        st.session_state['year'] = 'all'
    if 'up_down' not in st.session_state:
        #        st.session_state['up_down'] = 'all'
        st.session_state['up_down'] = 'Up to München'
    if 'season' not in st.session_state:
        st.session_state['season'] = 'all'
    if 'day_of_week' not in st.session_state:
        st.session_state['day_of_week'] = 'all'
    if 'hour_of_day' not in st.session_state:
        st.session_state['hour_of_day'] = 'all'
    if 'daylight' not in st.session_state:
        st.session_state['daylight'] = 'all'
    if 'road' not in st.session_state:
        st.session_state['road'] = 'all'
    if 'lkw' not in st.session_state:
        st.session_state['lkw'] = 'all'

    # # ====================================================================
    if 'c_year' not in st.session_state:
        st.session_state['c_year'] = 'all'
    if 'c_up_down' not in st.session_state:
        #        st.session_state['c_up_down'] = 'all'
        st.session_state['c_up_down'] = 'Down from München'
    if 'c_season' not in st.session_state:
        st.session_state['c_season'] = 'all'
    if 'c_day_of_week' not in st.session_state:
        st.session_state['c_day_of_week'] = 'all'
    if 'c_hour_of_day' not in st.session_state:
        st.session_state['c_hour_of_day'] = 'all'
    if 'c_daylight' not in st.session_state:
        st.session_state['c_daylight'] = 'all'
    if 'c_road' not in st.session_state:
        st.session_state['c_road'] = 'all'
    if 'c_lkw' not in st.session_state:
        st.session_state['c_lkw'] = 'all'

#    return state
    # # ====================================================================
    # filtering
    # # ====================================================================
    # control


def filter_data(df, filter_set):
    up_downs, years, seasons, day_of_weeks, hour_of_days, daylights, roads, lkws = filter_set
    df_comp = df.copy()
    # ---------------------------------------------
    if st.session_state['up_down'] == up_downs[1]:
        df = df[df['lane'] == 'up']
    if st.session_state['up_down'] == up_downs[2]:
        df = df[df['lane'] == 'down']
    # ---------------------------------------------
    if st.session_state['year'] == years[1]:
        df = df[df['UJAHR'] == 2020]
    if st.session_state['year'] == years[2]:
        df = df[df['UJAHR'] == 2019]
    if st.session_state['year'] == years[3]:
        df = df[df['UJAHR'] == 2018]
    if st.session_state['year'] == years[4]:
        df = df[df['UJAHR'] == 2017]
    if st.session_state['year'] == years[5]:
        df = df[df['UJAHR'] == 2016]
    # ---------------------------------------------
    if st.session_state['season'] == seasons[1]:
        df = df[df['UMONAT'].isin([11, 12, 1, 2, 3])]
    if st.session_state['season'] == seasons[2]:
        df = df[df['UMONAT'].isin([5, 6, 7, 8, 9])]
    if st.session_state['season'] == seasons[3]:
        df = df[df['UMONAT'].isin([7, 8])]
    # ---------------------------------------------
    if st.session_state['day_of_week'] == day_of_weeks[1]:
        df = df[df['UWOCHENTAG'].isin([2, 3, 4, 5, 6])]
    if st.session_state['day_of_week'] == day_of_weeks[2]:
        df = df[df['UWOCHENTAG'].isin([7, 1])]
    if st.session_state['day_of_week'] == day_of_weeks[3]:
        df = df[df['UWOCHENTAG'].isin([2])]
    # ---------------------------------------------
    if st.session_state['hour_of_day'] == hour_of_days[1]:
        df = df[df['USTUNDE'].isin([6, 7, 8, 9, 10, 11])]
    if st.session_state['hour_of_day'] == hour_of_days[2]:
        df = df[df['USTUNDE'].isin([12, 13, 14, 15, 16, 17])]
    if st.session_state['hour_of_day'] == hour_of_days[3]:
        df = df[df['USTUNDE'].isin([18, 19, 20, 21, 22, 23])]
    if st.session_state['hour_of_day'] == hour_of_days[4]:
        df = df[df['USTUNDE'].isin([0, 1, 2, 3, 4, 5])]
    # ---------------------------------------------
    if st.session_state['daylight'] == daylights[1]:
        df = df[df['ULICHTVERH'] == 0]
    if st.session_state['daylight'] == daylights[2]:
        df = df[df['ULICHTVERH'] == 1]
    if st.session_state['daylight'] == daylights[3]:
        df = df[df['ULICHTVERH'] == 2]
    # ---------------------------------------------
    if st.session_state['road'] == roads[1]:
        df = df[df['STRZUSTAND'] == 0]
    if st.session_state['road'] == roads[2]:
        df = df[df['STRZUSTAND'] == 1]
    if st.session_state['road'] == roads[3]:
        df = df[df['STRZUSTAND'] == 2]
    # ---------------------------------------------
    if st.session_state['lkw'] == lkws[1]:
        df = df[df['IstGkfz'] == 1]

    # =============================================
    if st.session_state['c_up_down'] == up_downs[1]:
        df_comp = df_comp[df_comp['lane'] == 'up']
    if st.session_state['c_up_down'] == up_downs[2]:
        df_comp = df_comp[df_comp['lane'] == 'down']
    # ---------------------------------------------
    if st.session_state['c_year'] == years[1]:
        df_comp = df_comp[df_comp['UJAHR'] == 2020]
    if st.session_state['c_year'] == years[2]:
        df_comp = df_comp[df_comp['UJAHR'] == 2019]
    if st.session_state['c_year'] == years[3]:
        df_comp = df_comp[df_comp['UJAHR'] == 2018]
    if st.session_state['c_year'] == years[4]:
        df_comp = df_comp[df_comp['UJAHR'] == 2017]
    if st.session_state['c_year'] == years[5]:
        df_comp = df_comp[df_comp['UJAHR'] == 2016]
    # ---------------------------------------------
    if st.session_state['c_season'] == seasons[1]:
        df_comp = df_comp[df_comp['UMONAT'].isin([11, 12, 1, 2, 3])]
    if st.session_state['c_season'] == seasons[2]:
        df_comp = df_comp[df_comp['UMONAT'].isin([5, 6, 7, 8, 9])]
    if st.session_state['c_season'] == seasons[3]:
        df_comp = df_comp[df_comp['UMONAT'].isin([7, 8])]
    # ---------------------------------------------
    if st.session_state['c_day_of_week'] == day_of_weeks[1]:
        df_comp = df_comp[df_comp['UWOCHENTAG'].isin([2, 3, 4, 5, 6])]
    if st.session_state['c_day_of_week'] == day_of_weeks[2]:
        df_comp = df_comp[df_comp['UWOCHENTAG'].isin([7, 1])]
    if st.session_state['c_day_of_week'] == day_of_weeks[3]:
        df_comp = df_comp[df_comp['UWOCHENTAG'].isin([2])]
    # ---------------------------------------------
    if st.session_state['c_hour_of_day'] == hour_of_days[1]:
        df_comp = df_comp[df_comp['USTUNDE'].isin([6, 7, 8, 9, 10, 11])]
    if st.session_state['c_hour_of_day'] == hour_of_days[2]:
        df_comp = df_comp[df_comp['USTUNDE'].isin([12, 13, 14, 15, 16, 17])]
    if st.session_state['c_hour_of_day'] == hour_of_days[3]:
        df_comp = df_comp[df_comp['USTUNDE'].isin([18, 19, 20, 21, 22, 23])]
    if st.session_state['c_hour_of_day'] == hour_of_days[4]:
        df_comp = df_comp[df_comp['USTUNDE'].isin([0, 1, 2, 3, 4, 5])]
    # ---------------------------------------------
    if st.session_state['c_daylight'] == daylights[1]:
        df_comp = df_comp[df_comp['ULICHTVERH'] == 0]
    if st.session_state['c_daylight'] == daylights[2]:
        df_comp = df_comp[df_comp['ULICHTVERH'] == 1]
    if st.session_state['c_daylight'] == daylights[3]:
        df_comp = df_comp[df_comp['ULICHTVERH'] == 2]
    # ---------------------------------------------
    if st.session_state['c_road'] == roads[1]:
        df_comp = df_comp[df_comp['STRZUSTAND'] == 0]
    if st.session_state['c_road'] == roads[2]:
        df_comp = df_comp[df_comp['STRZUSTAND'] == 1]
    if st.session_state['c_road'] == roads[3]:
        df_comp = df_comp[df_comp['STRZUSTAND'] == 2]
    # ---------------------------------------------
    if st.session_state['c_lkw'] == lkws[1]:
        df_comp = df_comp[df_comp['IstGkfz'] == 1]

    return df, df_comp
    # # ====================================================================
    # initialize
    # # ====================================================================


def visualize(df, df_comp, df_a8_as, df_a8_route_up, x_range):
    # heatmap
    size = 1024
    m_1 = visualize_heatmap(df, df_a8_as)
    m_2 = visualize_heatmap(df_comp, df_a8_as)

    # # ====================================================================
    # histogram
    size = 1024
#    lane = 'up'  # 'down', 'down at up'
    # fig_control = visualize_strecke(df, df_a8_as, df_a8_route_up, size, lane)
    comp = False
    fig_cntl = visualize_strecke_comp(
        df, df_a8_as, df_a8_route_up, size, comp, x_range)

    # df_a8_route_up['coord']
    # a, b = zip(*a8_coord)
    # histogram
#    lane = 'down'  # 'down', 'down at up'
    comp = True
    # fig_comparison = visualize_strecke(df, df_a8_as, df_a8_route_down, size, lane)
    fig_comp = visualize_strecke_comp(
        df_comp, df_a8_as, df_a8_route_up, size, comp, x_range)

    fig_annual = visualize_annual(df, df_comp, size)

    return fig_cntl, fig_comp, fig_annual, m_1, m_2


# ---------------------------------------------------------

def read_note(NOTE_FILE):
    #    NOTE_FILE = config.NOTE_FILE
    note = None
    with open(NOTE_FILE, 'r') as s:
        try:
            note = yaml.safe_load(s)
        except yaml.YAMLError as e:
            print(e)

    return note

    # =========================================================
