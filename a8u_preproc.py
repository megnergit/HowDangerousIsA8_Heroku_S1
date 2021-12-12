from pathlib import Path
import pandas as pd
import config
from a8u.a8u import *
# import streamlit as st
# from streamlit_autorefresh import st_autorefresh
import importlib
import os
# importlib.reload(config)
# # ====================================================================
# need 3 files
# 1) unfall data
# 2) A8 route data, coordinates and elevation
# 3) A8 Anschluss stelle
# # ====================================================================
# Path('.').cwd()
if __name__ == '__main__':

    #    args = sys.argv

    #     if len(args) == 5:
    #         outfile = args[1]
    #         n_stopper = int(args[2])
    #         t_sleep = int(args[3])
    #         innen = args[4]
    #     else:
    #         print(f'\033[33mUsage : \033[96mpython3 \033[0mpolling_mbs.py \
    # [outfile="tweet.csv"] [n_stopper=100] [t_sleep=1] [innen False]')
    #         exit()

    DATA_DIR = config.DATA_DIR

    # --------------------------------------------------------------------
    # First, route files 2)
    # --------------------------------------------------------------------
    loc1 = 'Zusmarshausen'
#    loc2 = 'Adelsried, Bayern'
    loc2 = 'Schloss Blutenburg, München'
#    loc2 = 'Munchen Schloss Blutenburg'

    df_a8_route_up = get_a8_route(loc1, loc2, add_address=False)
    df_a8_route_up.to_csv(DATA_DIR/'a8u_route_up.csv', index=False)

    df_a8_route_down = get_a8_route(loc2, loc1, add_address=False)
    df_a8_route_down.to_csv(DATA_DIR/'a8u_route_down.csv', index=False)

# --------------------------------------------------------------------
# Second, Anshlussstelle files 3)
# --------------------------------------------------------------------

    df_a8_as = get_a8_as(df_a8_route_up, df_a8_route_down)
    df_a8_as.to_csv(DATA_DIR/'a8u_as.csv', index=False)

# --------------------------------------------------------------------
# Third, Unfall data files 1)
# --------------------------------------------------------------------

#    get_unfall_data()
#    custom_cleaning()
    df_unfall_locations = pd.read_csv(DATA_DIR/'a8u_locations.csv',
                                      converters={'coord': coord_o2f})

# ====================================================================
#  add up/down dist_interp
# ====================================================================

# df_a8_route_up = pd.read_csv(DATA_DIR/'a8u_route_up.csv',
#                               converters={'coord': coord_o2f})
# df_a8_route_down = pd.read_csv(DATA_DIR/'a8u_route_down.csv',
#                               converters={'coord': coord_o2f})

    df = get_up_down(df_unfall_locations, df_a8_route_up, df_a8_route_down)
    df.to_csv(DATA_DIR/'a8u_unfall.csv', index=False)

# # ====================================================================
# df_a8_route_down.info()


# def coord_o2f(ser):
#     # object (string) to float
#     x = [(np.float64(c.split(',')[0]), np.float64(c.split(',')[1]))
#          for c in ser]
#     return pd.Series(x, name=name)

# x = pd.read_csv(DATA_DIR/'a8u_locations.csv',
#                 converters={'coord': coord_o2f})

# df_a8_route_up = pd.read_csv(DATA_DIR/'a8u_route_up.csv',
#                              converters={'coord': coord_o2f})


# # ====================================================================
# CWD = '/Users/meg/git11/ds'
# Path('.').cwd()
# os.chdir(CWD)
# # --------------------------------------------------------------------
# DATA_DIR = config.DATA_DIR
# SHAPE_DIR = config.SHAPE_DIR
# show_all()

# N = 48.44
# S = 48.16
# W = 10.55
# E = 11.68
# # --------------------------------------------------------------------
# unfall_list = ['Unfallorte_2016_LinRef.txt',
#                'Unfallorte2017_LinRef.txt',
#                'Unfallorte2018_LinRef.txt',
#                'Unfallorte2019_LinRef.txt',
#                'Unfallorte2020_LinRef.csv']
# unfall_list = ['Unfallorte2020_LinRef.csv']

# # df = pd.read_csv(DATA_DIR/'Unfallorte2020_LinRef.csv',
# # #                 dtype={'UIDENTSTLAE': np.int64},
# #                  decimal=',',
# #                  engine='c',
# #                  sep=";")

# # x = df.loc[df['coord'] == '48.19313228900006, 11.58407324500007', :]['coord']
# # df.info()
# # df = df[df['ULAND'] == 9].reset_index(drop=True)
# # df.columns

# locator = Nominatim(user_agent='myGeocoer', timeout=60)  # minutes
# rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.002)

# for u in unfall_list:
#     df = pd.read_csv(DATA_DIR/u,
#                      decimal=',',
#                      engine='c',
#                      sep=";")

#     if 'UIDENTSTLAE' in df.columns:
#         df['UIDENTSTLAE'] = df['UIDENTSTLAE'].astype(np.int64)

#     if 'UIDENTSTLA' in df.columns:
#         df['UIDENTSTLA'] = df['UIDENTSTLA'].astype(np.int64)

#     df = df[(df['YGCSWGS84'] >= S) & (df['YGCSWGS84'] <= N) &
#             (df['XGCSWGS84'] >= W) & (df['XGCSWGS84'] <= E)].reset_index(drop=True)

#     print(f'len(df): {len(df)}')
#     df['coord'] = df['YGCSWGS84'].map(str) + ', ' + df['XGCSWGS84'].map(str)

#     t1 = datetime.now()
#     tqdm.pandas(desc='my bar')
#     df['location_dict'] = [rgeocode(loc).raw for loc in tqdm(df['coord'])]
#     t2 = datetime.now()
#     print((t2-t1).total_seconds())
#     # set address
#     df['address'] = [loc['display_name'] for loc in df['location_dict']]

#     df['road'] = [(loc['address'])['road'] if 'road' in loc['address'].keys()
#                   else None for loc in df['location_dict']]

#     year = re.findall('[0-9][0-9][0-9][0-9]', u)[0]
#     df.to_csv(DATA_DIR/f'a8u_locations_{year:4}.csv', index=False)

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

# # ------------------------------------------
# # need intensive clean up for each year
# # df_16 'IstSonstig', 'IstGkfz'
# # df_17 'IstSonstig'
# # df_18 'IstSonstig', 'IstGkfz'
# # okay, 86-87% of Gkfz is GKFZ not Bus or strassenbahn
# # df_17 is assumed all Sonstig is Gkfz

# # -----------------
# df_a8_list = []
# for a in a2u_unfall_list:
#     print(f'\033[33m{a}\033[0m')
#     df = pd.read_csv(a)
#     if 'UIDENTSTLA' in df.columns:
#         df['UIDENTSTLAE'] = df['UIDENTSTLA'].copy()
#         df.drop(['UIDENTSTLA'], axis=1, inplace=True, errors='ignore')

#     if 'UIDENTSTLAE' not in df.columns:
#         df['UIDENTSTLAE'] = 0

#     if 'OBJECTID_1' in df.columns:
#         df['OBJECTID'] = df['OBJECTID_1'].copy()
#         df.drop(['OBJECTID_1'], axis=1, inplace=True, errors='ignore')

#     if 'FID' in df.columns:
#         df.drop(['FID'], axis=1, inplace=True, errors='ignore')

#     if 'LICHT' in df.columns:
#         df['ULICHTVERH'] = df['LICHT'].copy()
#         df.drop(['LICHT'], axis=1, inplace=True, errors='ignore')

#     if 'IstStrasse' in df.columns:
#         df['STRZUSTAND'] = df['IstStrasse'].copy()
#         df.drop(['IstStrasse'], axis=1, inplace=True, errors='ignore')

#     if 'IstSonstig' in df.columns:
#         df['IstSonstige'] = df['IstSonstig'].copy()
#         df.drop(['IstSonstig'], axis=1, inplace=True, errors='ignore')

#     if 'IstGkfz' not in df.columns:
#         df['IstGkfz'] = df['IstSonstige'].copy()
#         df.drop(['IstSonstig'], axis=1, inplace=True, errors='ignore')

#     df_a8_p1 = df[df['road'].str.contains('A 8', na=False)]
#     df_a8_list.append(df_a8_p1)
#     df.info()

# df_a8 = pd.concat(df_a8_list, axis=0, ignore_index=True).reset_index(drop=True)
# df_a8.to_csv(DATA_DIR/'a8u_locations.csv', index=False)

# # =========================================================
# # visualization
# # df_geo = extract_place(df_stx)
# # return m_1

# # =========================================================

# def timeit(method):
#     def timed(*args, **kw):
#         ts = time.time()
#         result = method(*args, **kw)
#         te = time.time()
#         if 'log_time' in kw:
#             name = kw.get('log_name', method.__name__.upper())
#             kw['log_time'][name] = int((te - ts) * 1000)
#         else:
#             #            print '%r  %2.2f ms' % \
#             #                  (method.__name__, (te - ts) * 1000)
#             print(f'{method.__name__:r}, {(te - ts) * 1000:2.2f} ms')
#         return result
#     return timed

# for a in a2u_unfall_list[2:3]:
#     df = pd.read_csv(a)
#     print(f'\033[33m{a}\033[0m')
# #    df.info()
#     len(df)
# #    df[['IstSonstig', 'IstGkfz']]
#     (df['IstSonstig'] == df['IstGkfz']).sum()

#     if 'UIDENTSTLAE' in df.columns:
#         print(len(df), df['UIDENTSTLAE'].nunique())

#     if 'UIDENTSTLA' in df.columns:
#         print(len(df), df['UIDENTSTLA'].nunique())

#     if 'OBJECTID' in df.columns:
#         print(len(df), df['OBJECTID'].nunique())

#     if 'OBJECTID_1' in df.columns:
#         print(len(df), df['OBJECTID_1'].nunique())

#     if 'FID' in df.columns:
#         print(len(df), df['FID'].nunique())

# df_20.info()
# 4879/5569
# 4737/5451

# df_18 = df
# df_16 = df
# df_16.info()

# df_20['STRZUSTAND']
# df_17['USTRZUSTAND']
# df_16['IstStrasse']

# df_17 = df
# df.info()
# df_17.info()

# df_17['LICHT']
# df_17['IstSonstig']

# df['ULICHTVERH']
# #    df.info()

# df.info()
# df['UIDENTSTLAE'].nunique()
# df['OBJECTID'].nunique()
# len(df)

# df_a8.info()
# --------------
# # pick up only A 8
# df_a8_list = []
# for a in a2u_unfall_list:
#     df = pd.read_csv(a)
#     df_a8_p1 = df[df['road'].str.contains('A 8', na=False)]
#     df_a8_list.append(df_a8_p1)

# df_a8 = pd.concat(df_a8_list, axis=0, ignore_index=True).reset_index(drop=True)
# df_a8.info()
# df_a8.head(3)

# # DATA_DIR
# # (t2-t1).min
# # dir(t2-t1)
# # df.to_csv(DATA_DIR/'a8u_address.csv', index=False)

# loc = df.loc[0:0, 'coord'].apply(rgeocode)
# loc = rgeocode(df.loc[0:0, 'coord'])
# loc.raw
# df.loc[0:0, 'address'].raw
