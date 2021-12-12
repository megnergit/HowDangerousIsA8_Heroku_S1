from pathlib import Path
# =========================================================
#  Authentication
# =========================================================
#
# =========================================================
#  environment
# =========================================================
CSV_DIR = Path('./data/csv')
DATA_DIR = Path('./data/a8u')
SHAPE_DIR = Path('./data/ShapeFile')
RECTANGLE = dict(N=48.44, S=48.16, W=10.55, E=11.68)
OSRM_URL = "http://router.project-osrm.org/route/v1/foot/"
OPEN_ELEVATION_URL = 'https://api.open-elevation.com/api/v1/lookup'
UNFALL_LIST =  ['Unfallorte_2016_LinRef.txt',
               'Unfallorte2017_LinRef.txt',
               'Unfallorte2018_LinRef.txt',
               'Unfallorte2019_LinRef.txt',
               'Unfallorte2020_LinRef.csv']
ST_KEY="moyashi"
NOTE_FILE = './note/summary.yaml'
