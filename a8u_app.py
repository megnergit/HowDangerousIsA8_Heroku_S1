from pathlib import Path
import pandas as pd
import config
from a8u.a8u import *
from datetime import datetime, timedelta
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yaml
import pdb
import importlib
import os
# Path('.').cwd()
# importlib.reload(config)
# # ====================================================================
# need 3 files
# 1) unfall data
# 2) A8 route data, coordinates and elevation
# 3) A8 Anschluss stelle
# # ====================================================================
DATA_DIR = config.DATA_DIR
ST_KEY = config.ST_KEY
NOTE_FILE = config.NOTE_FILE
# # ====================================================================
# callback function
# # ====================================================================


def show():
    st.set_page_config(layout='wide')
    padding = 2
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)
    # # ====================================================================
    # visualization
    # # ====================================================================
    size = 1024
    # --------------------------------------------
    years = ['all', '2020', '2019', '2018', '2017', '2016']
    up_downs = ['all', 'Up to München', 'Down from München']
    seasons = ['all', 'Winter [Nov-Mar]',
               'Summar [May-Sep]', 'Vacation [Jul-Aug]']
    day_of_weeks = ['all', 'Working Days [Mon-Fri]',
                    'Week End [Sat-Sun]', 'Monday [Mon]']
    hour_of_days = ['all', '6-12', '12-18', '18-24', '24-6']
    daylights = ['all', 'Daylight', 'Twilight', 'Dark']
    roads = ['all', 'Dry', 'Wet', 'Frozen']
    lkws = ['all', 'LkW Involved']
    filter_set = (up_downs, years, seasons, day_of_weeks,
                  hour_of_days, daylights, roads, lkws)
    # --------------------------------------------
    df, df_a8_route_up, df_a8_route_down, df_a8_as = load_data(DATA_DIR)
    initialize_state()
#    current_state()
    df, df_comp = filter_data(df, filter_set)
    xmax = measure_xmax(df, df_comp)
    fig_cntl, fig_comp, fig_annual, m_1, m_2 = visualize(
        df, df_comp, df_a8_as, df_a8_route_up, [0, xmax * 1.2])
    #    state = st.session_state
    #    current_state()
    # --------------------------------------------
    # get note
    # --------------------------------------------
#    print(NOTE_FILE)
#    pdb.set_trace()
    note = read_note(NOTE_FILE)
    # =====================================================================
    # streamlit building
    # =====================================================================
    # autoreload
    count = st_autorefresh(interval=1000 * 3600 * 12, limit=24, key=ST_KEY)
    # --------------------------------------------
    # 1. row : title and headlines
    # --------------------------------------------
    st.title("How Dangerous Is A8?")
    col1, col2, col3 = st.columns((0.48, 0.04, 0.48))
    with col1:
        st.markdown(note['heading'])
        st.markdown('#### Number of Accidents')
        st.plotly_chart(fig_annual, use_container_width=True)
    with col3:
        st.markdown(note['conclusion'])
        st.markdown(note['link'])
    # --------------------------------------------
    # 2. row : cautions
    # --------------------------------------------
    col1, col2, col3 = st.columns((0.26, 0.24, 0.5))
    with col1:
        st.markdown('### Reference')
        st.plotly_chart(fig_cntl, use_container_width=True)

    with col2:
        st.markdown('### Comparison')
        st.plotly_chart(fig_comp, use_container_width=True)

    with col3:
        # ================================
        st.markdown('##### Reference')
        m_1.to_streamlit(height=size * 0.32)
        st.markdown('##### Comparison')
        m_2.to_streamlit(height=size * 0.32)
        st.markdown("""___""")
        # -------------------------------
        st.markdown('##### Reference Sample')
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',
                 unsafe_allow_html=True)
        up_down = st.radio("Up/Down", up_downs, key='up_down')
        year = st.radio("Year", years, key='year')
        season = st.radio("Season", seasons, key='season')
        day_of_week = st.radio("Day of Week", day_of_weeks, key='day_of_week')
        hour_of_day = st.radio("Hour of Day", hour_of_days, key='hour_of_day')
        daylight = st.radio("Daylight", daylights, key='daylight')
        road = st.radio("Road Condition", roads, key='road')
        lkw = st.radio("LkW Involved", lkws, key='lkw')
        st.markdown("""___""")
        # -------------------------------
        st.markdown('##### Comparison Sample')
        c_up_down = st.radio("Up/Down ", up_downs, key='c_up_down')
        c_year = st.radio("Year", years, key='c_year')
        c_season = st.radio("Season", seasons, key='c_season')
        c_day_of_week = st.radio(
            "Day of Week", day_of_weeks, key='c_day_of_week')
        c_hour_of_day = st.radio(
            "Hour of Day", hour_of_days, key='c_hour_of_day')
        c_daylight = st.radio("Daylight", daylights, key='c_daylight')
        c_road = st.radio("Road Condition", roads, key='c_road')
        c_lkw = st.radio("LkW Involved", lkws, key='c_lkw')
#        st.markdown("""___""")

    st.markdown("""___""")


if __name__ == "__main__":
    show()
