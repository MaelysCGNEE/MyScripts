# Standard Libraries
import os  # Interactions avec le système de fichiers
import logging  # Gestion des logs
import json  # Manipulation de fichiers JSON
import math  # Fonctions mathématiques de base
import cmath  # Fonctions mathématiques complexes
import statistics  # Statistiques de base
import time  # Gestion du temps
import warnings  # Gestion des avertissements
import itertools  # Outils pour les itérations et les combinaisons
import glob  # Recherche de fichiers
import base64  # Encodage/décodage en base64
import importlib  # Chargement dynamique de modules
from datetime import datetime as dt, timedelta  # Gestion des dates et heures
from dateutil.relativedelta import relativedelta  # Calcul sur des dates avec des décalages relatifs
import calendar  # Gestion des calendriers

# Data Handling
import pandas as pd  # Manipulation de données tabulaires
import numpy as np  # Calculs numériques
import ast  # Conversion de chaînes en objets Python
import humanfriendly  # Gestion des durées et tailles humanisables

# HTTP Requests
import requests  # Requêtes HTTP

# Geolocation
from geopy.geocoders import Nominatim  # Géolocalisation

# Climate Data Store APIs and Grib Files
import cdsapi  # Client API de Climate Data Store
import cfgrib  # Lecture de fichiers GRIB

# Machine Learning & Statistics
from sklearn import linear_model, metrics  # Modèles linéaires et évaluation de performances
from sklearn.linear_model import LinearRegression  # Régression linéaire
from sklearn.model_selection import train_test_split, GridSearchCV  # Partitionnement et recherche d'hyperparamètres
from sklearn.ensemble import RandomForestRegressor  # Modèles d'ensemble : forêts aléatoires
from sklearn.preprocessing import MinMaxScaler  # Mise à l'échelle des données
from sklearn.pipeline import Pipeline  # Pipelines de modélisation
import joblib  # Sauvegarde des modèles entraînés
import scipy.stats  # Tests statistiques

# Change Point Detection
import ruptures as rpt  # Détection des points de rupture dans les séries temporelles

# Visualization
import plotly.graph_objs as go  # Objets graphiques de Plotly
import plotly.express as px  # API de visualisation simplifiée de Plotly
from plotly.subplots import make_subplots  # Création de sous-graphiques dans Plotly

# Dash Framework for Web Apps
import dash  # Framework Dash
from dash import dcc, html  # Composants Dash Core et HTML
from dash.dependencies import Input, Output, State  # Gestion des dépendances dans Dash
from dash import dash_table  # Tableau interactif Dash

# Natural Sorting
from natsort import index_natsorted, natsorted, natsort_keygen  # Tri naturel

# Additional Libraries for Geolocation Calculations
from math import radians, degrees  # Conversion d'angles
from cmath import rect, phase  # Calculs complexes pour les coordonnées polaires

# Memory Management
import gc  # Collecteur de déchets (garbage collector)


warnings.filterwarnings('ignore') #ignore all warnings in the script

class Fake(object):
    def __init__(self, li_obj):
        self.obj = li_obj

#create default figures
default_figure = go.Figure()
default_figure.update_xaxes(visible=False, showticklabels=False)
default_figure.update_yaxes(visible=False, showticklabels=False)
default_figure.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

default_figure_2 = go.Figure()
default_figure_2.update_xaxes(visible=False, showticklabels=False)
default_figure_2.update_yaxes(visible=False, showticklabels=False)
default_figure_2.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',height=900)

#define style for tab in Dash
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
#    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

TITLE = "Wind assets dashboard"

image_filename = 'CGNEE_Logo.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

clusters={"Fr+Be+Nl":["France","Belgium","Netherlands"],"Ir+UK":["Ireland","United Kingdom"],"Sweden":["Sweden","Unknown"]}

def median_angle(deg): #Calculate the median angle for each column in a DataFrame, normalized between -180° and 180°
    averages={}
    for col in deg:
        alist = deg[col].dropna().tolist()
        if len(alist) > 0:
            alist=sorted(alist)
            median=alist[int(len(alist)*50/100)]
            if median < 0:
                median = 360 + median
        else:
            median=np.nan
        averages[col]=median
        averages[col]=(((averages[col]+180) % 360) - 180)
    df=pd.Series(averages)
    return df


print("["+dt.today().strftime('%d/%m/%Y %H:%M')+"] [INFO] KPIs calculation starting")    

#Fetch metadata from the API
url = "https://cgnee.greenbyte.cloud/api/2.0/sites.json"
headers = {"Breeze-ApiToken": "b9ac2c7d06a64e06af896d9e26f051fa"} #Nicolaz Guidon API key
params={
    "fields":["title","metadata"],
    "pageSize":1000
    }

try:
    r = requests.get(url=url,headers=headers,params=params) 
    data_list = r.json()
    metadata = pd.DataFrame(data_list)
except:
    time.sleep(30)
    r = requests.get(url=url,headers=headers,params=params) 
    data_list = r.json()
    metadata = pd.DataFrame(data_list)

# Process metadata to extract asset manager and portfolio information
metadata["metadata"]=metadata["metadata"].apply(lambda x: {item['key']:item['value'] for item in x})
metadata["AM"]=metadata["metadata"].apply(lambda x: x["Asset manager"] if "Asset manager" in x.keys() else "Undefined")
metadata["Portfolio"]=metadata["metadata"].apply(lambda x: x["CGN Portfolio"] if "CGN Portfolio" in x.keys() else "Undefined")
metadata.index=metadata["title"]
metadata=metadata[["AM","Portfolio"]]

#Fetch device information
url = "https://cgnee.greenbyte.cloud/api/2.0/devices.json"
headers = {"Breeze-ApiToken": "b9ac2c7d06a64e06af896d9e26f051fa"} #Nicolaz Guidon API key
params={
    "deviceTypeIds":1, 
    "fields":["site","title","deviceId","turbineType","longitude","latitude",],
    "pageSize":1000
    }

try:
    r = requests.get(url=url,headers=headers,params=params) 
    data_list = r.json()
    devices_info = pd.DataFrame(data_list)
except:
    time.sleep(30)
    r = requests.get(url=url,headers=headers,params=params) 
    data_list = r.json()
    devices_info = pd.DataFrame(data_list)

#Process wind farm and turbine info
devices_info_wfs=devices_info["site"].tolist()
devices_info_wfs=pd.DataFrame(devices_info_wfs)
devices_info_wfs=devices_info_wfs[["title","siteId"]].rename(columns={"title":'Wind farm',"siteId":'WFId'})

manufacturers=devices_info["turbineType"].tolist()
manufacturers=pd.DataFrame(manufacturers)
manufacturers=manufacturers[["manufacturer"]].rename(columns={"manufacturer":'Manufacturer'})

devices_infos=pd.concat([devices_info_wfs,devices_info["title"],devices_info["deviceId"],devices_info["longitude"],devices_info["latitude"],manufacturers], axis=1)

wt_info_frame=devices_infos.rename(columns={"title":'Wind turbine',"deviceId":'WTId',"longitude":'WTLongitude',"latitude":'WTLatitude'})

wt_ids=wt_info_frame["WTId"].tolist()

wt_names=wt_info_frame["Wind turbine"].tolist()
wf_names=wt_info_frame['Wind farm'].tolist()
wf_names=sorted(list(set(wf_names)))

wf_coordinates_df=wt_info_frame[["Wind farm","WTLatitude","WTLongitude"]]
wf_coordinates_df['WTLatitude'] = pd.to_numeric(wf_coordinates_df['WTLatitude'])
wf_coordinates_df['WTLongitude'] = pd.to_numeric(wf_coordinates_df['WTLongitude'])
wf_coordinates_df=wf_coordinates_df.groupby(["Wind farm"]).mean()

#Geocode wind farms
geolocator = Nominatim(user_agent="WindAssetsDashboard",timeout=10)
try:
    wf_coordinates_df["Country"]=wf_coordinates_df.apply(lambda row: geolocator.reverse(str(row["WTLatitude"])+", "+str(row["WTLongitude"]),language="en"),axis=1)
except:
    time.sleep(30)
    wf_coordinates_df["Country"]=wf_coordinates_df.apply(lambda row: geolocator.reverse(str(row["WTLatitude"])+", "+str(row["WTLongitude"]),language="en"),axis=1)

wf_coordinates_df["Country"]=wf_coordinates_df["Country"].apply(lambda x: x.raw["address"]["country"] if "country" in x.raw["address"] else "Undefined")

for wf in wf_names:
    country=wf_coordinates_df.at[wf,"Country"]
    selected_wf_wt_names=wt_info_frame.loc[wt_info_frame['Wind farm']==wf]['Wind turbine'].tolist()
    wt_info_frame.loc[wt_info_frame['Wind turbine'].isin(selected_wf_wt_names),"Country"]=country

params={
    "deviceTypeIds":11,
    "fields":["site","deviceId"],
    "pageSize":1000
    }

try:
    r = requests.get(url=url,headers=headers,params=params) 
    data_list = r.json()
    grid_metering_devices_info = pd.DataFrame(data_list)
except:
    time.sleep(30)
    r = requests.get(url=url,headers=headers,params=params) 
    data_list = r.json()
    grid_metering_devices_info = pd.DataFrame(data_list)

grid_metering_devices_info_1=grid_metering_devices_info["site"].tolist()
grid_metering_devices_info_1=pd.DataFrame(grid_metering_devices_info_1)
grid_metering_devices_info_1=grid_metering_devices_info_1[["title"]]

grid_metering_devices_info_2=grid_metering_devices_info[["deviceId"]]

grid_metering_devices_info=pd.concat([grid_metering_devices_info_1,grid_metering_devices_info_2], axis=1)

#delete unused dataframes
del grid_metering_devices_info_1
del grid_metering_devices_info_2
gc.collect()

grid_metering_devices_info=grid_metering_devices_info.rename(columns={"title":'Wind farm',"deviceId":'MeterId'})
grid_metering_devices_info.index=grid_metering_devices_info["Wind farm"]

start_date=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-240)
end_date=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

#Fetch monthly energy production data for wind turbines
url = "https://cgnee.greenbyte.cloud/api/2.0/data.json"

params={
    "deviceIds":wt_ids,
    "dataSignalIds":[4],
    "timestampStart":start_date,
    "timestampEnd":end_date,
    "resolution":"monthly"
    }

try:
    r = requests.get(url=url,headers=headers,params=params)
    data_list = r.json()
    df = pd.DataFrame(data_list)
except:
    time.sleep(30)
    r = requests.get(url=url,headers=headers,params=params)
    data_list = r.json()
    df = pd.DataFrame(data_list)

#Process the fetched data
truc=df[['deviceIds','dataSignal','data']]
truc['WT']=truc['deviceIds'].apply(lambda x: wt_info_frame.loc[wt_info_frame['WTId']==x[0]]['Wind turbine'].iloc[0])
truc['Param']=truc['dataSignal'].apply(lambda x: x['title'])
tmp = pd.DataFrame((d for idx, d in truc['data'].items()))
truc=truc.drop(columns=['deviceIds','dataSignal','data'])

#Prepare a MultiIndex DataFrame for energy production
timestamps=pd.DataFrame(tmp.iloc[0]).index
columns=pd.MultiIndex.from_product([wt_names,timestamps],names=["WT","Month"])
scada_prods=pd.DataFrame(index=["Energy Export"],columns=columns)
scada_prods=pd.DataFrame.transpose(scada_prods)

#Populate the energy production dataframe
for item in truc.index:
    current_wt=truc["WT"].iloc[item]
    current_param=truc["Param"].iloc[item]
    current_data=tmp.iloc[item]
    current_data=pd.DataFrame(current_data)
    current_data[current_param]=current_data[item]
    for timestamp in timestamps:
        scada_prods.at[(current_wt,timestamp),current_param]=current_data.at[timestamp,current_param]

#delete unused dataframe
del truc
gc.collect()

#Reset index can clean up
scada_prods=scada_prods.reset_index(level=["WT","Month"])
scada_prods["Month"]=scada_prods["Month"].astype('datetime64[ns]')
scada_prods=scada_prods.dropna()
scada_prods.index=range(len(scada_prods.index))

#Fetch cascading potential power data
params={
    "deviceIds":wt_ids,
    "dataSignalIds":[1457],
    "timestampStart":start_date,
    "timestampEnd":end_date,
    "resolution":"monthly"
     }

try:
    r = requests.get(url=url,headers=headers,params=params)
    data_list = r.json()
    df = pd.DataFrame(data_list)
except:
    time.sleep(30)
    r = requests.get(url=url,headers=headers,params=params)
    data_list = r.json()
    df = pd.DataFrame(data_list)

#Process the potential power data
truc=df[['deviceIds','dataSignal','data']]

#delete unused dataframe
del df
gc.collect()

truc['WT']=truc['deviceIds'].apply(lambda x: wt_info_frame.loc[wt_info_frame['WTId']==x[0]]['Wind turbine'].iloc[0])
truc['Param']=truc['dataSignal'].apply(lambda x: x['title'])

tmp = pd.DataFrame((d for idx, d in truc['data'].items()))

truc=truc.drop(columns=['deviceIds','dataSignal','data'])

#Prepare MultiIndex dataframe for potential power
timestamps=pd.DataFrame(tmp.iloc[0]).index

columns=pd.MultiIndex.from_product([wt_names,timestamps],names=["WT","Month"])

potential_powers=pd.DataFrame(index=["Cascading potential power"],columns=columns)
potential_powers=pd.DataFrame.transpose(potential_powers)

#Populate the potential power dataframe
for item in truc.index:
    current_wt=truc["WT"].iloc[item]

    current_param=truc["Param"].iloc[item]

    current_data=tmp.iloc[item]
    current_data=pd.DataFrame(current_data)
    current_data[current_param]=current_data[item]

    for timestamp in timestamps:
        potential_powers.at[(current_wt,timestamp),current_param]=current_data.at[timestamp,current_param]

#delete unused dataframe
del truc
gc.collect()

#Reset index and clean up
potential_powers=potential_powers.reset_index(level=["WT","Month"])
potential_powers["Month"]=potential_powers["Month"].astype('datetime64[ns]')
potential_powers=potential_powers.dropna()
potential_powers.index=range(len(potential_powers.index))

#Fetch net production data for each wind farm
net_prods = pd.DataFrame()

for wf in wf_names:
    meter_id=grid_metering_devices_info.at[wf,"MeterId"]

    url = "https://cgnee.greenbyte.cloud/api/2.0/data.json"

    params={
        "deviceIds":meter_id,
        "dataSignalIds":2857,
        "timestampStart":start_date,
        "timestampEnd":end_date,
        "resolution":"monthly"
        }

    try:
        r = requests.get(url=url,headers=headers,params=params) 
        data_list = r.json()
        net_aeps = pd.DataFrame(data_list)

    except:
        time.sleep(30)
        r = requests.get(url=url,headers=headers,params=params)
        data_list = r.json()
        net_aeps = pd.DataFrame(data_list)

    net_aeps=net_aeps['data'].tolist()
    net_aeps=pd.DataFrame(net_aeps)
    net_aeps.index=[wf]
    net_prods=pd.concat([net_prods,net_aeps])

#delete df not used anymore
del net_aeps
gc.collect()

net_prods=pd.DataFrame.transpose(net_prods)
net_prods.index = net_prods.index.astype('datetime64[ns]')
net_prods=net_prods/1000
net_prods = net_prods.replace(0, np.nan)

end_date=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)

tmp_cumulated_net_prods=net_prods.copy()

years=pd.period_range(start_date,end_date,freq='1M').to_timestamp().tolist()
years=[i.year for i in years]
years=sorted(list(set(years)))

cumulated_net_prods=pd.DataFrame()

for year in years:
    tmp_tmp_cumulated_net_prods=tmp_cumulated_net_prods.loc[tmp_cumulated_net_prods.index.year==year]
    tmp_tmp_cumulated_net_prods=tmp_tmp_cumulated_net_prods.cumsum(skipna=False)

    cumulated_net_prods=pd.concat([cumulated_net_prods,tmp_tmp_cumulated_net_prods])

#delete useless dataframes
del tmp_cumulated_net_prods
del tmp_tmp_cumulated_net_prods
gc.collect()    

#Load budget data from csv files
budgets = pd.concat(map(pd.read_csv, glob.glob('Budgets/*.csv')), axis=1)
budgets=budgets.loc[:,~budgets.columns.duplicated()]
budgets.index=budgets["Unnamed: 0"]

wf_budgets=budgets[[col for col in budgets.columns if col in wf_names]]

#delete unused dataframe
del budgets
gc.collect()

budget_params = [
    "January Net production (GWh)","February Net production (GWh)","March Net production (GWh)",
    "April Net production (GWh)","May Net production (GWh)","June Net production (GWh)",
    "July Net production (GWh)","August Net production (GWh)","September Net production (GWh)",
    "October Net production (GWh)","November Net production (GWh)","December Net production (GWh)",

    "January P75 (GWh)","February P75 (GWh)","March P75 (GWh)",
    "April P75 (GWh)","May P75 (GWh)","June P75 (GWh)",
    "July P75 (GWh)","August P75 (GWh)","September P75 (GWh)","October P75 (GWh)",
    "November P75 (GWh)","December P75 (GWh)",

    "January YTD P50 (GWh)","February YTD P50 (GWh)","March YTD P50 (GWh)",
    "April YTD P50 (GWh)","May YTD P50 (GWh)","June YTD P50 (GWh)",
    "July YTD P50 (GWh)","August YTD P50 (GWh)","September YTD P50 (GWh)",
    "October YTD P50 (GWh)","November YTD P50 (GWh)","December YTD P50 (GWh)",

    "January YTD P75 (GWh)","February YTD P75 (GWh)","March YTD P75 (GWh)",
    "April YTD P75 (GWh)","May YTD P75 (GWh)","June YTD P75 (GWh)",
    "July YTD P75 (GWh)","August YTD P75 (GWh)","September YTD P75 (GWh)",
    "October YTD P75 (GWh)","November YTD P75 (GWh)","December YTD P75 (GWh)"
    ]

wf_budgets=wf_budgets.loc[wf_budgets.index.isin(budget_params)]
wf_budgets=wf_budgets*1000

#Generate monthly production periods
production_months=pd.period_range(start_date,end_date,freq='1M').to_timestamp().tolist()

#Initialize DataFrame for production KPIs
production_kpis = pd.DataFrame(index = wf_names)
production_kpis["WF"] = production_kpis.index
production_kpis.index = range(len(production_kpis.index))

#Add country information for each wind farm
production_kpis["Country"] = production_kpis["WF"].apply(
    lambda x: wf_coordinates_df.at[x,"Country"] if x in wf_coordinates_df.index else "Unknown"
    )

#Populate monthly production for each wind farm
production_kpis["Monthly productions temp"] = production_kpis["WF"].apply(
    lambda x: net_prods[[x]].to_dict()
    )

production_kpis["Monthly productions"] = production_kpis.apply(
    lambda row: row["Monthly productions temp"][row["WF"]],axis = 1
    )


#Populate cumulative monthly YTD productions
production_kpis["Monthly YTD productions temp"] = production_kpis["WF"].apply(
    lambda x: cumulated_net_prods[[x]].to_dict()
    )

production_kpis["Monthly YTD productions"] = production_kpis.apply(
    lambda row: row["Monthly YTD productions temp"][row["WF"]],axis = 1
    )


#Populate P50, P75, P25 metrics
production_kpis["Monthly P50s temp"] = production_kpis["WF"].apply(
    lambda x: wf_budgets.loc[wf_budgets.index.str.contains('Net')][x].tolist()
    )

production_kpis["Monthly P75s temp"] = production_kpis["WF"].apply(
    lambda x: wf_budgets.loc[wf_budgets.index.str.contains('P75') & ~wf_budgets.index.str.contains('YTD')][x].tolist()
    )

production_kpis["Monthly P25s temp"] = production_kpis.apply(
    lambda row: [2*p50-p75 for p50, p75 in zip(row["Monthly P50s temp"],row["Monthly P75s temp"])], axis = 1
    )


#Populate YTD P50, P75, P25 metrics
production_kpis["Monthly YTD P50s temp"] = production_kpis["WF"].apply(
    lambda x: wf_budgets.loc[wf_budgets.index.str.contains('YTD P50')][x].tolist()
    )

production_kpis["Monthly YTD P75s temp"] = production_kpis["WF"].apply(
    lambda x: wf_budgets.loc[wf_budgets.index.str.contains('YTD P75')][x].tolist()
    )

production_kpis["Monthly YTD P25s temp"] = production_kpis.apply(
    lambda row: [2*p50-p75 for p50, p75 in zip(row["Monthly YTD P50s temp"],row["Monthly YTD P75s temp"])], axis = 1
    )


#Format P50, P75, P25 into dictionaries
production_kpis["Monthly P50s"] = production_kpis["Monthly P50s temp"].apply(
    lambda x: {item:x[(item.month)-1] for item in production_months}
    )

production_kpis["Monthly P75s"] = production_kpis["Monthly P75s temp"].apply(
    lambda x: {item:x[(item.month)-1] for item in production_months}
    )

production_kpis["Monthly P25s"] = production_kpis["Monthly P25s temp"].apply(
    lambda x: {item:x[(item.month)-1] for item in production_months}
    )


#Calculate P90s and P10s based on P50s and P75s
production_kpis["Monthly P90s"] = production_kpis.apply(
    lambda row: {item:(row["Monthly P50s"][item]-(1.282/0.675)*(row["Monthly P50s"][item]-row["Monthly P75s"][item])) for item in production_months}, axis = 1
    )

production_kpis["Monthly P10s"] = production_kpis.apply(
    lambda row: {item:(row["Monthly P50s"][item]+(1.282/0.675)*(row["Monthly P50s"][item]-row["Monthly P75s"][item])) for item in production_months}, axis = 1
    )

#Format YTD production metrics
production_kpis["Monthly YTD P50s"] = production_kpis["Monthly YTD P50s temp"].apply(
    lambda x: {item:x[(item.month)-1] for item in production_months}
    )

production_kpis["Monthly YTD P75s"] = production_kpis["Monthly YTD P75s temp"].apply(
    lambda x: {item:x[(item.month)-1] for item in production_months}
    )

production_kpis["Monthly YTD P25s"] = production_kpis["Monthly YTD P25s temp"].apply(
    lambda x: {item:x[(item.month)-1] for item in production_months}
    )


#Calculate P90s and P10s YTD
production_kpis["Monthly YTD P90s"] = production_kpis.apply(
    lambda row: {item:(row["Monthly YTD P50s"][item]-(1.282/0.675)*(row["Monthly YTD P50s"][item]-row["Monthly YTD P75s"][item])) for item in production_months}, axis = 1
    )

production_kpis["Monthly YTD P10s"] = production_kpis.apply(
    lambda row: {item:(row["Monthly YTD P50s"][item]+(1.282/0.675)*(row["Monthly YTD P50s"][item]-row["Monthly YTD P75s"][item])) for item in production_months}, axis = 1
    )

#Calculate monthly producton KPIs
production_kpis["Monthly production KPI"] = production_kpis.apply(
    lambda row: '⚫' if (end_date not in row["Monthly productions"].keys()) | (pd.isnull(row["Monthly productions"][end_date]) == True) 
    else '🟢'+" (+"+str(round(100 * ((row["Monthly productions"][end_date]-row["Monthly P50s"][end_date])/row["Monthly P50s"][end_date]),1))+"%)",axis = 1
    )

production_kpis["Monthly production KPI"] = production_kpis.apply(
    lambda row: '🟠'+" ("+str(round(100 * ((row["Monthly productions"][end_date]-row["Monthly P50s"][end_date])/row["Monthly P50s"][end_date]),1))+"%)" 
    if (end_date in row["Monthly productions"].keys()) and 
    (row["Monthly productions"][end_date] >= row["Monthly P75s"][end_date]) and 
    (row["Monthly productions"][end_date] < row["Monthly P50s"][end_date]) 
    else row["Monthly production KPI"],axis = 1
    )

production_kpis["Monthly production KPI"] = production_kpis.apply(
    lambda row: '🔴'+" ("+str(round(100 * ((row["Monthly productions"][end_date]-row["Monthly P50s"][end_date])/row["Monthly P50s"][end_date]),1))+"%)" 
    if (end_date in row["Monthly productions"].keys()) and 
    (row["Monthly productions"][end_date] < row["Monthly P75s"][end_date]) 
    else row["Monthly production KPI"],axis = 1
    )


#Calculate KPI for YTD production
production_kpis["YTD production KPI"] = production_kpis.apply(
    lambda row: '⚫' if (end_date not in row["Monthly YTD productions"].keys()) | (pd.isnull(row["Monthly YTD productions"][end_date]) == True) 
    else '🟢'+" (+"+str(round(100 * ((row["Monthly YTD productions"][end_date]-row["Monthly YTD P50s"][end_date])/row["Monthly YTD P50s"][end_date]),1))+"%)",axis = 1
    )

production_kpis["YTD production KPI"] = production_kpis.apply(
    lambda row: '🟠'+" ("+str(round(100 * ((row["Monthly YTD productions"][end_date]-row["Monthly YTD P50s"][end_date])/row["Monthly YTD P50s"][end_date]),1))+"%)" 
    if (end_date in row["Monthly YTD productions"].keys()) and 
    (row["Monthly YTD productions"][end_date] >= row["Monthly YTD P75s"][end_date]) and 
    (row["Monthly YTD productions"][end_date] < row["Monthly YTD P50s"][end_date]) else row["YTD production KPI"],axis = 1
    )

production_kpis["YTD production KPI"] = production_kpis.apply(
    lambda row: '🔴'+" ("+str(round(100 * ((row["Monthly YTD productions"][end_date] - row["Monthly YTD P50s"][end_date])/row["Monthly YTD P50s"][end_date]),1))+"%)" 
    if (end_date in row["Monthly YTD productions"].keys()) and 
    (row["Monthly YTD productions"][end_date]<row["Monthly YTD P75s"][end_date]) 
    else row["YTD production KPI"],axis = 1
    )


#Sort production KPIs
#production_kpis=production_kpis.sort_values(by=['Country', 'WF'])
production_kpis=production_kpis.sort_values(by=['Country','WF'],key=natsort_keygen())

#Create a dataframe for current producitn KPIs
current_production_kpis=production_kpis[["Country","WF","Monthly production KPI","YTD production KPI"]]
current_production_kpis["id"]=range(len(current_production_kpis.index))

#delete unused dataframes
del production_kpis
gc.collect()

#Rename columns
current_production_kpis=current_production_kpis.rename(columns={"WF":"Wind farm","YTD production KPI":"Monthly Year-To-Date production KPI"})

print("["+dt.today().strftime('%d/%m/%Y %H:%M')+"] [INFO] Successfully computed production KPIs")    

#Read the csv file and set column names
with open('AvailabilityAndStatusData.csv') as f:
    availability_kpis = pd.read_csv(f,sep=',',header=0,names=["Wind turbine","Month","Production-based System Avail.","Time-based System Avail.","details","details2","details3"])

availability_kpis["details"]=availability_kpis["details"].apply(lambda x: eval(x))
availability_kpis["details2"]=availability_kpis["details2"].apply(lambda x: eval(x))
availability_kpis["details3"]=availability_kpis["details3"].apply(lambda x: eval(x))

#Create 'AllStops', 'OwnStops', and 'ScheduleMaintenanceStops' columns
#Filtering events based on global contract and calculating duration
availability_kpis["AllStops"] = availability_kpis["details"].apply(
    lambda x: [{"Start": event["Start"],
    "End": event["End"],
    "Duration": (pd.to_datetime(event["End"], format = '%d/%m/%Y %H:%M') - 
        pd.to_datetime(event["Start"], format = '%d/%m/%Y %H:%M')).total_seconds(),
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "Loss": event["Loss"]} 
    if (event["Global contract category"] == "Electrical transmission outage" 
        or event["Global contract category"] == "Turbine outage" 
        or event["Global contract category"] == "Requested shutdown" 
        or event["Global contract category"] == "Turbine scheduled maintenance" 
        or event["Global contract category"] == "BOP scheduled maintenance" 
        or event["Global contract category"] == "BOP outage") 
    else np.nan for event in x]
    )

availability_kpis["OwnStops"] = availability_kpis["details"].apply(
    lambda x: [{"Start": event["Start"],
    "End": event["End"],
    "Duration": (pd.to_datetime(event["End"], format = '%d/%m/%Y %H:%M') - 
        pd.to_datetime(event["Start"], format = '%d/%m/%Y %H:%M')).total_seconds(),
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "Loss": event["Loss"]} 
    if (event["Global contract category"] == "Requested shutdown") 
    else np.nan for event in x]
    )

availability_kpis["ScheduledMaintenanceStops"] = availability_kpis["details"].apply(
    lambda x: [{"Start": event["Start"],
    "End": event["End"],
    "Duration": (pd.to_datetime(event["End"], format = '%d/%m/%Y %H:%M') -
        pd.to_datetime(event["Start"], format = '%d/%m/%Y %H:%M')).total_seconds(),
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "Loss": event["Loss"]} 
    if (event["Global contract category"] == "Turbine scheduled maintenance" 
        or event["Global contract category"] == "BOP scheduled maintenance") 
    else np.nan for event in x]
    )

#Remove NaN values
availability_kpis["AllStops"] = availability_kpis["AllStops"].apply(
    lambda x: [item for item in x if pd.isnull(item) == False]
    )
availability_kpis["OwnStops"] = availability_kpis["OwnStops"].apply(
    lambda x: [item for item in x if pd.isnull(item) == False]
    )

availability_kpis["ScheduledMaintenanceStops"] = availability_kpis["ScheduledMaintenanceStops"].apply(
    lambda x: [item for item in x if pd.isnull(item) == False]
    )

#Convert Month column to datetime format and create a formatted 'Date' columnb
availability_kpis["Month"] = pd.to_datetime(availability_kpis["Month"], format = '%B %Y')
availability_kpis["Date"] = availability_kpis["Month"].apply(lambda x: x.strftime('%B %Y'))

#Calculate potential power based on the turbine and month
availability_kpis["PotentialPower"] = availability_kpis.apply(
    lambda row: (potential_powers.loc[
        (potential_powers["WT"] == row["Wind turbine"]) &
        (potential_powers["Month"] == row["Date"]),"Cascading potential power"
        ].iloc[0])*1000 
    if (pd.isnull(row["Wind turbine"]) == False and pd.isnull(row["Date"]) == False 
        and potential_powers.loc[
        (potential_powers["WT"] == row["Wind turbine"]) &
        (potential_powers["Month"] == row["Date"])
        ].shape[0]>0) else np.nan,axis = 1
    )

#Calculate days in the month
availability_kpis["DaysInMonth"] = availability_kpis["Month"].apply(
    lambda x: calendar.monthrange(x.year, x.month)[1]
    )

#Create a copy for YTD calculations
ytd_availability_kpis=availability_kpis.copy()

years_in=[k.year for k in ytd_availability_kpis["Month"].tolist()]
months_in=sorted(list(set(ytd_availability_kpis["Month"].tolist())))
min_year_in=min(years_in)

#logic filter out incomplete years
first_year_months_count=len([p for p in months_in if p.year == min_year_in])
if first_year_months_count<12:
    months_to_keep=[i for i in months_in if i.year!=min_year_in]
    ytd_availability_kpis=ytd_availability_kpis.loc[ytd_availability_kpis["Month"].isin(months_to_keep)]

restructured_ytd_availability_kpis=pd.DataFrame(columns=ytd_availability_kpis.columns)

#Loop through each wind turbine and month to calculate averages and merges
for wt in sorted(list(set(ytd_availability_kpis["Wind turbine"].tolist()))):
    for period in sorted(list(set(ytd_availability_kpis["Month"].tolist()))):
        #Define months to keep based on the current period
        months_to_keep=[i for i in months_in if i.year==period.year and i.month<=period.month]
        tmp=ytd_availability_kpis.loc[(ytd_availability_kpis["Wind turbine"]==wt) & (ytd_availability_kpis["Month"].isin(months_to_keep))]
        tmp["Weighted Time-based System Avail."]=tmp.apply(
            lambda row: row["Time-based System Avail."]*row["DaysInMonth"] if pd.isnull(row["Time-based System Avail."])==False else np.nan,axis=1
            )
        
        #Weighted calculations
        tmp["Weighted Production-based System Avail."]=tmp.apply(
            lambda row: row["Production-based System Avail."]*row["PotentialPower"] if (pd.isnull(row["Production-based System Avail."])==False and pd.isnull(row["PotentialPower"])==False) 
            else np.nan,axis=1
            )
        
        avg_time_avail=tmp["Weighted Time-based System Avail."].sum(skipna=False)/tmp["DaysInMonth"].sum(skipna=False)
        avg_production_avail=tmp["Weighted Production-based System Avail."].sum(skipna=False)/tmp["PotentialPower"].sum(skipna=False)
        total_potential_power=tmp["PotentialPower"].sum()
        total_days=tmp["DaysInMonth"].sum()

        merged_details=[]
        for d in tmp["details"].tolist():
            merged_details=merged_details+d

        merged_all_stops=[]
        for d in tmp["AllStops"].tolist():
            merged_all_stops=merged_all_stops+d

        merged_own_stops=[]
        for d in tmp["OwnStops"].tolist():
            merged_own_stops=merged_own_stops+d

        merged_scheduled_maintenance_stops=[]
        for d in tmp["ScheduledMaintenanceStops"].tolist():
            merged_scheduled_maintenance_stops=merged_scheduled_maintenance_stops+d

        #Append to restructure DataFrame    
        restructured_ytd_availability_kpis = restructured_ytd_availability_kpis._append({
            "Wind turbine":wt,
            "Month":period,
            "Production-based System Avail.":avg_production_avail,
            "Time-based System Avail.":avg_time_avail,
            "details":merged_details,
            "AllStops":merged_all_stops,
            "OwnStops":merged_own_stops,
            "ScheduledMaintenanceStops":merged_scheduled_maintenance_stops,
            "Date":period.strftime('%B %Y'),
            "PotentialPower":total_potential_power,
            "DaysInMonth":total_days
            }, ignore_index = True)

del tmp

#Additional processing on AllStops, OwnStops and Schedule Maintenance Stops
availability_kpis["AllStops2"] = availability_kpis["AllStops"].apply(
    lambda x: [{
    "Start": event["Start"],
    "End": event["End"],
    "Duration": (pd.to_datetime(event["End"], format = '%d/%m/%Y %H:%M') -
        pd.to_datetime(event["Start"], format = '%d/%m/%Y %H:%M')).total_seconds(),
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "Loss": event["Loss"]} for event in x]
    )

availability_kpis["OwnStops2"] = availability_kpis["OwnStops"].apply(
    lambda x: [{"Start": event["Start"],
    "End": event["End"],
    "Duration": (pd.to_datetime(event["End"], format = '%d/%m/%Y %H:%M') -
        pd.to_datetime(event["Start"], format = '%d/%m/%Y %H:%M')).total_seconds(),
    "Code": event["Code"],"Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "Loss": event["Loss"]} for event in x]
    )

availability_kpis["ScheduledMaintenanceStops2"] = availability_kpis["ScheduledMaintenanceStops"].apply(
    lambda x: [{"Start": event["Start"],
    "End": event["End"],
    "Duration": (pd.to_datetime(event["End"], format = '%d/%m/%Y %H:%M') -
        pd.to_datetime(event["Start"], format = '%d/%m/%Y %H:%M')).total_seconds(),
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "Loss": event["Loss"]} for event in x]
    )

#Additional calculations on stops
availability_kpis["AllStops3"] = availability_kpis.apply(
    lambda row: [{"Start": event["Start"],
    "End": event["End"],
    "Duration": event["Duration"],
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "% Time": 100 * event["Duration"]/row["DaysInMonth"] /60/60/24,
    "% Loss": 100 * event["Loss"]/row["PotentialPower"]} for event in row["AllStops2"]],axis = 1
    )

availability_kpis["OwnStops3"] = availability_kpis.apply(
    lambda row: [{
    "Start": event["Start"],
    "End": event["End"],
    "Duration": event["Duration"],
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "% Time": 100 * event["Duration"] / row["DaysInMonth"] / 60/60/24,
    "% Loss": 100 * event["Loss"] / row["PotentialPower"],
    "Stop optimization factor": (event["Loss"]/row["PotentialPower"]) / (event["Duration"] / row["DaysInMonth"] /60/60/24) 
    if event["Duration"]!= 0 else np.nan} for event in row["OwnStops2"]], axis = 1
    )

availability_kpis["ScheduledMaintenanceStops3"] = availability_kpis.apply(
    lambda row: [{
    "Start": event["Start"],
    "End": event["End"],
    "Duration": event["Duration"],
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "% Time": 100 * event["Duration"] / row["DaysInMonth"]/60/60/24,
    "% Loss": 100 * event["Loss"] / row["PotentialPower"],
    "Stop optimization factor": (event["Loss"] / row["PotentialPower"]) / (event["Duration"] / row["DaysInMonth"]/60/60/24) 
    if event["Duration"] !=  0 else np.nan} for event in row["ScheduledMaintenanceStops2"]],axis = 1
    )

#Calculations KPIs monthly stops
availability_kpis["Monthly own stops value KPI"] = availability_kpis["OwnStops3"].apply(
    lambda x: sum([event["% Loss"] for event in x])/sum([event["% Time"] for event in x]) if sum([event["% Time"] for event in x]) !=  0 else np.nan
    )
availability_kpis["Monthly scheduled maintenance stops value KPI"] = availability_kpis["ScheduledMaintenanceStops3"].apply(
    lambda x: sum([event["% Loss"] for event in x]) / sum([event["% Time"] for event in x]) if sum([event["% Time"] for event in x]) !=  0 else np.nan
    )

#Colors stops on graphs
availability_kpis["Monthly own stops Graph Colors"] = availability_kpis["Monthly own stops value KPI"].apply(
    lambda x: "red" 
    if x>2 
    else("orange" if x>1 
        else ("green" if x<= 1 
            else "black")
        )
    )
availability_kpis["Monthly scheduled maintenance stops Graph Colors"] = availability_kpis["Monthly scheduled maintenance stops value KPI"].apply(
    lambda x: "red" 
    if x>2 
    else("orange" if x>1 
        else ("green" if x<= 1 
            else "black")
        )
    )

availability_kpis["Monthly own stops KPI"] = availability_kpis["Monthly own stops value KPI"].apply(
    lambda x: "🔴"+" (Opt.Fact. = "+str(round(x,1))+")" 
    if x>2 
    else("🟠"+" (Opt.Fact. = "+str(round(x,1))+")" 
        if x>1 
        else ("🟢"+" (Opt.Fact. = "+str(round(x,1))+")" 
            if x<= 1 
            else "⚫"))
    )
availability_kpis["Monthly scheduled maintenance stops KPI"] = availability_kpis["Monthly scheduled maintenance stops value KPI"].apply(
    lambda x: "🔴"+" (Opt.Fact. = "+str(round(x,1))+")" 
    if x>2 
    else("🟠"+" (Opt.Fact. = "+str(round(x,1))+")" 
        if x>1 
        else ("🟢"+" (Opt.Fact. = "+str(round(x,1))+")" 
            if x<= 1 
            else "⚫"))
    )


#Transforming event data into a structured format for various stop types
restructured_ytd_availability_kpis["AllStops2"] = restructured_ytd_availability_kpis["AllStops"].apply(
    lambda x: [{
    "Start": event["Start"],
    "End": event["End"],
    "Duration": (pd.to_datetime(event["End"], format = '%d/%m/%Y %H:%M') -
        pd.to_datetime(event["Start"], format = '%d/%m/%Y %H:%M')).total_seconds(),
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "Loss": event["Loss"]} for event in x]
    )

restructured_ytd_availability_kpis["OwnStops2"] = restructured_ytd_availability_kpis["OwnStops"].apply(
    lambda x: [{
    "Start": event["Start"],
    "End": event["End"],
    "Duration": (pd.to_datetime(event["End"], format = '%d/%m/%Y %H:%M') -
        pd.to_datetime(event["Start"], format = '%d/%m/%Y %H:%M')).total_seconds(),
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "Loss": event["Loss"]} for event in x]
    )

restructured_ytd_availability_kpis["ScheduledMaintenanceStops2"] = restructured_ytd_availability_kpis["ScheduledMaintenanceStops"].apply(
    lambda x: [{
    "Start": event["Start"],
    "End": event["End"],
    "Duration": (pd.to_datetime(event["End"], format = '%d/%m/%Y %H:%M') -
        pd.to_datetime(event["Start"], format = '%d/%m/%Y %H:%M')).total_seconds(),
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "Loss": event["Loss"]} for event in x]
    )


#Calculate metrics for each type of stop 
restructured_ytd_availability_kpis["AllStops3"] = restructured_ytd_availability_kpis.apply(
    lambda row: [{
    "Start": event["Start"],
    "End": event["End"],
    "Duration": event["Duration"],
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "% Time": 100 * event["Duration"] / row["DaysInMonth"] / 60 / 60 / 24,
    "% Loss": 100 * event["Loss"] / row["PotentialPower"] if row["PotentialPower"] !=  0 else np.nan} for event in row["AllStops2"]],axis = 1
    )

restructured_ytd_availability_kpis["OwnStops3"] = restructured_ytd_availability_kpis.apply(
    lambda row: [{
    "Start": event["Start"],
    "End": event["End"],
    "Duration": event["Duration"],
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "% Time": 100 * event["Duration"] / row["DaysInMonth"] / 60 / 60 / 24,
    "% Loss": 100 * event["Loss"] / row["PotentialPower"] if row["PotentialPower"] !=  0 else np.nan,
    "Stop optimization factor": (event["Loss"]/row["PotentialPower"]) / (event["Duration"] / row["DaysInMonth"] / 60 / 60 / 24 ) 
    if (event["Duration"] !=  0 and row["PotentialPower"] !=  0) else np.nan} for event in row["OwnStops2"]],axis = 1
    )

restructured_ytd_availability_kpis["ScheduledMaintenanceStops3"] = restructured_ytd_availability_kpis.apply(
    lambda row: [{
    "Start": event["Start"],
    "End": event["End"],
    "Duration": event["Duration"],
    "Code": event["Code"],
    "Message": event["Message"],
    "Category": event["Category"],
    "Global contract category": event["Global contract category"],
    "% Time": 100 * event["Duration"] / row["DaysInMonth"] / 60 / 60 / 24,
    "% Loss": 100 * event["Loss"] / row["PotentialPower"] if row["PotentialPower"] !=  0 else np.nan,
    "Stop optimization factor": (event["Loss"] / row["PotentialPower"]) / (event["Duration"] / row["DaysInMonth"] / 60 / 60 / 24) 
    if (event["Duration"] !=  0 and row["PotentialPower"] !=  0) 
    else np.nan} for event in row["ScheduledMaintenanceStops2"]],axis = 1
    )

#Calculate Monthly KPIs for Own and Schedule Maintenance Stops or icons
restructured_ytd_availability_kpis["Monthly own stops value KPI"] = restructured_ytd_availability_kpis["OwnStops3"].apply(
    lambda x: sum([event["% Loss"] for event in x]) / sum([event["% Time"] for event in x]) 
    if sum([event["% Time"] for event in x]) !=  0 
    else np.nan
    )

restructured_ytd_availability_kpis["Monthly scheduled maintenance stops value KPI"] = restructured_ytd_availability_kpis["ScheduledMaintenanceStops3"].apply(
    lambda x: sum([event["% Loss"] for event in x]) / sum([event["% Time"] for event in x]) 
    if sum([event["% Time"] for event in x]) !=  0 
    else np.nan
    )

#Assign colors based on KPI values for graphs
restructured_ytd_availability_kpis["Monthly own stops Graph Colors"] = restructured_ytd_availability_kpis["Monthly own stops value KPI"].apply(
    lambda x: "red" if x > 2 else(
        "orange" if x > 1 else (
            "green" if x <=  1 
            else "black" )
        )
    )

restructured_ytd_availability_kpis["Monthly scheduled maintenance stops Graph Colors"] = restructured_ytd_availability_kpis["Monthly scheduled maintenance stops value KPI"].apply(
    lambda x: "red" if x > 2 else(
        "orange" if x > 1 else (
            "green" if x <=  1 
            else "black" )
        )
    )

restructured_ytd_availability_kpis["Monthly own stops KPI"] = restructured_ytd_availability_kpis["Monthly own stops value KPI"].apply(
    lambda x: "🔴"+" (Opt.Fact. = "+str(round(x,1))+")" if x>2 
    else("🟠"+" (Opt.Fact. = "+str(round(x,1))+")" if x>1 
        else ("🟢"+" (Opt.Fact. = "+str(round(x,1))+")" if x<= 1 
            else "⚫"))
    )
restructured_ytd_availability_kpis["Monthly scheduled maintenance stops KPI"] = restructured_ytd_availability_kpis["Monthly scheduled maintenance stops value KPI"].apply(
    lambda x: "🔴"+" (Opt.Fact. = "+str(round(x,1))+")" if x>2 
    else("🟠"+" (Opt.Fact. = "+str(round(x,1))+")" if x>1 
        else ("🟢"+" (Opt.Fact. = "+str(round(x,1))+")" if x<= 1 
            else "⚫")
        )
    )

#Sort
availability_kpis=availability_kpis.sort_values("Time-based System Avail.")
availability_kpis.index=range(len(availability_kpis.index))

#Calculate percentiles for Time-based System Availability
availability_kpis["Time-based System Avail. P50"]=availability_kpis["Time-based System Avail."].iloc[int(len(availability_kpis.index)/2)]
availability_kpis["Time-based System Avail. P95"]=availability_kpis["Time-based System Avail."].iloc[int(len(availability_kpis.index)/20)]
availability_kpis["Time-based System Avail. P99"]=availability_kpis["Time-based System Avail."].iloc[int(len(availability_kpis.index)/100)]

#Sort
availability_kpis=availability_kpis.sort_values("Production-based System Avail.")
availability_kpis.index=range(len(availability_kpis.index))

#Calculate percentiles for Production-based System Availability
availability_kpis["Production-based System Avail. P50"]=availability_kpis["Production-based System Avail."].iloc[int(len(availability_kpis.index)/2)]
availability_kpis["Production-based System Avail. P95"]=availability_kpis["Production-based System Avail."].iloc[int(len(availability_kpis.index)/20)]
availability_kpis["Production-based System Avail. P99"]=availability_kpis["Production-based System Avail."].iloc[int(len(availability_kpis.index)/100)]

#Sort
availability_kpis=availability_kpis.sort_values("Month")

#Calculate percentile values for each month
for period in [i+1 for i in range(12)]:
    months_to_include = [i for i in months_in if i.month ==  period]
    tmp = restructured_ytd_availability_kpis.loc[restructured_ytd_availability_kpis["Month"].isin(months_to_include)]


    #Time-Based System Avail. Percentiles
    tmp = tmp.sort_values("Time-based System Avail.")
    tmp.index = range(len(tmp.index))
    restructured_ytd_availability_kpis.loc[
    restructured_ytd_availability_kpis["Month"].isin(months_to_include),
    "Time-based System Avail. P50"] = tmp["Time-based System Avail."].iloc[int(len(tmp.index)/2)
    ]

    restructured_ytd_availability_kpis.loc[
    restructured_ytd_availability_kpis["Month"].isin(months_to_include),
    "Time-based System Avail. P95"] = tmp["Time-based System Avail."].iloc[int(len(tmp.index)/20)
    ]

    restructured_ytd_availability_kpis.loc[
    restructured_ytd_availability_kpis["Month"].isin(months_to_include),
    "Time-based System Avail. P99"] = tmp["Time-based System Avail."].iloc[int(len(tmp.index)/100)
    ]
    

    #Production Based System Availability Percentiles
    tmp = tmp.sort_values("Production-based System Avail.")
    tmp.index = range(len(tmp.index))

    restructured_ytd_availability_kpis.loc[
    restructured_ytd_availability_kpis["Month"].isin(months_to_include),
    "Production-based System Avail. P50"] = tmp["Production-based System Avail."].iloc[int(len(tmp.index)/2)
    ]

    restructured_ytd_availability_kpis.loc[
    restructured_ytd_availability_kpis["Month"].isin(months_to_include),
    "Production-based System Avail. P95"] = tmp["Production-based System Avail."].iloc[int(len(tmp.index)/20)
    ]

    restructured_ytd_availability_kpis.loc[
    restructured_ytd_availability_kpis["Month"].isin(months_to_include),
    "Production-based System Avail. P99"] = tmp["Production-based System Avail."].iloc[int(len(tmp.index)/100)
    ]

#delete unused dataframe
del tmp 
gc.collect()

#Apply icons depending of Time-based System Avail. KPI values 
availability_kpis["Time-based System Avail. KPI"] = availability_kpis.apply(
    lambda row: '⚫',axis = 1
    )

availability_kpis["Time-based System Avail. KPI"] = availability_kpis.apply(
    lambda row: '🔴' if row["Time-based System Avail."] < row["Time-based System Avail. P99"] 
    else row["Time-based System Avail. KPI"],axis = 1
    )

availability_kpis["Time-based System Avail. KPI"] = availability_kpis.apply(
    lambda row: '🟠' if (row["Time-based System Avail. KPI"] != '🔴' and row["Time-based System Avail."] < row["Time-based System Avail. P95"]) 
    else row["Time-based System Avail. KPI"],axis = 1
    )

availability_kpis["Time-based System Avail. KPI"] = availability_kpis.apply(
    lambda row: '🟢' if (row["Time-based System Avail. KPI"] == '⚫' and row["Time-based System Avail."] >= row["Time-based System Avail. P95"]) 
    else row["Time-based System Avail. KPI"],axis = 1
    )

availability_kpis["Time-based System Avail. KPI"] = availability_kpis.apply(
    lambda row: row["Time-based System Avail. KPI"]+" ("+str(round(row["Time-based System Avail."],1))+"%)" if row["Time-based System Avail. KPI"] != '⚫' 
    else row["Time-based System Avail. KPI"],axis = 1
    )

#Apply icons depending of Production-based System Avail. KPI values 
availability_kpis["Production-based System Avail. KPI"] = availability_kpis.apply(
    lambda row: '⚫',axis = 1
    )

availability_kpis["Production-based System Avail. KPI"] = availability_kpis.apply(
    lambda row: '🔴' if row["Production-based System Avail."] < row["Production-based System Avail. P99"] 
    else row["Production-based System Avail. KPI"],axis = 1
    )

availability_kpis["Production-based System Avail. KPI"] = availability_kpis.apply(
    lambda row: '🟠' if (row["Production-based System Avail. KPI"] != '🔴' and row["Production-based System Avail."] < row["Production-based System Avail. P95"]) 
    else row["Production-based System Avail. KPI"],axis = 1
    )

availability_kpis["Production-based System Avail. KPI"] = availability_kpis.apply(
    lambda row: '🟢' if (row["Production-based System Avail. KPI"] == '⚫' and row["Production-based System Avail."] >= row["Production-based System Avail. P95"]) 
    else row["Production-based System Avail. KPI"], axis = 1
    )

availability_kpis["Production-based System Avail. KPI"] = availability_kpis.apply(
    lambda row: row["Production-based System Avail. KPI"]+" ("+str(round(row["Production-based System Avail."],1))+"%)" if row["Production-based System Avail. KPI"] != '⚫' 
    else row["Production-based System Avail. KPI"],axis = 1
    )


#Assign icons on graphs
#Time based System Avail KPI
restructured_ytd_availability_kpis["Time-based System Avail. KPI"] = restructured_ytd_availability_kpis.apply(
    lambda row: '⚫', axis = 1
    )

restructured_ytd_availability_kpis["Time-based System Avail. KPI"] = restructured_ytd_availability_kpis.apply(
    lambda row: '🔴' if row["Time-based System Avail."] < row["Time-based System Avail. P99"] 
    else row["Time-based System Avail. KPI"], axis = 1
    )

restructured_ytd_availability_kpis["Time-based System Avail. KPI"] = restructured_ytd_availability_kpis.apply(
    lambda row: '🟠' if (row["Time-based System Avail. KPI"] != '🔴' and 
        row["Time-based System Avail."] < row["Time-based System Avail. P95"]) 
    else row["Time-based System Avail. KPI"], axis = 1
    )

restructured_ytd_availability_kpis["Time-based System Avail. KPI"] = restructured_ytd_availability_kpis.apply(
    lambda row: '🟢' if (row["Time-based System Avail. KPI"] == '⚫' and row["Time-based System Avail."] >= row["Time-based System Avail. P95"]) 
    else row["Time-based System Avail. KPI"], axis = 1
    )

restructured_ytd_availability_kpis["Time-based System Avail. KPI"] = restructured_ytd_availability_kpis.apply(
    lambda row: row["Time-based System Avail. KPI"]+" ("+str(round(row["Time-based System Avail."],1))+"%)" if row["Time-based System Avail. KPI"] != '⚫' 
    else row["Time-based System Avail. KPI"], axis = 1
    )


#Production based System Avail KPI
restructured_ytd_availability_kpis["Production-based System Avail. KPI"] = restructured_ytd_availability_kpis.apply(
    lambda row: '⚫', axis = 1
    )

restructured_ytd_availability_kpis["Production-based System Avail. KPI"] = restructured_ytd_availability_kpis.apply(
    lambda row: '🔴' if row["Production-based System Avail."]<row["Production-based System Avail. P99"] 
    else row["Production-based System Avail. KPI"], axis = 1
    )

restructured_ytd_availability_kpis["Production-based System Avail. KPI"] = restructured_ytd_availability_kpis.apply(
    lambda row: '🟠' if (row["Production-based System Avail. KPI"] != '🔴' and row["Production-based System Avail."] < row["Production-based System Avail. P95"]) 
    else row["Production-based System Avail. KPI"], axis = 1
    )

restructured_ytd_availability_kpis["Production-based System Avail. KPI"] = restructured_ytd_availability_kpis.apply(
    lambda row: '🟢' if (row["Production-based System Avail. KPI"] == '⚫' and row["Production-based System Avail."] >= row["Production-based System Avail. P95"]) 
    else row["Production-based System Avail. KPI"],axis = 1
    )

restructured_ytd_availability_kpis["Production-based System Avail. KPI"] = restructured_ytd_availability_kpis.apply(
    lambda row: row["Production-based System Avail. KPI"]+" ("+str(round(row["Production-based System Avail."],1))+"%)" if row["Production-based System Avail. KPI"] != '⚫' 
    else row["Production-based System Avail. KPI"],axis = 1
    )

#Add manufacturer informations
availability_kpis["Manufacturer"] = availability_kpis["Wind turbine"].apply(
    lambda x: wt_info_frame.loc[wt_info_frame['Wind turbine'] == x]['Manufacturer'].iloc[0]
    )
restructured_ytd_availability_kpis["Manufacturer"] = restructured_ytd_availability_kpis["Wind turbine"].apply(
    lambda x: wt_info_frame.loc[wt_info_frame['Wind turbine'] == x]['Manufacturer'].iloc[0]
    )

#Add wind farm information based on wind turbines
availability_kpis["Wind farm"] = availability_kpis["Wind turbine"].apply(
    lambda x: wt_info_frame.loc[wt_info_frame['Wind turbine'] == x]['Wind farm'].iloc[0]
    )

#Add country information for each winf farm
availability_kpis["Country"] = availability_kpis["Wind farm"].apply(
    lambda x: wf_coordinates_df.at[x,"Country"] if x in wf_coordinates_df.index else "Unknown"
    )

#Create DataFrames to store status code statistics
status_code_stats = pd.DataFrame(index=list(set(availability_kpis["Wind farm"].tolist())),columns=["Counts","Durations"])
status_code_info = pd.DataFrame(index=list(set(availability_kpis["Wind farm"].tolist())),columns=["Info"])

#Loop through each unique wind farm to gather status code statistics
for wf in list(set(availability_kpis["Wind farm"].tolist())):
    wf_status_code_counts = pd.DataFrame()
    wf_status_code_durations = pd.DataFrame()
    wf_status_code_info = pd.DataFrame()

    #Filter data for the current wind farm
    tmp = availability_kpis.loc[availability_kpis["Wind farm"] == wf]

    #del availability_kpis
    gc.collect()

    for item in range(len(tmp.index)):
        try:
            #extract details for each wind farm status code
            tmp2=tmp["details2"].iloc[item]
            tmp2=pd.DataFrame(tmp2)
            wf_status_code_info = pd.concat([wf_status_code_info,tmp2.T])
            tmp3=tmp2.loc["Count"]
            wf_status_code_counts = pd.concat([wf_status_code_counts,tmp3],axis=1)
            tmp4=tmp2.loc["Duration"]
            wf_status_code_durations = pd.concat([wf_status_code_durations,tmp4],axis=1)

        except:
            pass

    #Process unique status codes for the wind farm        
    unique_codes=list(wf_status_code_info.index.values)
    wf_status_code_info["Global contract category"]=wf_status_code_info["Global contract category"].astype(str)

    for unique_code in unique_codes:
        code_global_cats=wf_status_code_info.loc[unique_code]
        try:
            #Get unique global categories for each status code
            code_global_cats=list(set(code_global_cats["Global contract category"].to_list()))
            code_global_cats=', '.join(code_global_cats)

            wf_status_code_info.loc[unique_code,"Global contract category"]=code_global_cats
        except:
            code_global_cats=code_global_cats["Global contract category"]
            wf_status_code_info.loc[unique_code,"Global contract category"]=code_global_cats

    #Remove duplicate entries and drop unnecessary columns        
    wf_status_code_info=wf_status_code_info.loc[~wf_status_code_info.index.duplicated(keep='first')]
    wf_status_code_info=wf_status_code_info.drop(columns=["Count","Loss"])

    #Fill NaN
    wf_status_code_counts=wf_status_code_counts.fillna(0)
    wf_status_code_durations=wf_status_code_durations.fillna(0)

#Calculate percentils 
    wf_status_code_counts["ValsList"] = wf_status_code_counts.apply(
        lambda row: row.to_list(),axis = 1
        )
    wf_status_code_counts["ValsList"] = wf_status_code_counts["ValsList"].apply(
        lambda x: sorted(x)
        )
    wf_status_code_counts["P99"] = wf_status_code_counts["ValsList"].apply(
        lambda x: x[int(len(x) * 99/100)]
        )
    wf_status_code_counts["P95"] = wf_status_code_counts["ValsList"].apply(
        lambda x: x[int(len(x) * 95/100)]
        )
    wf_status_code_counts = wf_status_code_counts[["P99","P95"]]

    #Calculte percentiles for durations
    wf_status_code_durations["ValsList"] = wf_status_code_durations.apply(
        lambda row: row.to_list(),axis = 1
        )
    wf_status_code_durations["ValsList"] = wf_status_code_durations["ValsList"].apply(
        lambda x: sorted(x)
        )
    wf_status_code_durations["P99"] = wf_status_code_durations["ValsList"].apply(
        lambda x: x[int(len(x) * 99/100)]
        )
    wf_status_code_durations["P95"] = wf_status_code_durations["ValsList"].apply(
        lambda x: x[int(len(x) * 95/100)]
        )
    wf_status_code_durations = wf_status_code_durations[["P99","P95"]]

    #Store statistics for current WF
    status_code_stats.at[wf,"Counts"] = wf_status_code_counts.to_dict('index')
    status_code_stats.at[wf,"Durations"] = wf_status_code_durations.to_dict('index')
    status_code_info.at[wf,"Info"] = wf_status_code_info.to_dict('index')

#Icons according to parameters
availability_kpis["Status Codes KPIb"] = availability_kpis.apply(
    lambda row: {item: '🟠' if ((row["details2"][item]["Count"]> status_code_stats["Counts"].loc[row["Wind farm"]][item]["P95"] and 
        row["details2"][item]["Count"]>2)|(row["details2"][item]["Duration"] > status_code_stats["Durations"].loc[row["Wind farm"]][item]["P95"] and 
        row["details2"][item]["Duration"]>60*60)) 
    else '🟢' for item in row["details2"].keys()},axis = 1
    )

availability_kpis["Status Codes KPIb"] = availability_kpis.apply(
    lambda row: {item: '🔴' if ((row["details2"][item]["Count"] > status_code_stats["Counts"].loc[row["Wind farm"]][item]["P99"] and 
        row["details2"][item]["Count"] > 2)|(row["details2"][item]["Duration"] > status_code_stats["Durations"].loc[row["Wind farm"]][item]["P99"] and 
        row["details2"][item]["Duration"] > 60 * 60)) 
    else row["Status Codes KPIb"][item] for item in row["details2"].keys()},axis = 1
    )

availability_kpis["tmp"] = availability_kpis["Status Codes KPIb"].apply(
    lambda x: [item if x[item]!= '🟢' else np.nan for item in x.keys()]
    )

#Identify recurring status issues for each win turbine
for wt in list(set(availability_kpis["Wind turbine"].tolist())):
    if "Ballywater" in wt:
        tmp = availability_kpis.loc[availability_kpis["Wind turbine"] == wt]
        tmp["recurring_status_issues"] = tmp.apply(
            lambda row: [item for item in row["tmp"] if isinstance(row["tmp"], list) == True],axis = 1
            )
        availability_kpis.loc[availability_kpis["Wind turbine"] == wt,"recurring_status_issues"] = tmp["tmp"]
    else:
        tmp = availability_kpis.loc[availability_kpis["Wind turbine"] == wt]
        tmp["tmp_shifted_once"] = tmp["tmp"].shift(1)
        tmp["tmp_shifted_twice"] = tmp["tmp_shifted_once"].shift(1)
        tmp["recurring_status_issues"] = tmp.apply(
            lambda row: [item if (isinstance(row["tmp_shifted_once"], list) == True and 
                isinstance(row["tmp_shifted_twice"], list) == True and 
                item in row["tmp_shifted_once"] and 
                item in row["tmp_shifted_twice"]) else np.nan for item in row["tmp"] if isinstance(row["tmp"], list) == True],axis = 1
            )
        tmp["recurring_status_issues"] = tmp["recurring_status_issues"].apply(
            lambda x: [item for item in x if pd.isnull(item) == False]
            )
        tmp["recurring_status_issues_shifted_once_backwards"] = tmp["recurring_status_issues"].shift(-1)
        tmp["recurring_status_issues_shifted_twice_backwards"] = tmp["recurring_status_issues_shifted_once_backwards"].shift(-1)
        tmp["recurring_status_issues"] = tmp.apply(
            lambda row: list(set(row["recurring_status_issues"] + row["recurring_status_issues_shifted_once_backwards"])) 
            if isinstance(row["recurring_status_issues_shifted_once_backwards"], list) == True 
            else row["recurring_status_issues"],axis = 1
            )
        tmp["recurring_status_issues"] = tmp.apply(
            lambda row: list(set(row["recurring_status_issues"] + row["recurring_status_issues_shifted_twice_backwards"])) 
            if isinstance(row["recurring_status_issues_shifted_twice_backwards"], list) == True else row["recurring_status_issues"],axis = 1
            )
        availability_kpis.loc[availability_kpis["Wind turbine"] == wt,"recurring_status_issues"] = tmp["recurring_status_issues"]

del tmp

availability_kpis["Status Codes KPIb"] = availability_kpis.apply(
    lambda row: {item: row["Status Codes KPIb"][item] 
    for item in row["recurring_status_issues"] 
    if (pd.isnull(item) == False and item in row["Status Codes KPIb"])} 
    if isinstance(row["recurring_status_issues"], list) == True 
    else {}, axis = 1
    )

#Generate KPI data for viz - icons + color
availability_kpis["Status Codes KPIc"] = availability_kpis["Status Codes KPIb"].apply(
    lambda x: [x[item] for item in x.keys()]
    )
availability_kpis["Status Codes KPI"] = availability_kpis["Status Codes KPIc"].apply(
    lambda x: '🔴' if '🔴' in x else ('🟠' if '🟠' in x else '🟢')
    )
availability_kpis["Status Codes KPI Graph Colors"] = availability_kpis["Status Codes KPIc"].apply(
    lambda x: 'red' if '🔴' in x else ('orange' if '🟠' in x else 'green')
    )

#Construct detailed KPI data for each wind turbine
availability_kpis["Status Codes KPI Data"] = availability_kpis.apply(
    lambda row: {item: {
    "Color":row["Status Codes KPIb"][item],
    "Message":status_code_info["Info"].loc[row["Wind farm"]][item]["Message"],
    "Category":status_code_info["Info"].loc[row["Wind farm"]][item]["Category"],
    "Global contract category":status_code_info["Info"].loc[row["Wind farm"]][item]["Global contract category"],
    "Duration":row["details2"][item]["Duration"],
    "Duration P95":status_code_stats["Durations"].loc[row["Wind farm"]][item]["P95"],
    "Duration P99":status_code_stats["Durations"].loc[row["Wind farm"]][item]["P99"],
    "Count":row["details2"][item]["Count"],
    "Count P95":status_code_stats["Counts"].loc[row["Wind farm"]][item]["P95"],
    "Count P99":status_code_stats["Counts"].loc[row["Wind farm"]][item]["P99"]} 
    for item in row["recurring_status_issues"] 
    if (pd.isnull(item) == False and item in row["Status Codes KPIb"])},axis = 1
    )

#Create DataFrames to store statistics
warning_stats=pd.DataFrame(index=list(set(availability_kpis["Wind farm"].tolist())),columns=["Counts","Durations"])
warning_info=pd.DataFrame(index=list(set(availability_kpis["Wind farm"].tolist())),columns=["Info"])

#Loop through each unique wind farm to gather warning statistics
for wf in list(set(availability_kpis["Wind farm"].tolist())):
    wf_warning_counts=pd.DataFrame()
    wf_warning_durations=pd.DataFrame()
    wf_warning_info=pd.DataFrame()

    #Filter data for the current wind farm
    tmp=availability_kpis.loc[availability_kpis["Wind farm"]==wf]

    #Loop through each row of the filtered DataFrame
    for item in range(len(tmp.index)):

        try:
            #Extract warning details
            tmp2=tmp["details3"].iloc[item]
            tmp2=pd.DataFrame(tmp2)
            wf_warning_info=pd.concat([wf_warning_info,tmp2.T])

            #Collect counts and durations
            tmp3=tmp2.loc["Count"]
            wf_warning_counts=pd.concat([wf_warning_counts,tmp3],axis=1)
            tmp4=tmp2.loc["Duration"]
            wf_warning_durations=pd.concat([wf_warning_durations,tmp4],axis=1)

        except:
            pass

    #Process unique warning codes for the wind farm        
    unique_codes=list(wf_warning_info.index.values)
    wf_warning_info["Global contract category"]=wf_warning_info["Global contract category"].astype(str)

    for unique_code in unique_codes:
        code_global_cats=wf_warning_info.loc[unique_code]
        try:
            #Get unique global categories for each warning code
            code_global_cats=list(set(code_global_cats["Global contract category"].to_list()))
            code_global_cats=', '.join(code_global_cats)
            wf_warning_info.loc[unique_code,"Global contract category"]=code_global_cats
        except:
            code_global_cats=code_global_cats["Global contract category"]
            wf_warning_info.loc[unique_code,"Global contract category"]=code_global_cats

    #remùove duplicate entries and drop column Count        
    wf_warning_info=wf_warning_info.loc[~wf_warning_info.index.duplicated(keep='first')]
    wf_warning_info=wf_warning_info.drop(columns=["Count"])

    #Fill NaN values
    wf_warning_counts=wf_warning_counts.fillna(0)
    wf_warning_durations=wf_warning_durations.fillna(0)

   #Calculate percentiles for durations
    wf_warning_counts["ValsList"] = wf_warning_counts.apply(
        lambda row: row.to_list(),axis = 1
        )
    wf_warning_counts["ValsList"] = wf_warning_counts["ValsList"].apply(
        lambda x: sorted(x)
        )
    wf_warning_counts["P99"] = wf_warning_counts["ValsList"].apply(
        lambda x: x[int(len(x)*99/100)]
        )
    wf_warning_counts["P95"] = wf_warning_counts["ValsList"].apply(
        lambda x: x[int(len(x)*95/100)]
        )
    wf_warning_counts = wf_warning_counts[["P99","P95"]]

    #Calculte percentiles for durations
    wf_warning_durations["ValsList"] = wf_warning_durations.apply(
        lambda row: row.to_list(),axis = 1
        )
    wf_warning_durations["ValsList"] = wf_warning_durations["ValsList"].apply(
        lambda x: sorted(x)
        )
    wf_warning_durations["P99"] = wf_warning_durations["ValsList"].apply(
        lambda x: x[int(len(x)*99/100)]
        )
    wf_warning_durations["P95"] = wf_warning_durations["ValsList"].apply(
        lambda x: x[int(len(x)*95/100)]
        )
    wf_warning_durations = wf_warning_durations[["P99","P95"]]

    #Store statistics for the current wind farm
    warning_stats.at[wf,"Counts"]=wf_warning_counts.to_dict('index')
    warning_stats.at[wf,"Durations"]=wf_warning_durations.to_dict('index')
    warning_info.at[wf,"Info"]=wf_warning_info.to_dict('index')

#Update Warning KPIb with critical warnings based on P99 thresolds
availability_kpis["Warnings KPIb"] = availability_kpis.apply(
    lambda row: {item: '🟠' if ((row["details3"][item]["Count"] > warning_stats["Counts"].loc[row["Wind farm"]][item]["P95"] and 
        row["details3"][item]["Count"] > 2)|(row["details3"][item]["Duration"] > warning_stats["Durations"].loc[row["Wind farm"]][item]["P95"] and 
        row["details3"][item]["Duration"] > 60 * 60)) 
    else '🟢' for item in row["details3"].keys()}, axis = 1
    )

availability_kpis["Warnings KPIb"] = availability_kpis.apply(
    lambda row: {item: '🔴' if ((row["details3"][item]["Count"] > warning_stats["Counts"].loc[row["Wind farm"]][item]["P99"] and 
        row["details3"][item]["Count"]>2)|(row["details3"][item]["Duration"]>warning_stats["Durations"].loc[row["Wind farm"]][item]["P99"] and 
        row["details3"][item]["Duration"]>60*60)) 
    else row["Warnings KPIb"][item] for item in row["details3"].keys()},axis = 1
    )

#Create a temporary column 'temp2' to store non-green warnings
availability_kpis["tmp2"] = availability_kpis["Warnings KPIb"].apply(
    lambda x: [item if x[item]!= '🟢' else np.nan for item in x.keys()]
    )


#Loop through each unique wind turbine
for wt in list(set(availability_kpis["Wind turbine"].tolist())):

    if "Ballywater" in wt: #Specific case
        tmp = availability_kpis.loc[availability_kpis["Wind turbine"] == wt]
        tmp["recurring_warning_issues"] = tmp.apply(
            lambda row: [item for item in row["tmp"] if isinstance(row["tmp"], list) == True],axis = 1
            )
        availability_kpis.loc[availability_kpis["Wind turbine"] == wt,"recurring_warning_issues"] = tmp["tmp"]

    else: #the others (than Ballywater)
        tmp = availability_kpis.loc[availability_kpis["Wind turbine"] == wt]
        tmp["tmp2_shifted_once"] = tmp["tmp2"].shift(1)
        tmp["tmp2_shifted_twice"] = tmp["tmp2_shifted_once"].shift(1)

        #Identify recurring warning issues by checking against shifted lists
        tmp["recurring_warning_issues"] = tmp.apply(
            lambda row: [item if (isinstance(row["tmp2_shifted_once"], list) == True and 
                isinstance(row["tmp2_shifted_twice"], list) == True and 
                item in row["tmp2_shifted_once"] and item in row["tmp2_shifted_twice"]) 
            else np.nan for item in row["tmp2"] if isinstance(row["tmp2"], list) == True],axis = 1
            )

        #Remove NaN values
        tmp["recurring_warning_issues"] = tmp["recurring_warning_issues"].apply(
            lambda x: [item for item in x if pd.isnull(item) == False]
            )

        #Shift the recurring issues backwards
        tmp["recurring_warning_issues_shifted_once_backwards"] = tmp["recurring_warning_issues"].shift(-1)
        tmp["recurring_warning_issues_shifted_twice_backwards"] = tmp["recurring_warning_issues_shifted_once_backwards"].shift(-1)

        #Clean up
        tmp["recurring_warning_issues"] = tmp.apply(
            lambda row: list(set(row["recurring_warning_issues"] + row["recurring_warning_issues_shifted_once_backwards"])) 
            if isinstance(row["recurring_warning_issues_shifted_once_backwards"], list) == True 
            else row["recurring_warning_issues"],axis = 1
            )
        tmp["recurring_warning_issues"] = tmp.apply(
            lambda row: list(set(row["recurring_warning_issues"] + row["recurring_warning_issues_shifted_twice_backwards"])) 
            if isinstance(row["recurring_warning_issues_shifted_twice_backwards"], list) == True 
            else row["recurring_warning_issues"], axis = 1
            )

        #Store clean up data
        availability_kpis.loc[availability_kpis["Wind turbine"] == wt,"recurring_warning_issues"] = tmp["recurring_warning_issues"]

#Include only relevant warinings
availability_kpis["Warnings KPIb"] = availability_kpis.apply(
    lambda row: {item: row["Warnings KPIb"][item] for item in row["recurring_warning_issues"] 
    if (pd.isnull(item) == False and item in row["Warnings KPIb"])} 
    if isinstance(row["recurring_warning_issues"], list) == True 
    else {},axis = 1
    )

#Create Warnings KPIc to gather all warning values into a list
availability_kpis["Warnings KPIc"] = availability_kpis["Warnings KPIb"].apply(
    lambda x: [x[item] for item in x.keys()]
    )

#Determine the overall 'Warning KPI' based on the presence of Warnings KPIc
availability_kpis["Warnings KPI"] = availability_kpis["Warnings KPIc"].apply(
    lambda x: '🔴' if '🔴' in x else ('🟠' if '🟠' in x else '🟢')
    )

#Set graph colors
availability_kpis["Warnings KPI Graph Colors"] = availability_kpis["Warnings KPIc"].apply(
    lambda x: 'red' if '🔴' in x else ('orange' if '🟠' in x else 'green')
    )

#Compile detailed warning data into a structured format
availability_kpis["Warnings KPI Data"] = availability_kpis.apply(
    lambda row: {item: {
    "Color":row["Warnings KPIb"][item],
    "Message":warning_info["Info"].loc[row["Wind farm"]][item]["Message"],
    "Category":warning_info["Info"].loc[row["Wind farm"]][item]["Category"],
    "Global contract category":warning_info["Info"].loc[row["Wind farm"]][item]["Global contract category"],
    "Duration":row["details3"][item]["Duration"],
    "Duration P95":warning_stats["Durations"].loc[row["Wind farm"]][item]["P95"],
    "Duration P99":warning_stats["Durations"].loc[row["Wind farm"]][item]["P99"],
    "Count":row["details3"][item]["Count"],
    "Count P95":warning_stats["Counts"].loc[row["Wind farm"]][item]["P95"],
    "Count P99":warning_stats["Counts"].loc[row["Wind farm"]][item]["P99"]} for item in row["recurring_warning_issues"] 
    if (pd.isnull(item) == False and item in row["Warnings KPIb"])},axis = 1
    )

print("["+dt.today().strftime('%d/%m/%Y %H:%M')+"] [INFO] Successfully computed availability KPIs")

#Open csv file
with open('FilteredDataAndPrecomputedPerformanceKPIs.csv') as f:
    kpis = pd.read_csv(f,sep = '\t',header = 0,names = ["Month",
                                            "Wind farm","Wind turbine","Wind turbine type",
                                            "Current wind direction offset to true north (°)","Yaw misalignment P50 (°)",
                                            "Yaw misalignment P75 (°)","Yaw misalignment P25 (°)","Filtered data",
                                            "Monthly ambient temperature","Unclassified curtailment periods"]
                                            )


#Convert columns into the right type
#kpis = kpis.loc[lambda kpis: kpis["Wind farm"] != 'Markbygden ETT'] #Filter out the 'Markbygden ETT' wind farm (North Pole)
print(kpis["Wind farm"].unique())    
kpis["MonthAsDate"]=pd.to_datetime(kpis["Month"], format='%B %Y')
kpis=kpis.sort_values("MonthAsDate")
print(kpis.head())
print(kpis.columns)

kpis.index=range(len(kpis.index))
#kpis["DateAsDate"]=kpis["DateAsDate"].apply(lambda x: (x+relativedelta(months=-1)))
#kpis["Date"]=kpis["DateAsDate"].apply(lambda x: x.strftime('%B %Y'))

kpis["Asset Manager"]=kpis["Wind farm"].apply(lambda x: metadata.at[x,"AM"])
kpis["Manufacturer"]=kpis["Wind turbine"].apply(lambda x: wt_info_frame.loc[wt_info_frame['Wind turbine']==x]['Manufacturer'].iloc[0])

#Assign 'Asset Manager' by looking at the metadata
kpis["Asset Manager"] = kpis["Wind farm"].apply(
    lambda x: metadata.at[x,"AM"]
    )

#Assign 'Manufacturer' by looking up the manufacturer based on the wind turbine's info
kpis["Manufacturer"] = kpis["Wind turbine"].apply(
    lambda x: wt_info_frame.loc[wt_info_frame['Wind turbine'] == x]['Manufacturer'].iloc[0]
    )

#Compute the median monthly ambient temp per wind farm and month
for wf in list(set(kpis["Wind farm"].tolist())):
    for month in list(set(kpis["Month"].tolist())):
        try:
            tmp = kpis.loc[(kpis["Wind farm"] == wf) & (kpis["Month"] == month)]
            med = tmp["Monthly ambient temperature"].median()
            kpis.loc[(kpis["Wind farm"] == wf) & (kpis["Month"] == month),"WF median monthly ambient temperature"] = med
        except:
            kpis.loc[(kpis["Wind farm"] == wf) & (kpis["Month"] == month),"WF median monthly ambient temperature"] = np.nan

#Apply right icons depending of the variation of the monthly ambient temp comparing to the WF
kpis["Ambient temperature KPI"] = kpis.apply(
    lambda row: '🔴' if abs(row["Monthly ambient temperature"] - row["WF median monthly ambient temperature"]) > 1 
    else '⚫', axis = 1
    )
kpis["Ambient temperature KPI"] = kpis.apply(
    lambda row: '🟠' if (row["Ambient temperature KPI"] != '🔴') and 
    (abs(row["Monthly ambient temperature"] - row["WF median monthly ambient temperature"]) > 0.5) 
    else row["Ambient temperature KPI"], axis = 1
    )
kpis["Ambient temperature KPI"] = kpis.apply(
    lambda row: '🟢' if row["Ambient temperature KPI"] == '⚫' and 
    (abs(row["Monthly ambient temperature"] - row["WF median monthly ambient temperature"]) <= 0.5) 
    else row["Ambient temperature KPI"], axis = 1
    )

#Detect static yaw misaligment for each wtg using PELT algo
for wt in list(set(kpis["Wind turbine"].tolist())):
    try:
        tmp = kpis.loc[kpis["Wind turbine"] == wt] #Specific wtg
        tmp.index = tmp["MonthAsDate"] #change index

        #Filter data where yaw misaligment is less than 5°
        tmp2 = tmp.loc[tmp["Yaw misalignment P25 (°)"] - tmp["Yaw misalignment P75 (°)"] < 5]
        tmp2.index = tmp2["MonthAsDate"]

        #Prepare sign data
        signal = tmp2["Yaw misalignment P50 (°)"]
        signal = signal.values.reshape(-1, 1)

        T, d = signal.shape

        sigma = np.std(signal)

        bic = 2 * sigma * sigma * np.log(T) * (d + 1)

        #Apply PELT
        algo = rpt.Pelt(model = "l2", jump = 1, min_size = 1).fit(signal) #PELT = Pruned Exact Linear Time used for change-point detection

        result = algo.predict(pen = bic)

        #For each segment detected, compute the median yaw misaligment
        if len(result) > 1:
            for i in range(len(result)):
                if i == 0:
                    period_start = tmp.index[0]
                else:
                    period_start = tmp2.index[result[i - 1]]
                if i == len(result) - 1:
                    period_end = tmp.index[-1] + relativedelta(months = 1)
                else:
                    period_end = tmp2.index[result[i]]

                #Calculate median yaw misaligment for each segment
                tmp3 = tmp2.loc[(tmp2.index >= period_start) & (tmp2.index < period_end)]
                kpis.loc[
                (kpis["Wind turbine"] == wt) & (kpis["MonthAsDate"] >= period_start) & (kpis["MonthAsDate"] <= period_end),
                "Static yaw misalignment best guess"
                ] = round(tmp3["Yaw misalignment P50 (°)"].median(),1)
        else:
            #Handle case where only one segment is detected
            period_start = tmp.index[0]
            period_end = tmp.index[-1]+relativedelta(months = 1)
            tmp3 = tmp2.loc[(tmp2.index >= period_start) & (tmp2.index<period_end)]
            kpis.loc[
            (kpis["Wind turbine"] == wt) & (kpis["MonthAsDate"] >= period_start) & (kpis["MonthAsDate"] <= period_end),
            "Static yaw misalignment best guess"
            ] = round(tmp3["Yaw misalignment P50 (°)"].median(),1)
        
        importlib.reload(rpt) #Reloading change-point detectin module                  

    except:
        kpis.loc[kpis["Wind turbine"] == wt,"Static yaw misalignment best guess"] = np.nan

del availability_kpis
del tmp
del tmp2
del tmp3

#Assign Static Yaw Misaligment KPIs based on thresholds
kpis["Static yaw misalignment KPI"] = kpis.apply(
    lambda row: '🔴' if (row["Static yaw misalignment best guess"] > 5)|(row["Static yaw misalignment best guess"] <- 5) 
    else '⚫',axis = 1
    )
kpis["Static yaw misalignment KPI"] = kpis.apply(
    lambda row: '🟠' if row["Static yaw misalignment KPI"] != '🔴' and 
    (row["Static yaw misalignment best guess"] >= 2.5)|(row["Static yaw misalignment best guess"] <= -2.5) 
    else row["Static yaw misalignment KPI"],axis = 1
    )
kpis["Static yaw misalignment KPI"] = kpis.apply(
    lambda row: '🟢' if row["Static yaw misalignment KPI"] == '⚫' and 
    (row["Static yaw misalignment best guess"] < 2.5) & (row["Static yaw misalignment best guess"] >- 2.5)
     else row["Static yaw misalignment KPI"],axis = 1
     )

#Normalize wind direction offsets to a range of [-180°, 180°]
kpis["Current wind direction offset to true north (°)"] = ((kpis["Current wind direction offset to true north (°)"]+180)%360)-180

#Compute SCADA Wind Direction KPI based on wind direction offset thresholds
kpis["SCADA wind direction KPI"] = kpis.apply(
    lambda row: '🔴' if (row["Current wind direction offset to true north (°)"] > 30)|(row["Current wind direction offset to true north (°)"] <- 30) 
    else '⚫',axis = 1
    )
kpis["SCADA wind direction KPI"] = kpis.apply(
    lambda row: '🟠' if row["SCADA wind direction KPI"] != '🔴' and 
    (row["Current wind direction offset to true north (°)"] > 15)|(row["Current wind direction offset to true north (°)"] <- 15) 
    else row["SCADA wind direction KPI"],axis = 1
    )
kpis["SCADA wind direction KPI"] = kpis.apply(
    lambda row: '🟢' if row["SCADA wind direction KPI"] == '⚫' and 
    (row["Current wind direction offset to true north (°)"] <= 15) & (row["Current wind direction offset to true north (°)"] >= -15) 
    else row["SCADA wind direction KPI"],axis = 1
    )

#Extract SCADA production data (Energy Export) for each wind turbine and month
kpis["SCADA Prod"] = kpis.apply(
    lambda row: scada_prods.loc[(scada_prods["WT"] == row["Wind turbine"]) & (scada_prods["Month"] == row["MonthAsDate"]),"Energy Export"].iloc[0] 
    if scada_prods.loc[(scada_prods["WT"] == row["Wind turbine"]) & (scada_prods["Month"] == row["MonthAsDate"]),"Energy Export"].shape[0]>0 
    else np.nan,axis = 1
    )

#Replace 'nan' with  numpy Nan in the 'Filtered data' column
kpis["Filtered data"] = kpis["Filtered data"].apply(
    lambda x: x.replace("nan",'np.nan')
    )

#Convert 'Filtered data' to string, then evaluate it back to its original format
kpis["Filtered data"] = kpis["Filtered data"].astype(str)
kpis["Filtered data"] = kpis["Filtered data"].apply(
    lambda x: eval(x)
    )

#Create dataframes for relationships and dynamic yaw misaligment for each wind turbine type
wt_types_relationships = pd.DataFrame(index = list(set(kpis["Wind turbine type"].tolist())),columns = ["Pitch vs Power",
                                                                                    "Pitch vs RPM","RPM vs Power","Power vs Speed","Pitch vs Speed",
                                                                                    "RPM vs Speed","Power vs RPM"]
                                                                                    )
wt_types_dymb=pd.DataFrame(index=list(set(kpis["Wind turbine type"].tolist())),columns=["Dynamic yaw misalignment"])


#Compute power-angle relationships for each wind turbine type
for wt_type in list(set(kpis["Wind turbine type"].tolist())):
    wt_type_filtered_data = pd.DataFrame()
    tmp = kpis.loc[kpis["Wind turbine type"] == wt_type]

    #Aggregate filtered data for each turbine type
    for item in range(len(tmp.index)):
        tmp2 = tmp["Filtered data"].iloc[item]
        tmp2 = pd.DataFrame(tmp2)
        wt_type_filtered_data = pd.concat([wt_type_filtered_data,tmp2])

    del tmp
    del tmp2

    # Check if yaw misaligment data exists and filter by valid yaw range    
    check_if_yaw_misalignment_data = wt_type_filtered_data["Yaw misalignment"].count()

    if check_if_yaw_misalignment_data > 0:
        wt_type_filtered_data = wt_type_filtered_data.loc[(wt_type_filtered_data["Yaw misalignment"] >= -5) & (wt_type_filtered_data["Yaw misalignment"] <= 5)]

    try:
        #Power and Blade angle
        bin_value = 50
        binnedtable = pd.DataFrame()

        #Prepare data for binning
        tempnewdf = wt_type_filtered_data[["Power","Blade angle"]]
        tempnewdf = tempnewdf.dropna()
        tempnewdf["Blade angle"] = tempnewdf["Blade angle"].apply(
            lambda x: ((x+180)%360)-180
            )
        tempnewdf["bin"] = (tempnewdf["Power"]-(bin_value/2))/bin_value
        tempnewdf["bin"] = tempnewdf["bin"].astype("int64")

        #Compute averages and standard deviations for each bin
        ultratempone = tempnewdf[["bin","Power"]]
        ultratemptwo = tempnewdf[["bin","Blade angle"]]

        del tempnewdf

        tempbinnedtable1 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable2 = ultratemptwo.groupby(["bin"]).apply(median_angle)

        tempnewdf2 = pd.concat([tempbinnedtable1, tempbinnedtable2[["Blade angle"]]], axis = 1)
        tempnewdf2 = tempnewdf2.rename(columns = {"Blade angle":'Avg'})

        del tempbinnedtable1
        del tempbinnedtable2

        #tempbinnedtable3 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable4 = ultratemptwo.groupby(["bin"]).std()

        tempnewdf3 = pd.concat([tempbinnedtable3,tempbinnedtable4], axis = 1)
        tempnewdf3 = tempnewdf3.rename(columns = {"Blade angle":'Stdev'})

        del tempbinnedtable4

        #Concat both previous temp df into one table
        tempnewdf4 = pd.concat([tempnewdf2,tempnewdf3], axis = 1)

        del tempnewdf2
        del tempnewdf3

        #tempbinnedtable5 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable6 = ultratemptwo.groupby(["bin"]).count()

        del ultratempone
        del ultratemptwo

        tempnewdf5 = pd.concat([tempbinnedtable1,tempbinnedtable6], axis = 1)
        tempnewdf5 = tempnewdf5.rename(columns = {"Blade angle":'Count'})

        del tempbinnedtable6

        tempnewdf6 = pd.concat([tempnewdf4,tempnewdf5], axis = 1)
        tempnewdf6 = tempnewdf6.loc[tempnewdf6["Count"]>5] #filter bins

        del tempnewdf5

        tempnewdf4 = tempnewdf6.drop(columns = ["Count"])
        tempnewdf4 = tempnewdf4.loc[:,~tempnewdf4.columns.duplicated()]
        tempnewdf4.index = tempnewdf4["Power"]

        del tempnewdf6

        #Interpolate
        if tempnewdf4.empty == False:
            steps = np.around(np.arange(tempnewdf4["Power"].max()/20, tempnewdf4["Power"].max(), 1),0).tolist()
            steps_tmp = pd.DataFrame(index = steps,columns = tempnewdf4.columns)

            tempnewdf4 = tempnewdf4._append(steps_tmp)
            tempnewdf4.sort_index(inplace = True)
            tempnewdf4 = tempnewdf4.interpolate(method = "index")
            tempnewdf4 = tempnewdf4.loc[steps]
            tempnewdf4 = tempnewdf4.dropna()
            tempnewdf4 = tempnewdf4.loc[~tempnewdf4.index.duplicated(keep = 'first')]

        #Save results
        binnedtable = tempnewdf4[["Avg","Stdev"]]
        binnedtable.sort_index(inplace = True)

        del tempnewdf4

        wt_types_relationships.at[wt_type,"Pitch vs Power"] = binnedtable.to_dict()

    except:
        wt_types_relationships.at[wt_type,"Pitch vs Power"] = {}

    del binnedtable
    gc.collect()

    try:

        #Rotor Speed and Blade angle
        bin_value = 0.1
        binnedtable = pd.DataFrame()

        #Filter relevant columns & create temp df
        tempnewdf = wt_type_filtered_data[["Rotor speed","Blade angle"]]
        tempnewdf = tempnewdf.dropna()

        #Normalize Blade angle
        tempnewdf["Blade angle"] = tempnewdf["Blade angle"].apply(
            lambda x: ((x+180)%360)-180
            )
#        tempnewdf = tempnewdf.loc[tempnewdf["Blade angle"]<20]

        #Bin the rotor speed
        tempnewdf["bin"] = (tempnewdf["Rotor speed"]-(bin_value/2))/bin_value
        tempnewdf["bin"] = tempnewdf["bin"].astype("int64")

        #Create separate df
        ultratempone = tempnewdf[["bin","Rotor speed"]]
        ultratemptwo = tempnewdf[["bin","Blade angle"]]

        del tempnewdf

        tempbinnedtable1 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable2 = ultratemptwo.groupby(["bin"]).apply(median_angle)

        #Combine
        tempnewdf2 = pd.concat([tempbinnedtable1, tempbinnedtable2[["Blade angle"]]], axis = 1)
        tempnewdf2 = tempnewdf2.rename(columns = {"Blade angle":'Avg'})

        del tempbinnedtable2

        #tempbinnedtable3 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable4 = ultratemptwo.groupby(["bin"]).std()

        del ultratempone
        del ultratemptwo

        tempnewdf3 = pd.concat([tempbinnedtable1,tempbinnedtable4], axis = 1)
        tempnewdf3 = tempnewdf3.rename(columns = {"Blade angle":'Stdev'})

        del tempbinnedtable4

        tempnewdf4 = pd.concat([tempnewdf2,tempnewdf3], axis = 1)

        del tempnewdf2
        del tempnewdf3

        #tempbinnedtable5 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable6 = ultratemptwo.groupby(["bin"]).count()

        del ultratemptwo

        tempnewdf5 = pd.concat([tempbinnedtable1,tempbinnedtable6], axis = 1)

        del tempbinnedtable1
        del tempbinnedtable6

        tempnewdf5 = tempnewdf5.rename(columns = {"Blade angle":'Count'})
        tempnewdf6 = pd.concat([tempnewdf4,tempnewdf5], axis = 1)

        del tempnewdf5

        tempnewdf6 = tempnewdf6.loc[tempnewdf6["Count"]>5]
        tempnewdf4 = tempnewdf6.drop(columns = ["Count"])

        del tempnewdf6

        tempnewdf4 = tempnewdf4.loc[:,~tempnewdf4.columns.duplicated()]
        tempnewdf4.index = tempnewdf4["Rotor speed"]

        #Interpolate missing values
        if tempnewdf4.empty == False:
            steps = np.around(np.arange(max(5,tempnewdf4["Rotor speed"].min()), tempnewdf4["Rotor speed"].max(),0.1),1).tolist()
            steps_tmp = pd.DataFrame(index = steps,columns = tempnewdf4.columns)

            tempnewdf4 = tempnewdf4._append(steps_tmp)
            tempnewdf4.sort_index(inplace = True)
            tempnewdf4 = tempnewdf4.interpolate(method = "index")
            tempnewdf4 = tempnewdf4.loc[steps]
            tempnewdf4 = tempnewdf4.dropna()
            tempnewdf4 = tempnewdf4.loc[~tempnewdf4.index.duplicated(keep = 'first')]

        binnedtable = tempnewdf4[["Avg","Stdev"]]
        binnedtable.sort_index(inplace = True)

        del tempnewdf4

        wt_types_relationships.at[wt_type,"Pitch vs RPM"] = binnedtable.to_dict()

    except:
        wt_types_relationships.at[wt_type,"Pitch vs RPM"] = {}

    del binnedtable
    gc.collect()

    #Power and Rotor speed
    bin_value = 50

    binnedtable = pd.DataFrame()

    tempnewdf = wt_type_filtered_data[["Power","Rotor speed"]]
    tempnewdf = tempnewdf.dropna()
    tempnewdf["bin"] = (tempnewdf["Power"]-(bin_value/2))/bin_value
    tempnewdf["bin"] = tempnewdf["bin"].astype("int64")

    ultratempone = tempnewdf[["bin","Power"]]
    ultratemptwo = tempnewdf[["bin","Rotor speed"]]

    del tempnewdf

    tempbinnedtable1 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable2 = ultratemptwo.groupby(["bin"]).median()

    del ultratempone

    tempnewdf2 = pd.concat([tempbinnedtable1, tempbinnedtable2[["Rotor speed"]]], axis = 1)
    tempnewdf2 = tempnewdf2.rename(columns = {"Rotor speed":'Avg'})

    del tempbinnedtable2

    #tempbinnedtable3 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable4 = ultratemptwo.groupby(["bin"]).std()

    tempnewdf3 = pd.concat([tempbinnedtable1,tempbinnedtable4], axis = 1)
    tempnewdf3 = tempnewdf3.rename(columns = {"Rotor speed":'Stdev'})
    tempnewdf4 = pd.concat([tempnewdf2,tempnewdf3], axis = 1)

    del tempnewdf2
    del tempnewdf3

    #tempbinnedtable5 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable6 = ultratemptwo.groupby(["bin"]).count()

    tempnewdf5 = pd.concat([tempbinnedtable1,tempbinnedtable6], axis = 1)
    tempnewdf5 = tempnewdf5.rename(columns = {"Rotor speed":'Count'})

    del tempbinnedtable1
    del tempbinnedtable6

    tempnewdf6 = pd.concat([tempnewdf4,tempnewdf5], axis = 1)
    tempnewdf6 = tempnewdf6.loc[tempnewdf6["Count"]>5]

    del tempnewdf5

    tempnewdf4 = tempnewdf6.drop(columns = ["Count"])
    tempnewdf4 = tempnewdf4.loc[:,~tempnewdf4.columns.duplicated()]
    tempnewdf4.index = tempnewdf4["Power"]

    del tempnewdf6

    if tempnewdf4.empty == False:
        steps = np.around(np.arange(tempnewdf4["Power"].max()/20,tempnewdf4["Power"].max(),1),0).tolist()
        steps_tmp = pd.DataFrame(index = steps,columns = tempnewdf4.columns)

        tempnewdf4 = tempnewdf4._append(steps_tmp)
        tempnewdf4.sort_index(inplace = True)
        tempnewdf4 = tempnewdf4.interpolate(method = "index")
        tempnewdf4 = tempnewdf4.loc[steps]
        tempnewdf4 = tempnewdf4.dropna()
        tempnewdf4 = tempnewdf4.loc[~tempnewdf4.index.duplicated(keep = 'first')]

    binnedtable = tempnewdf4[["Avg","Stdev"]]
    binnedtable.sort_index(inplace = True)

    del tempnewdf4

    wt_types_relationships.at[wt_type,"RPM vs Power"] = binnedtable.to_dict()

    del binnedtable
    gc.collect()

 #Corrected wind speed and Power
    bin_value = 0.1

    binnedtable = pd.DataFrame()

    tempnewdf = wt_type_filtered_data[["Corrected wind speed","Power"]]
    tempnewdf = tempnewdf.dropna()
    tempnewdf["bin"] = (tempnewdf["Corrected wind speed"]-(bin_value/2))/bin_value
    tempnewdf["bin"] = tempnewdf["bin"].astype("int64")

    ultratempone = tempnewdf[["bin","Corrected wind speed"]]
    ultratemptwo = tempnewdf[["bin","Power"]]

    del tempnewdf

    tempbinnedtable1 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable2 = ultratemptwo.groupby(["bin"]).median()

    tempnewdf2 = pd.concat([tempbinnedtable1, tempbinnedtable2[["Power"]]], axis = 1)
    tempnewdf2 = tempnewdf2.rename(columns = {"Power":'Avg'})

    del tempbinnedtable2

    #tempbinnedtable3 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable4 = ultratemptwo.groupby(["bin"]).std()

    tempnewdf3 = pd.concat([tempbinnedtable1,tempbinnedtable4], axis = 1)
    tempnewdf3 = tempnewdf3.rename(columns = {"Power":'Stdev'})

    del tempbinnedtable4

    tempnewdf4 = pd.concat([tempnewdf2,tempnewdf3], axis = 1)

    del tempnewdf2
    del tempnewdf3

    #tempbinnedtable5 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable6 = ultratemptwo.groupby(["bin"]).count()

    tempnewdf5 = pd.concat([tempbinnedtable1,tempbinnedtable6], axis = 1)
    tempnewdf5 = tempnewdf5.rename(columns = {"Power":'Count'})

    del tempbinnedtable6
    del tempbinnedtable1

    tempnewdf6 = pd.concat([tempnewdf4,tempnewdf5], axis = 1)
    tempnewdf6 = tempnewdf6.loc[tempnewdf6["Count"]>5]

    tempnewdf4 = tempnewdf6.drop(columns = ["Count"])

    del tempnewdf6
    del tempnewdf5

    tempnewdf4 = tempnewdf4.loc[:,~tempnewdf4.columns.duplicated()]
    tempnewdf4.index = tempnewdf4["Corrected wind speed"]

    if tempnewdf4.empty == False:
        steps = np.around(np.arange(3, min(20,tempnewdf4["Corrected wind speed"].max()),0.1),1).tolist()
        steps_tmp = pd.DataFrame(index = steps,columns = tempnewdf4.columns)

        tempnewdf4 = tempnewdf4._append(steps_tmp)
        tempnewdf4.sort_index(inplace = True)
        tempnewdf4 = tempnewdf4.interpolate(method = "index")
        tempnewdf4 = tempnewdf4.loc[steps]
        tempnewdf4 = tempnewdf4.dropna()
        tempnewdf4 = tempnewdf4.loc[~tempnewdf4.index.duplicated(keep = 'first')]

    binnedtable = tempnewdf4[["Avg","Stdev"]]
    binnedtable.sort_index(inplace = True)

    wt_types_relationships.at[wt_type,"Power vs Speed"] = binnedtable.to_dict()

    del tempnewdf4
    del binnedtable
    gc.collect()

    try:
        #Wind Speed & Blade angle
        bin_value = 0.1
        binnedtable = pd.DataFrame()

        tempnewdf = wt_type_filtered_data[["Wind speed","Blade angle"]]
        tempnewdf = tempnewdf.dropna()
        tempnewdf["Blade angle"] = tempnewdf["Blade angle"].apply(lambda x: ((x+180)%360)-180)
#        tempnewdf = tempnewdf.loc[tempnewdf["Blade angle"]<20]
        tempnewdf["bin"] = (tempnewdf["Wind speed"]-(bin_value/2))/bin_value
        tempnewdf["bin"] = tempnewdf["bin"].astype("int64")

        ultratempone = tempnewdf[["bin","Wind speed"]]
        ultratemptwo = tempnewdf[["bin","Blade angle"]]

        del tempnewdf

        tempbinnedtable1 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable2 = ultratemptwo.groupby(["bin"]).apply(median_angle)

        tempnewdf2 = pd.concat([tempbinnedtable1, tempbinnedtable2[["Blade angle"]]], axis = 1)
        tempnewdf2 = tempnewdf2.rename(columns = {"Blade angle":'Avg'})

        del tempbinnedtable2

        #tempbinnedtable3 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable4 = ultratemptwo.groupby(["bin"]).std()

        tempnewdf3 = pd.concat([tempbinnedtable1,tempbinnedtable4], axis = 1)
        tempnewdf3 = tempnewdf3.rename(columns = {"Blade angle":'Stdev'})

        del tempbinnedtable4

        tempnewdf4 = pd.concat([tempnewdf2,tempnewdf3], axis = 1)

        del tempnewdf2
        del tempnewdf3

        #tempbinnedtable5 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable6 = ultratemptwo.groupby(["bin"]).count()

        tempnewdf5 = pd.concat([tempbinnedtable1,tempbinnedtable6], axis = 1)
        tempnewdf5 = tempnewdf5.rename(columns = {"Blade angle":'Count'})

        del tempbinnedtable6
        del tempbinnedtable1

        tempnewdf6 = pd.concat([tempnewdf4,tempnewdf5], axis = 1)
        tempnewdf6 = tempnewdf6.loc[tempnewdf6["Count"]>5]

        del tempnewdf5

        tempnewdf4 = tempnewdf6.drop(columns = ["Count"])

        del tempnewdf6

        tempnewdf4 = tempnewdf4.loc[:,~tempnewdf4.columns.duplicated()]
        tempnewdf4.index = tempnewdf4["Wind speed"]

        if tempnewdf4.empty == False:
            steps = np.around(np.arange(3, min(20,tempnewdf4["Wind speed"].max()),0.1),1).tolist()
            steps_tmp = pd.DataFrame(index = steps,columns = tempnewdf4.columns)
            tempnewdf4 = tempnewdf4._append(steps_tmp)
            tempnewdf4.sort_index(inplace = True)
            tempnewdf4 = tempnewdf4.interpolate(method = "index")
            tempnewdf4 = tempnewdf4.loc[steps]
            tempnewdf4 = tempnewdf4.dropna()
            tempnewdf4 = tempnewdf4.loc[~tempnewdf4.index.duplicated(keep = 'first')]

        binnedtable = tempnewdf4[["Avg","Stdev"]]
        binnedtable.sort_index(inplace = True)

        del tempnewdf4

        wt_types_relationships.at[wt_type,"Pitch vs Speed"] = binnedtable.to_dict()

    except:
        wt_types_relationships.at[wt_type,"Pitch vs Speed"] = {}

    del binnedtable
    gc.collect()

    #Wind Speed & Rotor Speed
    bin_value = 0.1
    binnedtable = pd.DataFrame()

    tempnewdf = wt_type_filtered_data[["Wind speed","Rotor speed"]]
    tempnewdf = tempnewdf.dropna()

    tempnewdf["bin"] = (tempnewdf["Wind speed"]-(bin_value/2))/bin_value
    tempnewdf["bin"] = tempnewdf["bin"].astype("int64")

    ultratempone = tempnewdf[["bin","Wind speed"]]
    ultratemptwo = tempnewdf[["bin","Rotor speed"]]

    del tempnewdf

    tempbinnedtable1 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable2 = ultratemptwo.groupby(["bin"]).median()

    tempnewdf2 = pd.concat([tempbinnedtable1, tempbinnedtable2[["Rotor speed"]]], axis = 1)
    tempnewdf2 = tempnewdf2.rename(columns = {"Rotor speed":'Avg'})

    del tempbinnedtable2

    #tempbinnedtable3 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable4 = ultratemptwo.groupby(["bin"]).std()

    tempnewdf3 = pd.concat([tempbinnedtable1,tempbinnedtable4], axis = 1)
    tempnewdf3 = tempnewdf3.rename(columns = {"Rotor speed":'Stdev'})
    tempnewdf4 = pd.concat([tempnewdf2,tempnewdf3], axis = 1)

    del tempbinnedtable4
    del tempnewdf2
    del tempnewdf3

    #tempbinnedtable5 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable6 = ultratemptwo.groupby(["bin"]).count()

    tempnewdf5 = pd.concat([tempbinnedtable1,tempbinnedtable6], axis = 1)
    tempnewdf5 = tempnewdf5.rename(columns = {"Rotor speed":'Count'})

    del tempbinnedtable1
    del tempbinnedtable6

    tempnewdf6 = pd.concat([tempnewdf4,tempnewdf5], axis = 1)
    tempnewdf6 = tempnewdf6.loc[tempnewdf6["Count"]>5]

    tempnewdf4 = tempnewdf6.drop(columns = ["Count"])
    tempnewdf4 = tempnewdf4.loc[:,~tempnewdf4.columns.duplicated()]
    tempnewdf4.index = tempnewdf4["Wind speed"]

    if tempnewdf4.empty == False:
        steps = np.around(np.arange(3, min(20,tempnewdf4["Wind speed"].max()),0.1),1).tolist()
        steps_tmp = pd.DataFrame(index = steps,columns = tempnewdf4.columns)
        tempnewdf4 = tempnewdf4._append(steps_tmp)
        tempnewdf4.sort_index(inplace = True)
        tempnewdf4 = tempnewdf4.interpolate(method = "index")
        tempnewdf4 = tempnewdf4.loc[steps]
        tempnewdf4 = tempnewdf4.dropna()
        tempnewdf4 = tempnewdf4.loc[~tempnewdf4.index.duplicated(keep = 'first')]

    binnedtable = tempnewdf4[["Avg","Stdev"]]
    binnedtable.sort_index(inplace = True)

    del tempnewdf4

    wt_types_relationships.at[wt_type,"RPM vs Speed"] = binnedtable.to_dict()

    del binnedtable
    gc.collect()

    #Rotor Speed & Power
    bin_value = 0.1
    binnedtable = pd.DataFrame()

    tempnewdf = wt_type_filtered_data[["Rotor speed","Power"]]
    tempnewdf = tempnewdf.dropna()
    tempnewdf["bin"] = (tempnewdf["Rotor speed"]-(bin_value/2))/bin_value
    tempnewdf["bin"] = tempnewdf["bin"].astype("int64")

    ultratempone = tempnewdf[["bin","Rotor speed"]]
    ultratemptwo = tempnewdf[["bin","Power"]]

    del tempnewdf

    tempbinnedtable1 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable2 = ultratemptwo.groupby(["bin"]).median()

    del ultratempone

    tempnewdf2 = pd.concat([tempbinnedtable1, tempbinnedtable2[["Power"]]], axis = 1)
    tempnewdf2 = tempnewdf2.rename(columns = {"Power":'Avg'})

    #tempbinnedtable3 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable4 = ultratemptwo.groupby(["bin"]).std()

    tempnewdf3 = pd.concat([tempbinnedtable1,tempbinnedtable4], axis = 1)
    tempnewdf3 = tempnewdf3.rename(columns = {"Power":'Stdev'})

    tempnewdf4 = pd.concat([tempnewdf2,tempnewdf3], axis = 1)

    del tempnewdf2
    del tempnewdf3

    #tempbinnedtable5 = ultratempone.groupby(["bin"]).mean()
    tempbinnedtable6 = ultratemptwo.groupby(["bin"]).count()

    del ultratemptwo

    tempnewdf5 = pd.concat([tempbinnedtable1,tempbinnedtable6], axis = 1)
    tempnewdf5 = tempnewdf5.rename(columns = {"Power":'Count'})

    tempnewdf6 = pd.concat([tempnewdf4,tempnewdf5], axis = 1)
    tempnewdf6 = tempnewdf6.loc[tempnewdf6["Count"]>5]

    del tempnewdf5

    tempnewdf4 = tempnewdf6.drop(columns = ["Count"])

    del tempnewdf6

    tempnewdf4 = tempnewdf4.loc[:,~tempnewdf4.columns.duplicated()]
    tempnewdf4.index = tempnewdf4["Rotor speed"]

    if tempnewdf4.empty == False:
        steps = np.around(np.arange(max(5,tempnewdf4["Rotor speed"].min()), tempnewdf4["Rotor speed"].max(),0.1),1).tolist()
        steps_tmp = pd.DataFrame(index = steps,columns = tempnewdf4.columns)

        tempnewdf4 = tempnewdf4._append(steps_tmp)
        tempnewdf4.sort_index(inplace = True)
        tempnewdf4 = tempnewdf4.interpolate(method = "index")
        tempnewdf4 = tempnewdf4.loc[steps]
        tempnewdf4 = tempnewdf4.dropna()
        tempnewdf4 = tempnewdf4.loc[~tempnewdf4.index.duplicated(keep = 'first')]

    binnedtable = tempnewdf4[["Avg","Stdev"]]
    binnedtable.sort_index(inplace = True)

    del tempnewdf4

    wt_types_relationships.at[wt_type,"Power vs RPM"] = binnedtable.to_dict()

    del binnedtable
    gc.collect()

for wt_type in list(set(kpis["Wind turbine type"].tolist())):
    wt_type_filtered_data = pd.DataFrame()
    tmp = kpis.loc[kpis["Wind turbine type"] == wt_type]

    for item in range(len(tmp.index)):
        tmp2 = tmp["Filtered data"].iloc[item]
        tmp2 = pd.DataFrame(tmp2)
        wt_type_filtered_data = pd.concat([wt_type_filtered_data,tmp2])

    del tmp
    del tmp2
    gc.collect()

    try:
        #Wind Speed and Yaw Misaligment
        bin_value = 0.5
        binnedtable = pd.DataFrame()

        tempnewdf = wt_type_filtered_data[["Wind speed","Yaw misalignment"]].dropna()
        tempnewdf["bin"] = (tempnewdf["Wind speed"]-(bin_value/2))/bin_value
        tempnewdf["bin"] = tempnewdf["bin"].astype("int64")

        ultratempone = tempnewdf[["bin","Wind speed"]]
        ultratemptwo = tempnewdf[["bin","Yaw misalignment"]]

        del tempnewdf

        tempbinnedtable1 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable2 = ultratemptwo.groupby(["bin"]).apply(median_angle)

        tempnewdf2 = pd.concat([tempbinnedtable1, tempbinnedtable2[["Yaw misalignment"]]], axis = 1)
        tempnewdf2 = tempnewdf2.rename(columns = {"Yaw misalignment":'Avg'})

        del tempbinnedtable2

        #tempbinnedtable3 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable4 = ultratemptwo.groupby(["bin"]).std()

        tempnewdf3 = pd.concat([tempbinnedtable1,tempbinnedtable4], axis = 1)
        tempnewdf3 = tempnewdf3.rename(columns = {"Yaw misalignment":'Stdev'})
        tempnewdf4 = pd.concat([tempnewdf2,tempnewdf3], axis = 1)

        del tempbinnedtable4
        del tempnewdf2
        del tempnewdf3

        #tempbinnedtable5 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable6 = ultratemptwo.groupby(["bin"]).count()

        del ultratemptwo

        tempnewdf5 = pd.concat([tempbinnedtable1,tempbinnedtable6], axis = 1)
        tempnewdf5 = tempnewdf5.rename(columns = {"Yaw misalignment":'Count'})

        del tempbinnedtable1
        del tempbinnedtable6

        tempnewdf6 = pd.concat([tempnewdf4,tempnewdf5], axis = 1)
        tempnewdf6 = tempnewdf6.loc[tempnewdf6["Count"]>25]

        del tempnewdf5

        tempnewdf4 = tempnewdf6.drop(columns = ["Count"])
        tempnewdf4 = tempnewdf4.loc[:,~tempnewdf4.columns.duplicated()]
        tempnewdf4.index = tempnewdf4["Wind speed"]

        del tempnewdf6

        if tempnewdf4.empty == False:
            steps = np.around(np.arange(0,tempnewdf["Wind speed"].max(),0.1),1).tolist()
            steps_tmp = pd.DataFrame(index = steps,columns = tempnewdf4.columns)
            tempnewdf4 = tempnewdf4._append(steps_tmp)
            tempnewdf4.sort_index(inplace = True)
            tempnewdf4 = tempnewdf4.interpolate(method = "index")
            tempnewdf4 = tempnewdf4.loc[steps]
            tempnewdf4 = tempnewdf4.dropna()
            tempnewdf4 = tempnewdf4.loc[~tempnewdf4.index.duplicated(keep = 'first')]

        binnedtable = tempnewdf4[["Avg","Stdev"]]

        del tempnewdf4

        binnedtable.sort_index(inplace = True)

        wt_type_filtered_data["HiLim"] = wt_type_filtered_data.apply(
            lambda row: binnedtable.at[round(row["Wind speed"],1),"Avg"]+1.96*binnedtable.at[round(row["Wind speed"],1),"Stdev"] 
            if (round(row["Wind speed"],1) in binnedtable.index and pd.isnull(row["Wind speed"]) == False) else np.nan, axis = 1
            )
        wt_type_filtered_data["LoLim"] = wt_type_filtered_data.apply(
            lambda row: binnedtable.at[round(row["Wind speed"],1),"Avg"]-1.96*binnedtable.at[round(row["Wind speed"],1),"Stdev"] 
            if (round(row["Wind speed"],1) in binnedtable.index and pd.isnull(row["Wind speed"]) == False) else np.nan, axis = 1
            )
        wt_type_filtered_data["Filtered yaw misalignment"] = wt_type_filtered_data.apply(
            lambda row: row["Yaw misalignment"] if (row["Yaw misalignment"]>row["LoLim"] and row["Yaw misalignment"]<row["HiLim"] ) else np.nan, axis = 1
            )
        
        del binnedtable
        gc.collect()

        #Wind Speed & Filtered Yaw Misaligment
        bin_value = 0.5
        binnedtable = pd.DataFrame()

        tempnewdf = wt_type_filtered_data[["Wind speed","Filtered yaw misalignment"]].rename(columns = {"Filtered yaw misalignment":'Yaw misalignment'}).dropna()
        tempnewdf["bin"] = (tempnewdf["Wind speed"]-(bin_value/2))/bin_value
        tempnewdf["bin"] = tempnewdf["bin"].astype("int64")

        ultratempone = tempnewdf[["bin","Wind speed"]]
        ultratemptwo = tempnewdf[["bin","Yaw misalignment"]]

        del tempnewdf

        tempbinnedtable1 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable2 = ultratemptwo.groupby(["bin"]).max()

        del ultratempone

        tempnewdf2 = pd.concat([tempbinnedtable1, tempbinnedtable2[["Yaw misalignment"]]], axis = 1)
        tempnewdf2 = tempnewdf2.rename(columns = {"Yaw misalignment":'Max'})

        del tempbinnedtable2

        #tempbinnedtable3 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable4 = ultratemptwo.groupby(["bin"]).min()

        tempnewdf3 = pd.concat([tempbinnedtable1, tempbinnedtable4[["Yaw misalignment"]]], axis = 1)
        tempnewdf3 = tempnewdf3.rename(columns = {"Yaw misalignment":'Min'})

        del tempbinnedtable4

        #tempbinnedtable5 = ultratempone.groupby(["bin"]).mean()
        tempbinnedtable6 = ultratemptwo.groupby(["bin"]).count()

        del ultratemptwo

        tempnewdf5 = pd.concat([tempbinnedtable1,tempbinnedtable6], axis = 1)
        tempnewdf5 = tempnewdf5.rename(columns = {"Yaw misalignment":'Count'})

        del tempbinnedtable6

        tempnewdf6 = pd.concat([tempnewdf2,tempnewdf3,tempnewdf5], axis = 1)
        tempnewdf6 = tempnewdf6.loc[tempnewdf6["Count"]>25]

        del tempnewdf5
        del tempnewdf3

        tempnewdf2 = tempnewdf6.drop(columns = ["Count"])
        tempnewdf2 = tempnewdf2.loc[:,~tempnewdf2.columns.duplicated()]
        tempnewdf2.index = tempnewdf2["Wind speed"]

        del tempnewdf6

        if tempnewdf2.empty == False:
            steps = np.around(np.arange(0,tempnewdf["Wind speed"].max(),0.1),1).tolist()
            steps_tmp = pd.DataFrame(index = steps,columns = tempnewdf2.columns)
            tempnewdf2 = tempnewdf2._append(steps_tmp)
            tempnewdf2.sort_index(inplace = True)
            tempnewdf2 = tempnewdf2.interpolate(method = "index")
            tempnewdf2 = tempnewdf2.loc[steps]
            tempnewdf2 = tempnewdf2.dropna()
            tempnewdf2 = tempnewdf2.loc[~tempnewdf2.index.duplicated(keep = 'first')]

        binnedtable = tempnewdf2[["Min","Max"]]

        del tempnewdf2

        binnedtable.sort_index(inplace = True)

        wt_types_dymb.at[wt_type,"Dynamic yaw misalignment"] = binnedtable.to_dict()
    except:
        wt_types_dymb.at[wt_type,"Dynamic yaw misalignment"] = {}

    del binnedtable
    gc.collect()

#Initialize kpis
kpis["Aerodynamic rotor imbalance data"]=str({})
kpis["Aerodynamic rotor imbalance KPI"]='⚫'
kpis["Aerodynamic rotor imbalance KPI Color"]='black'

kpis["Mass rotor imbalance data"]=str({})
kpis["Mass rotor imbalance KPI"]='⚫'
kpis["Mass rotor imbalance KPI Color"]='black'

kpis["Global rotor imbalance data"]=str({})
kpis["Global rotor imbalance KPI"]='⚫'
kpis["Global rotor imbalance KPI Color"]='black'

kpis["Front bearing temperature data"]=str({})
kpis["Front bearing temperature KPI"]='⚫'
kpis["Front bearing temperature KPI Color"]='black'

kpis["Rear bearing temperature data"]=str({})
kpis["Rear bearing temperature KPI"]='⚫'
kpis["Rear bearing temperature KPI Color"]='black'

kpis["Rotor temperature data"]=str({})
kpis["Rotor temperature KPI"]='⚫'
kpis["Rotor temperature KPI Color"]='black'

kpis["Stator temperature data"]=str({})
kpis["Stator temperature KPI"]='⚫'
kpis["Stator temperature KPI Color"]='black'

kpis["Gearbox HSS bearing temperature data"]=str({})
kpis["Gearbox HSS bearing temperature KPI"]='⚫'
kpis["Gearbox HSS bearing temperature KPI Color"]='black'

kpis["Gearbox IMS/LSS bearing temperature data"]=str({})
kpis["Gearbox IMS/LSS bearing temperature KPI"]='⚫'
kpis["Gearbox IMS/LSS bearing temperature KPI Color"]='black'

kpis["Generator bearing front temperature data"]=str({})
kpis["Generator bearing front temperature KPI"]='⚫'
kpis["Generator bearing front temperature KPI Color"]='black'

kpis["Generator bearing rear temperature data"]=str({})
kpis["Generator bearing rear temperature KPI"]='⚫'
kpis["Generator bearing rear temperature KPI Color"]='black'

kpis["Main bearing temperature data"]=str({})
kpis["Main bearing temperature KPI"]='⚫'
kpis["Main bearing temperature KPI Color"]='black'

kpis["Metal particle count data"]=str({})
kpis["Metal particle count KPI"]='⚫'
kpis["Metal particle count KPI Color"]='black'

kpis["Gearbox oil temperature data"]=str({})
kpis["Gearbox oil temperature KPI"]='⚫'
kpis["Gearbox oil temperature KPI Color"]='black'

#Load confidence intervals from a CSV file
confidence_intervals=pd.read_csv('ML/confidence_intervals.csv',header=0,index_col=[0])
confidence_intervals.index=confidence_intervals["Wind turbine type"]
confidence_intervals=confidence_intervals.drop(columns=["Wind turbine type"])

#Prrocess Normal tower acceleration
#Replace NaN stringsd
confidence_intervals["Normal tower acceleration"]=confidence_intervals["Normal tower acceleration"].apply(
    lambda x: x.replace("nan",'np.nan') if pd.isnull(x)==False else np.nan
    )
confidence_intervals["Normal tower acceleration"]=confidence_intervals["Normal tower acceleration"].apply(
    lambda x: eval(x) if pd.isnull(x)==False else np.nan
    )
confidence_intervals["Lateral tower acceleration"]=confidence_intervals["Lateral tower acceleration"].apply(
    lambda x: x.replace("nan",'np.nan') if pd.isnull(x)==False else np.nan
    )
confidence_intervals["Lateral tower acceleration"]=confidence_intervals["Lateral tower acceleration"].apply(
    lambda x: eval(x) if pd.isnull(x)==False else np.nan
    )
confidence_intervals["Combined tower acceleration"]=confidence_intervals["Combined tower acceleration"].apply(
    lambda x: x.replace("nan",'np.nan') if pd.isnull(x)==False else np.nan
    )
confidence_intervals["Combined tower acceleration"]=confidence_intervals["Combined tower acceleration"].apply(
    lambda x: eval(x) if pd.isnull(x)==False else np.nan
    )

#Loop over each unique Turbine type in the KPIs dataframe
for wt_type in list(set(kpis["Wind turbine type"].tolist())):

    wt_type_filtered_data=pd.DataFrame()
    tmp=kpis.loc[kpis["Wind turbine type"]==wt_type] #FIlter on the current wtg type

    for item in range(len(tmp.index)):
        try:
            tmp2=tmp["Filtered data"].iloc[item]
            tmp2=pd.DataFrame(tmp2)
            tmp2["Timestamp"]=tmp2["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            tmp2.index=tmp2["Timestamp"]
            tmp2=tmp2.drop(columns=["Timestamp"])
            tmp2.index=range(len(tmp2.index))

            wt_type_filtered_data=pd.concat([wt_type_filtered_data,tmp2])

            del tmp2
            gc.collect()

        except:
            pass

    #check if data available
    check_if_yaw_misalignment_data=wt_type_filtered_data["Yaw misalignment"].count()

    if check_if_yaw_misalignment_data>0: #if data yaw misaligment is available

        try:            
            max_RPM=wt_type_filtered_data["Rotor speed"].max()

            m=joblib.load(os.path.join('./ML',wt_type+'_NormalTowerAcceleration.joblib')) #load the ML model for Normal Tower Acceleration from the confidence intervals table

            binnedtable=confidence_intervals.at[wt_type,"Normal tower acceleration"]
            binnedtable=pd.DataFrame(binnedtable)
            binnedtable.index=binnedtable["Rotor speed"] #Set Rotor speed as the index
            
            #Loop over each row in the KPIs dataframe again for the current wtg type
            for item in range(len(kpis.index)):
                if kpis["Wind turbine type"].iloc[item]==wt_type: #Match current type

                    try:
                        tmp3=kpis["Filtered data"].iloc[item]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3.loc[tmp3["Rotor speed"]>0.95*max_RPM] #filter

                        #Prepare column for prediction
                        tmp4=tmp3[["Timestamp","Power","Blade angle","Rotor speed","Yaw misalignment","LTA"]]
                        tmp4["Timestamp"]=tmp4["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp4.index=tmp4["Timestamp"]
                        tmp4=tmp4.drop(columns=["Timestamp"])
                        tmp4=tmp4.dropna()

                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.index=tmp3["Timestamp"]
                        tmp3=tmp3.drop(columns=["Timestamp"])

                        #ML model using tmp4
                        rebuilt=m.predict(tmp4)
                        rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                        #Combien predicted values with actual 
                        tmp5=pd.concat([tmp3[["Rotor speed","NTA"]],rebuilt],axis=1)
#                        tmp5["Timestamp"]=tmp5.index
#                        tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                        tmp5=tmp5.rename(columns={"NTA":'Actual'})
                        tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]

                        del tmp3 
                        del tmp4 
                        gc.collect()
                        
                        tmp5["Rounded rotor speed"]=tmp5["Rotor speed"].apply(
                            lambda x: round(x,1)
                            )

                        tmp5=tmp5.loc[tmp5["Rounded rotor speed"].isin(binnedtable.index)]

                        tmp5["p5"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P5"]
                            )
                        tmp5["p10"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P10"]
                            )
                        tmp5["p25"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P25"]
                            )
                        tmp5["p50"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P50"]
                            )
                        tmp5["p75"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P75"]
                            )
                        tmp5["p90"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P90"]
                            )
                        tmp5["p95"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P95"]
                            )

                        tmp5.index=range(len(tmp5.index))

                        kpis["Aerodynamic rotor imbalance data"].iloc[item]=str(tmp5.to_dict())

                        tmp5=tmp5.dropna()

                        tmp5["p5_exceeded"]=tmp5.apply(lambda row: 1 if row["Residuals"]>row["p5"] else 0,axis=1)
                        tmp5["p10_exceeded"]=tmp5.apply(lambda row: 1 if row["Residuals"]>row["p10"] else 0,axis=1)
                        tmp5["p25_exceeded"]=tmp5.apply(lambda row: 1 if row["Residuals"]>row["p25"] else 0,axis=1)

                        percentage_residuals_above_p25=100*(tmp5["p25_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p10=100*(tmp5["p10_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p5=100*(tmp5["p5_exceeded"].sum())/(tmp5.shape[0])

                        del tmp5
                        gc.collect()

                        if percentage_residuals_above_p25>75:
                            score_to_p25=2
                        elif percentage_residuals_above_p25>50:
                            score_to_p25=1
                        else:
                            score_to_p25=0

                        if percentage_residuals_above_p10>30:
                            score_to_p10=2                            
                        elif percentage_residuals_above_p10>20:
                            score_to_p10=1
                        else:
                            score_to_p10=0

                        if percentage_residuals_above_p5>15:
                            score_to_p5=2
                        elif percentage_residuals_above_p5>10:
                            score_to_p5=1
                        else:
                            score_to_p5=0

                        total_score=score_to_p25+score_to_p10+score_to_p5

                        if tmp5.shape[0]>50:

                            if total_score>3:
                                kpis["Aerodynamic rotor imbalance KPI"].iloc[item]='🔴'
                                kpis["Aerodynamic rotor imbalance KPI Color"].iloc[item]='red'

                            elif total_score>1:
                                kpis["Aerodynamic rotor imbalance KPI"].iloc[item]='🟠'
                                kpis["Aerodynamic rotor imbalance KPI Color"].iloc[item]='orange'

                            else:
                                kpis["Aerodynamic rotor imbalance KPI"].iloc[item]='🟢'
                                kpis["Aerodynamic rotor imbalance KPI Color"].iloc[item]='green'

                    except:
                        pass
                            
        except:
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Aerodynamic rotor imbalance KPI"]='⚪'
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Aerodynamic rotor imbalance KPI Color"]='white'

        #Lateral Tower Acceleration
        try:
            m=joblib.load(os.path.join('./ML',wt_type+'_LateralTowerAcceleration.joblib'))

            binnedtable=confidence_intervals.at[wt_type,"Lateral tower acceleration"]
            binnedtable=pd.DataFrame(binnedtable)
            binnedtable.index=binnedtable["Rotor speed"]
            
            #Iterate over the KPIs DataFrame
            for item in range(len(kpis.index)):

                if kpis["Wind turbine type"].iloc[item]==wt_type:

                    try:
                        tmp3=kpis["Filtered data"].iloc[item]
                        tmp3=pd.DataFrame(tmp3)

                        tmp4=tmp3[["Timestamp","Power","Blade angle","Rotor speed","Yaw misalignment","NTA"]]

                        tmp4["Timestamp"]=tmp4["Timestamp"].apply(
                            lambda x: dt.strptime(x,'%d/%m/%Y %H:%M')
                            )

                        tmp4.index=tmp4["Timestamp"]
                        tmp4=tmp4.drop(columns=["Timestamp"])
                        tmp4=tmp4.dropna()

                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(
                            lambda x: dt.strptime(x,'%d/%m/%Y %H:%M')
                            )
                        tmp3.index=tmp3["Timestamp"]
                        tmp3=tmp3.drop(columns=["Timestamp"])

                        rebuilt=m.predict(tmp4)
                        rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                        tmp5=pd.concat([tmp3[["Rotor speed","LTA"]],rebuilt],axis=1)
#                        tmp5["Timestamp"]=tmp5.index
#                        tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                        tmp5=tmp5.rename(columns={"LTA":'Actual'})
                        tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                        
                        tmp5["Rounded rotor speed"]=tmp5["Rotor speed"].apply(
                            lambda x: round(x,1)
                            )

                        tmp5=tmp5.loc[tmp5["Rounded rotor speed"].isin(binnedtable.index)]

                        tmp5["p5"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P5"]
                            )
                        tmp5["p10"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P10"]
                            )
                        tmp5["p25"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P25"]
                            )
                        tmp5["p50"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P50"]
                            )
                        tmp5["p75"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P75"]
                            )
                        tmp5["p90"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P90"]
                            )
                        tmp5["p95"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P95"]
                            )

                        tmp5.index=range(len(tmp5.index))

                        kpis["Mass rotor imbalance data"].iloc[item]=str(tmp5.to_dict())

                        tmp5=tmp5.dropna()
                        tmp5["p5_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p5"] else 0,axis=1
                            )
                        tmp5["p10_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p10"] else 0,axis=1
                            )
                        tmp5["p25_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p25"] else 0,axis=1
                            )

                        percentage_residuals_above_p25=100*(tmp5["p25_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p10=100*(tmp5["p10_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p5=100*(tmp5["p5_exceeded"].sum())/(tmp5.shape[0])

                        if percentage_residuals_above_p25>75:
                            score_to_p25=2
                        elif percentage_residuals_above_p25>50:
                            score_to_p25=1
                        else:
                            score_to_p25=0

                        if percentage_residuals_above_p10>30:
                            score_to_p10=2
                        elif percentage_residuals_above_p10>20:
                            score_to_p10=1
                        else:
                            score_to_p10=0

                        if percentage_residuals_above_p5>15:
                            score_to_p5=2
                        elif percentage_residuals_above_p5>10:
                            score_to_p5=1
                        else:
                            score_to_p5=0

                        total_score=score_to_p25+score_to_p10+score_to_p5

                        if tmp5.shape[0]>50:
                            if total_score>3:
                                kpis["Mass rotor imbalance KPI"].iloc[item]='🔴'
                                kpis["Mass rotor imbalance KPI Color"].iloc[item]='red'
                            elif total_score>1:
                                kpis["Mass rotor imbalance KPI"].iloc[item]='🟠'
                                kpis["Mass rotor imbalance KPI Color"].iloc[item]='orange'
                            else:
                                kpis["Mass rotor imbalance KPI"].iloc[item]='🟢'
                                kpis["Mass rotor imbalance KPI Color"].iloc[item]='green'
                    except:
                        pass
                        
    
        except:
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Mass rotor imbalance KPI"]='⚪'
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Mass rotor imbalance KPI Color"]='white'
    
        
        #Combined Tower Acceleration    
        try:

            m=joblib.load(os.path.join('./ML',wt_type+'_CombinedTowerAcceleration.joblib'))

            binnedtable=confidence_intervals.at[wt_type,"Combined tower acceleration"]
            binnedtable=pd.DataFrame(binnedtable)
            binnedtable.index=binnedtable["Rotor speed"]
            
            for item in range(len(kpis.index)):

                if kpis["Wind turbine type"].iloc[item]==wt_type:
                    try:
                        tmp3=kpis["Filtered data"].iloc[item]
                        tmp3=pd.DataFrame(tmp3)

                        tmp4=tmp3[["Timestamp","Power","Blade angle","Rotor speed","Yaw misalignment"]]
                        tmp4["Timestamp"]=tmp4["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp4.index=tmp4["Timestamp"]
                        tmp4=tmp4.drop(columns=["Timestamp"])
                        tmp4=tmp4.dropna()

                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.index=tmp3["Timestamp"]
                        tmp3=tmp3.drop(columns=["Timestamp"])

                        rebuilt=m.predict(tmp4)
                        rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                        tmp5=pd.concat([tmp3[["Rotor speed","CTA"]],rebuilt],axis=1)
#                        tmp5["Timestamp"]=tmp5.index
#                        tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                        tmp5=tmp5.rename(columns={"CTA":'Actual'})
                        tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]

                        del tmp3
                        del tmp4 
                        
                        tmp5["Rounded rotor speed"]=tmp5["Rotor speed"].apply(
                            lambda x: round(x,1)
                            )  

                        tmp5=tmp5.loc[tmp5["Rounded rotor speed"].isin(binnedtable.index)]

                        tmp5["p5"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P5"]
                            )
                        tmp5["p10"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P10"]
                            )
                        tmp5["p25"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P25"]
                            )
                        tmp5["p50"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P50"]
                            )
                        tmp5["p75"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P75"]
                            )
                        tmp5["p90"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P90"]
                            )
                        tmp5["p95"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P95"]
                            )
                        tmp5.index=range(len(tmp5.index))

                        kpis["Global rotor imbalance data"].iloc[item]=str(tmp5.to_dict())

                        tmp5=tmp5.dropna()

                        tmp5["p5_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p5"] else 0,axis=1
                            )
                        tmp5["p10_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p10"] else 0,axis=1
                            )
                        tmp5["p25_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p25"] else 0,axis=1
                            )

                        percentage_residuals_above_p25=100*(tmp5["p25_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p10=100*(tmp5["p10_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p5=100*(tmp5["p5_exceeded"].sum())/(tmp5.shape[0])

                        if percentage_residuals_above_p25>75:
                            score_to_p25=2
                        elif percentage_residuals_above_p25>50:
                            score_to_p25=1
                        else:
                            score_to_p25=0

                        if percentage_residuals_above_p10>30:
                            score_to_p10=2
                        elif percentage_residuals_above_p10>20:
                            score_to_p10=1
                        else:
                            score_to_p10=0

                        if percentage_residuals_above_p5>15:
                            score_to_p5=2
                        elif percentage_residuals_above_p5>10:
                            score_to_p5=1
                        else:
                            score_to_p5=0

                        total_score=score_to_p25+score_to_p10+score_to_p5

                        if tmp5.shape[0]>50:
                            if total_score>3:
                                kpis["Global rotor imbalance KPI"].iloc[item]='🔴'
                                kpis["Global rotor imbalance KPI Color"].iloc[item]='red'
                            elif total_score>1:
                                kpis["Global rotor imbalance KPI"].iloc[item]='🟠'
                                kpis["Global rotor imbalance KPI Color"].iloc[item]='orange'
                            else:
                                kpis["Global rotor imbalance KPI"].iloc[item]='🟢'
                                kpis["Global rotor imbalance KPI Color"].iloc[item]='green'

                    except:
                        pass
                        
    
        except:
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Global rotor imbalance KPI"]='⚪'
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Global rotor imbalance KPI Color"]='white'

    else:
    
        #Normal Tower Acceleration
        try:

            max_RPM=wt_type_filtered_data["Rotor speed"].max()

            m=joblib.load(os.path.join('./ML',wt_type+'_NormalTowerAcceleration.joblib'))

            binnedtable=confidence_intervals.at[wt_type,"Normal tower acceleration"]
            binnedtable=pd.DataFrame(binnedtable)
            binnedtable.index=binnedtable["Rotor speed"]
            
            for item in range(len(kpis.index)):
                if kpis["Wind turbine type"].iloc[item]==wt_type:

                    try:
                        tmp3=kpis["Filtered data"].iloc[item]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3.loc[tmp3["Rotor speed"]>0.95*max_RPM]

                        tmp4=tmp3[["Timestamp","Power","Blade angle","Rotor speed","LTA"]]
                        tmp4["Timestamp"]=tmp4["Timestamp"].apply(
                            lambda x: dt.strptime(x,'%d/%m/%Y %H:%M')
                            )
                        tmp4.index=tmp4["Timestamp"]
                        tmp4=tmp4.drop(columns=["Timestamp"])
                        tmp4=tmp4.dropna()

                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(
                            lambda x: dt.strptime(x,'%d/%m/%Y %H:%M')
                            )
                        tmp3.index=tmp3["Timestamp"]
                        tmp3=tmp3.drop(columns=["Timestamp"])

                        rebuilt=m.predict(tmp4)
                        rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                        tmp5=pd.concat([tmp3[["Rotor speed","NTA"]],rebuilt],axis=1)
#                        tmp5["Timestamp"]=tmp5.index
#                        tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                        tmp5=tmp5.rename(columns={"NTA":'Actual'})
                        tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]

                        del tmp3
                        del tmp4
                        
                        tmp5["Rounded rotor speed"]=tmp5["Rotor speed"].apply(
                            lambda x: round(x,1)
                            )  
                        tmp5=tmp5.loc[tmp5["Rounded rotor speed"].isin(binnedtable.index)]

                        tmp5["p5"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P5"]
                            )
                        tmp5["p10"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P10"]
                            )
                        tmp5["p25"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P25"]
                            )
                        tmp5["p50"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P50"]
                            )
                        tmp5["p75"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P75"]
                            )
                        tmp5["p90"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P90"]
                            )
                        tmp5["p95"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P95"]
                            )

                        tmp5.index=range(len(tmp5.index))

                        kpis["Aerodynamic rotor imbalance data"].iloc[item]=str(tmp5.to_dict())

                        tmp5=tmp5.dropna()

                        tmp5["p5_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p5"] else 0,axis=1
                            )
                        tmp5["p10_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p10"] else 0,axis=1
                            )
                        tmp5["p25_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p25"] else 0,axis=1
                            )

                        percentage_residuals_above_p25=100*(tmp5["p25_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p10=100*(tmp5["p10_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p5=100*(tmp5["p5_exceeded"].sum())/(tmp5.shape[0])

                        if percentage_residuals_above_p25>75:
                            score_to_p25=2
                        elif percentage_residuals_above_p25>50:
                            score_to_p25=1
                        else:
                            score_to_p25=0

                        if percentage_residuals_above_p10>30:
                            score_to_p10=2
                        elif percentage_residuals_above_p10>20:
                            score_to_p10=1
                        else:
                            score_to_p10=0

                        if percentage_residuals_above_p5>15:
                            score_to_p5=2
                        elif percentage_residuals_above_p5>10:
                            score_to_p5=1
                        else:
                            score_to_p5=0

                        total_score=score_to_p25+score_to_p10+score_to_p5

                        if tmp5.shape[0]>50:

                            if total_score>3:
                                kpis["Aerodynamic rotor imbalance KPI"].iloc[item]='🔴'
                                kpis["Aerodynamic rotor imbalance KPI Color"].iloc[item]='red'
                            elif total_score>1:
                                kpis["Aerodynamic rotor imbalance KPI"].iloc[item]='🟠'
                                kpis["Aerodynamic rotor imbalance KPI Color"].iloc[item]='orange'
                            else:
                                kpis["Aerodynamic rotor imbalance KPI"].iloc[item]='🟢'
                                kpis["Aerodynamic rotor imbalance KPI Color"].iloc[item]='green'
                    except:
                        pass
                        
    
        except:
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Aerodynamic rotor imbalance KPI"]='⚪'
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Aerodynamic rotor imbalance KPI Color"]='white'
    

        #Latera Tower Acceleration
        try:

            m=joblib.load(os.path.join('./ML',wt_type+'_LateralTowerAcceleration.joblib'))

            binnedtable=confidence_intervals.at[wt_type,"Lateral tower acceleration"]
            binnedtable=pd.DataFrame(binnedtable)
            binnedtable.index=binnedtable["Rotor speed"]
            
            for item in range(len(kpis.index)):
                if kpis["Wind turbine type"].iloc[item]==wt_type:

                    try:
                        tmp3=kpis["Filtered data"].iloc[item]
                        tmp3=pd.DataFrame(tmp3)

                        tmp4=tmp3[["Timestamp","Power","Blade angle","Rotor speed","NTA"]]
                        tmp4["Timestamp"]=tmp4["Timestamp"].apply(
                            lambda x: dt.strptime(x,'%d/%m/%Y %H:%M')
                            )
                        tmp4.index=tmp4["Timestamp"]
                        tmp4=tmp4.drop(columns=["Timestamp"])
                        tmp4=tmp4.dropna()

                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(
                            lambda x: dt.strptime(x,'%d/%m/%Y %H:%M')
                            )
                        tmp3.index=tmp3["Timestamp"]
                        tmp3=tmp3.drop(columns=["Timestamp"])

                        rebuilt=m.predict(tmp4)
                        rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                        tmp5=pd.concat([tmp3[["Rotor speed","LTA"]],rebuilt],axis=1)
#                        tmp5["Timestamp"]=tmp5.index
#                        tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                        tmp5=tmp5.rename(columns={"LTA":'Actual'})
                        tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                        
                        tmp5["Rounded rotor speed"]=tmp5["Rotor speed"].apply(lambda x: round(x,1))  
                        tmp5=tmp5.loc[tmp5["Rounded rotor speed"].isin(binnedtable.index)]

                        tmp5["p5"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P5"]
                            )
                        tmp5["p10"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P10"]
                            )
                        tmp5["p25"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P25"]
                            )
                        tmp5["p50"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P50"]
                            )
                        tmp5["p75"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P75"]
                            )
                        tmp5["p90"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P90"]
                            )
                        tmp5["p95"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P95"]
                            )

                        tmp5.index=range(len(tmp5.index))

                        kpis["Mass rotor imbalance data"].iloc[item]=str(tmp5.to_dict())

                        tmp5=tmp5.dropna()
                        tmp5["p5_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p5"] else 0,axis=1
                            )
                        tmp5["p10_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p10"] else 0,axis=1
                            )
                        tmp5["p25_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p25"] else 0,axis=1
                            )

                        percentage_residuals_above_p25=100*(tmp5["p25_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p10=100*(tmp5["p10_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p5=100*(tmp5["p5_exceeded"].sum())/(tmp5.shape[0])

                        if percentage_residuals_above_p25>75:
                            score_to_p25=2
                        elif percentage_residuals_above_p25>50:
                            score_to_p25=1
                        else:
                            score_to_p25=0

                        if percentage_residuals_above_p10>30:
                            score_to_p10=2
                        elif percentage_residuals_above_p10>20:
                            score_to_p10=1
                        else:
                            score_to_p10=0

                        if percentage_residuals_above_p5>15:
                            score_to_p5=2
                        elif percentage_residuals_above_p5>10:
                            score_to_p5=1
                        else:
                            score_to_p5=0

                        total_score=score_to_p25+score_to_p10+score_to_p5

                        if tmp5.shape[0]>50:
                            if total_score>3:
                                kpis["Mass rotor imbalance KPI"].iloc[item]='🔴'
                                kpis["Mass rotor imbalance KPI Color"].iloc[item]='red'
                            elif total_score>1:
                                kpis["Mass rotor imbalance KPI"].iloc[item]='🟠'
                                kpis["Mass rotor imbalance KPI Color"].iloc[item]='orange'
                            else:
                                kpis["Mass rotor imbalance KPI"].iloc[item]='🟢'
                                kpis["Mass rotor imbalance KPI Color"].iloc[item]='green'

                    except:
                        pass
                        
        except:
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Mass rotor imbalance KPI"]='⚪'
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Mass rotor imbalance KPI Color"]='white'
    
        #Combined Tower Acceleration
        try:

            m=joblib.load(os.path.join('./ML',wt_type+'_CombinedTowerAcceleration.joblib'))

            binnedtable=confidence_intervals.at[wt_type,"Combined tower acceleration"]
            binnedtable=pd.DataFrame(binnedtable)
            binnedtable.index=binnedtable["Rotor speed"]
            
            for item in range(len(kpis.index)):
                if kpis["Wind turbine type"].iloc[item]==wt_type:

                    try:
                        tmp3=kpis["Filtered data"].iloc[item]
                        tmp3=pd.DataFrame(tmp3)

                        tmp4=tmp3[["Timestamp","Power","Blade angle","Rotor speed"]]
                        tmp4["Timestamp"]=tmp4["Timestamp"].apply(
                            lambda x: dt.strptime(x,'%d/%m/%Y %H:%M')
                            )
                        tmp4.index=tmp4["Timestamp"]
                        tmp4=tmp4.drop(columns=["Timestamp"])
                        tmp4=tmp4.dropna()

                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(
                            lambda x: dt.strptime(x,'%d/%m/%Y %H:%M')
                            )
                        tmp3.index=tmp3["Timestamp"]
                        tmp3=tmp3.drop(columns=["Timestamp"])

                        rebuilt=m.predict(tmp4)
                        rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                        tmp5=pd.concat([tmp3[["Rotor speed","CTA"]],rebuilt],axis=1)
#                        tmp5["Timestamp"]=tmp5.index
#                        tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                        tmp5=tmp5.rename(columns={"CTA":'Actual'})
                        tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                        
                        tmp5["Rounded rotor speed"]=tmp5["Rotor speed"].apply(
                            lambda x: round(x,1)
                            )  
                        tmp5=tmp5.loc[tmp5["Rounded rotor speed"].isin(binnedtable.index)]
                        tmp5["p5"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P5"]
                            )
                        tmp5["p10"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P10"]
                            )
                        tmp5["p25"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P25"]
                            )
                        tmp5["p50"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P50"]
                            )
                        tmp5["p75"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P75"]
                            )
                        tmp5["p90"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P90"]
                            )
                        tmp5["p95"]=tmp5["Rounded rotor speed"].apply(
                            lambda x: binnedtable.at[x,"P95"]
                            )

                        tmp5.index=range(len(tmp5.index))

                        kpis["Global rotor imbalance data"].iloc[item]=str(tmp5.to_dict())

                        tmp5=tmp5.dropna()

                        tmp5["p5_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p5"] else 0,axis=1
                            )
                        tmp5["p10_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p10"] else 0,axis=1
                            )
                        tmp5["p25_exceeded"]=tmp5.apply(
                            lambda row: 1 if row["Residuals"]>row["p25"] else 0,axis=1
                            )

                        percentage_residuals_above_p25=100*(tmp5["p25_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p10=100*(tmp5["p10_exceeded"].sum())/(tmp5.shape[0])
                        percentage_residuals_above_p5=100*(tmp5["p5_exceeded"].sum())/(tmp5.shape[0])

                        if percentage_residuals_above_p25>75:
                            score_to_p25=2
                        elif percentage_residuals_above_p25>50:
                            score_to_p25=1
                        else:
                            score_to_p25=0

                        if percentage_residuals_above_p10>30:
                            score_to_p10=2
                        elif percentage_residuals_above_p10>20:
                            score_to_p10=1
                        else:
                            score_to_p10=0

                        if percentage_residuals_above_p5>15:
                            score_to_p5=2
                        elif percentage_residuals_above_p5>10:
                            score_to_p5=1
                        else:
                            score_to_p5=0

                        total_score=score_to_p25+score_to_p10+score_to_p5

                        if tmp5.shape[0]>50:
                            if total_score>3:
                                kpis["Global rotor imbalance KPI"].iloc[item]='🔴'
                                kpis["Global rotor imbalance KPI Color"].iloc[item]='red'

                            elif total_score>1:
                                kpis["Global rotor imbalance KPI"].iloc[item]='🟠'
                                kpis["Global rotor imbalance KPI Color"].iloc[item]='orange'

                            else:
                                kpis["Global rotor imbalance KPI"].iloc[item]='🟢'
                                kpis["Global rotor imbalance KPI Color"].iloc[item]='green'
                    except:
                        pass
                        
    
        except:
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Global rotor imbalance KPI"]='⚪'
            kpis.loc[kpis["Wind turbine type"]==wt_type,"Global rotor imbalance KPI Color"]='white'
    
for wt_type in list(set(kpis["Wind turbine type"].tolist())):

    try:

        m=joblib.load(os.path.join('./ML',wt_type+'_FrontBearingTemperature.joblib'))
        rmse=confidence_intervals.at[wt_type,"Front bearing temperature"]
        
        for item in range(len(kpis.index)):
            if kpis["Wind turbine type"].iloc[item]==wt_type:
                try:
                    
                    current_month=kpis["MonthAsDate"].iloc[item]
                    current_wt=kpis["Wind turbine"].iloc[item]

                    timestamps=pd.period_range(current_month,current_month+relativedelta(months=1),freq='10min')[:-1].to_timestamp().tolist()
                    
                    tmp=pd.DataFrame(index=timestamps)
                    
                    tmp3=kpis["Filtered data"].iloc[item]
                    tmp3=pd.DataFrame(tmp3)
                    tmp3=tmp3[["Timestamp","Power","Ambient temperature","Front bearing temperature"]]
                    tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                    tmp3.set_index("Timestamp",inplace=True)
                    
                    tmp=tmp.merge(tmp3,how='left',left_index=True,right_index=True)
                    
                    try:
                        timestamps=pd.period_range(current_month+relativedelta(months=-1),current_month,freq='10min')[:-1].to_timestamp().tolist()
                        
                        tmp2=pd.DataFrame(index=timestamps)
                        
                        tmp3=kpis.loc[(kpis["Wind turbine"]==current_wt) & (kpis["MonthAsDate"]==(current_month+relativedelta(months=-1)))]["Filtered data"].iloc[0]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3[["Timestamp","Power","Ambient temperature","Front bearing temperature"]]
                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.set_index("Timestamp",inplace=True)
                        
                        tmp2=tmp2.merge(tmp3,how='left',left_index=True,right_index=True)
                        
                        tmp=pd.concat([tmp2,tmp])

                    except:
                        pass

                    tmp["Timestamp"]=tmp.index
                    tmp["PastPower"]=tmp.rolling(144)["Power"].mean()
                    tmp["PastAmbientTemp"]=tmp.rolling(144)["Ambient temperature"].mean()
                    tmp["PastPower"]=tmp["PastPower"].ffill()
                    tmp["PastAmbientTemp"]=tmp["PastAmbientTemp"].ffill()

                    tmp=tmp.loc[tmp["Timestamp"]>=current_month]

                    tmp4=tmp[["Power","Ambient temperature","PastPower","PastAmbientTemp"]]
                    tmp4=tmp4.dropna()

                    rebuilt=m.predict(tmp4)
                    rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                    tmp5=pd.concat([tmp[["Front bearing temperature"]],rebuilt],axis=1)
                    tmp5["Timestamp"]=tmp5.index
                    tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                    tmp5=tmp5.rename(columns={"Front bearing temperature":'Actual'})
                    tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                    tmp5["p25"]=0.675*rmse
                    tmp5["p10"]=1.282*rmse
                    tmp5["p75"]=-0.675*rmse
                    tmp5["p90"]=-1.282*rmse
                    tmp5.index=range(len(tmp5.index))

                    kpis["Front bearing temperature data"].iloc[item]=str(tmp5.to_dict())

                    tmp5=tmp5.dropna()

                    percentage_residuals_above_p25=100*len(tmp5[tmp5['Residuals']>0.675*rmse])/tmp5.shape[0]
                    percentage_residuals_above_p10=100*len(tmp5[tmp5['Residuals']>1.282*rmse])/tmp5.shape[0]

                    if tmp5.shape[0]>10 and (percentage_residuals_above_p25>50 or percentage_residuals_above_p10>20):
                        kpis["Front bearing temperature KPI"].iloc[item]='🔴'
                        kpis["Front bearing temperature KPI Color"].iloc[item]='red'

                    elif tmp5.shape[0]>10 and (percentage_residuals_above_p25>37.5 or percentage_residuals_above_p10>15):
                        kpis["Front bearing temperature KPI"].iloc[item]='🟠'
                        kpis["Front bearing temperature KPI Color"].iloc[item]='orange'

                    elif tmp5.shape[0]>10 and percentage_residuals_above_p25<=37.5 and percentage_residuals_above_p10<=15:
                        kpis["Front bearing temperature KPI"].iloc[item]='🟢'
                        kpis["Front bearing temperature KPI Color"].iloc[item]='green'

                except:
                    pass
                    
    except:
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Front bearing temperature KPI"]='⚪'
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Front bearing temperature KPI Color"]='white'

    try:
        m=joblib.load(os.path.join('./ML',wt_type+'_RearBearingTemperature.joblib'))
        rmse=confidence_intervals.at[wt_type,"Rear bearing temperature"]
        
        for item in range(len(kpis.index)):
            if kpis["Wind turbine type"].iloc[item]==wt_type:

                try:
                    current_month=kpis["MonthAsDate"].iloc[item]
                    current_wt=kpis["Wind turbine"].iloc[item]

                    timestamps=pd.period_range(current_month,current_month+relativedelta(months=1),freq='10min')[:-1].to_timestamp().tolist()

                    tmp=pd.DataFrame(index=timestamps)

                    tmp3=kpis["Filtered data"].iloc[item]
                    tmp3=pd.DataFrame(tmp3)
                    tmp3=tmp3[["Timestamp","Power","Ambient temperature","Rear bearing temperature"]]
                    tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                    tmp3.set_index("Timestamp",inplace=True)

                    tmp=tmp.merge(tmp3,how='left',left_index=True,right_index=True)

                    try:
                        timestamps=pd.period_range(current_month+relativedelta(months=-1),current_month,freq='10min')[:-1].to_timestamp().tolist()

                        tmp2=pd.DataFrame(index=timestamps)

                        tmp3=kpis.loc[(kpis["Wind turbine"]==current_wt) & (kpis["MonthAsDate"]==(current_month+relativedelta(months=-1)))]["Filtered data"].iloc[0]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3[["Timestamp","Power","Ambient temperature","Rear bearing temperature"]]
                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.set_index("Timestamp",inplace=True)

                        tmp2=tmp2.merge(tmp3,how='left',left_index=True,right_index=True)

                        tmp=pd.concat([tmp2,tmp])

                    except:
                        pass

                    tmp["Timestamp"]=tmp.index
                    tmp["PastPower"]=tmp.rolling(144)["Power"].mean()
                    tmp["PastAmbientTemp"]=tmp.rolling(144)["Ambient temperature"].mean()
                    tmp["PastPower"]=tmp["PastPower"].ffill()
                    tmp["PastAmbientTemp"]=tmp["PastAmbientTemp"].ffill()

                    tmp=tmp.loc[tmp["Timestamp"]>=current_month]

                    tmp4=tmp[["Power","Ambient temperature","PastPower","PastAmbientTemp"]]
                    tmp4=tmp4.dropna()

                    rebuilt=m.predict(tmp4)
                    rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                    tmp5=pd.concat([tmp[["Rear bearing temperature"]],rebuilt],axis=1)

                    tmp5["Timestamp"]=tmp5.index
                    tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                    tmp5=tmp5.rename(columns={"Rear bearing temperature":'Actual'})
                    tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                    tmp5["p25"]=0.675*rmse
                    tmp5["p10"]=1.282*rmse
                    tmp5["p75"]=-0.675*rmse
                    tmp5["p90"]=-1.282*rmse
                    tmp5.index=range(len(tmp5.index))

                    kpis["Rear bearing temperature data"].iloc[item]=str(tmp5.to_dict())

                    tmp5=tmp5.dropna()

                    percentage_residuals_above_p25=100*len(tmp5[tmp5['Residuals']>0.675*rmse])/tmp5.shape[0]
                    percentage_residuals_above_p10=100*len(tmp5[tmp5['Residuals']>1.282*rmse])/tmp5.shape[0]

                    if tmp5.shape[0]>10 and (percentage_residuals_above_p25>50 or percentage_residuals_above_p10>20):
                        kpis["Rear bearing temperature KPI"].iloc[item]='🔴'
                        kpis["Rear bearing temperature KPI Color"].iloc[item]='red'

                    elif tmp5.shape[0]>10 and (percentage_residuals_above_p25>37.5 or percentage_residuals_above_p10>15):
                        kpis["Rear bearing temperature KPI"].iloc[item]='🟠'
                        kpis["Rear bearing temperature KPI Color"].iloc[item]='orange'

                    elif tmp5.shape[0]>10 and percentage_residuals_above_p25<=37.5 and percentage_residuals_above_p10<=15:
                        kpis["Rear bearing temperature KPI"].iloc[item]='🟢'
                        kpis["Rear bearing temperature KPI Color"].iloc[item]='green'

                except:
                    pass
                    
    except:
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Rear bearing temperature KPI"]='⚪'
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Rear bearing temperature KPI Color"]='white'

    try:
        m=joblib.load(os.path.join('./ML',wt_type+'_RotorTemperature.joblib'))
        rmse=confidence_intervals.at[wt_type,"Rotor temperature"]
        
        for item in range(len(kpis.index)):
            if kpis["Wind turbine type"].iloc[item]==wt_type:

                try:
                    current_month=kpis["MonthAsDate"].iloc[item]
                    current_wt=kpis["Wind turbine"].iloc[item]

                    timestamps=pd.period_range(current_month,current_month+relativedelta(months=1),freq='10min')[:-1].to_timestamp().tolist()

                    tmp=pd.DataFrame(index=timestamps)

                    tmp3=kpis["Filtered data"].iloc[item]
                    tmp3=pd.DataFrame(tmp3)
                    tmp3=tmp3[["Timestamp","Power","Ambient temperature","Rotor temperature"]]
                    tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                    tmp3.set_index("Timestamp",inplace=True)

                    tmp=tmp.merge(tmp3,how='left',left_index=True,right_index=True)

                    try:
                        timestamps=pd.period_range(current_month+relativedelta(months=-1),current_month,freq='10min')[:-1].to_timestamp().tolist()

                        tmp2=pd.DataFrame(index=timestamps)

                        tmp3=kpis.loc[(kpis["Wind turbine"]==current_wt) & (kpis["MonthAsDate"]==(current_month+relativedelta(months=-1)))]["Filtered data"].iloc[0]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3[["Timestamp","Power","Ambient temperature","Rotor temperature"]]
                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.set_index("Timestamp",inplace=True)

                        tmp2=tmp2.merge(tmp3,how='left',left_index=True,right_index=True)

                        tmp=pd.concat([tmp2,tmp])

                    except:
                        pass

                    tmp["Timestamp"]=tmp.index
                    tmp["PastPower"]=tmp.rolling(144)["Power"].mean()
                    tmp["PastAmbientTemp"]=tmp.rolling(144)["Ambient temperature"].mean()
                    tmp["PastPower"]=tmp["PastPower"].ffill()
                    tmp["PastAmbientTemp"]=tmp["PastAmbientTemp"].ffill()
                    tmp=tmp.loc[tmp["Timestamp"]>=current_month]

                    tmp4=tmp[["Power","Ambient temperature","PastPower","PastAmbientTemp"]]
                    tmp4=tmp4.dropna()

                    rebuilt=m.predict(tmp4)
                    rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                    tmp5=pd.concat([tmp[["Rotor temperature"]],rebuilt],axis=1)

                    tmp5["Timestamp"]=tmp5.index
                    tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                    tmp5=tmp5.rename(columns={"Rotor temperature":'Actual'})
                    tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                    tmp5["p25"]=0.675*rmse
                    tmp5["p10"]=1.282*rmse
                    tmp5["p75"]=-0.675*rmse
                    tmp5["p90"]=-1.282*rmse
                    tmp5.index=range(len(tmp5.index))

                    kpis["Rotor temperature data"].iloc[item]=str(tmp5.to_dict())

                    tmp5=tmp5.dropna()

                    percentage_residuals_above_p25=100*len(tmp5[tmp5['Residuals']>0.675*rmse])/tmp5.shape[0]
                    percentage_residuals_above_p10=100*len(tmp5[tmp5['Residuals']>1.282*rmse])/tmp5.shape[0]

                    if tmp5.shape[0]>10 and (percentage_residuals_above_p25>50 or percentage_residuals_above_p10>20):
                        kpis["Rotor temperature KPI"].iloc[item]='🔴'
                        kpis["Rotor temperature KPI Color"].iloc[item]='red'

                    elif tmp5.shape[0]>10 and (percentage_residuals_above_p25>37.5 or percentage_residuals_above_p10>15):
                        kpis["Rotor temperature KPI"].iloc[item]='🟠'
                        kpis["Rotor temperature KPI Color"].iloc[item]='orange'

                    elif tmp5.shape[0]>10 and percentage_residuals_above_p25<=37.5 and percentage_residuals_above_p10<=15:
                        kpis["Rotor temperature KPI"].iloc[item]='🟢'
                        kpis["Rotor temperature KPI Color"].iloc[item]='green'

                except:
                    pass
                    

    except:
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Rotor temperature KPI"]='⚪'
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Rotor temperature KPI Color"]='white'



    try:
        m=joblib.load(os.path.join('./ML',wt_type+'_StatorTemperature.joblib'))
        rmse=confidence_intervals.at[wt_type,"Stator temperature"]
        
        for item in range(len(kpis.index)):
            if kpis["Wind turbine type"].iloc[item]==wt_type:
                try:

                    current_month=kpis["MonthAsDate"].iloc[item]
                    current_wt=kpis["Wind turbine"].iloc[item]

                    timestamps=pd.period_range(current_month,current_month+relativedelta(months=1),freq='10min')[:-1].to_timestamp().tolist()

                    tmp=pd.DataFrame(index=timestamps)

                    tmp3=kpis["Filtered data"].iloc[item]
                    tmp3=pd.DataFrame(tmp3)
                    tmp3=tmp3[["Timestamp","Power","Ambient temperature","Stator temperature"]]
                    tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                    tmp3.set_index("Timestamp",inplace=True)

                    tmp=tmp.merge(tmp3,how='left',left_index=True,right_index=True)

                    try:
                        timestamps=pd.period_range(current_month+relativedelta(months=-1),current_month,freq='10min')[:-1].to_timestamp().tolist()

                        tmp2=pd.DataFrame(index=timestamps)

                        tmp3=kpis.loc[(kpis["Wind turbine"]==current_wt) & (kpis["MonthAsDate"]==(current_month+relativedelta(months=-1)))]["Filtered data"].iloc[0]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3[["Timestamp","Power","Ambient temperature","Stator temperature"]]
                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.set_index("Timestamp",inplace=True)

                        tmp2=tmp2.merge(tmp3,how='left',left_index=True,right_index=True)

                        tmp=pd.concat([tmp2,tmp])

                    except:
                        pass

                    tmp["Timestamp"]=tmp.index
                    tmp["PastPower"]=tmp.rolling(144)["Power"].mean()
                    tmp["PastAmbientTemp"]=tmp.rolling(144)["Ambient temperature"].mean()
                    tmp["PastPower"]=tmp["PastPower"].ffill()
                    tmp["PastAmbientTemp"]=tmp["PastAmbientTemp"].ffill()
                    tmp=tmp.loc[tmp["Timestamp"]>=current_month]

                    tmp4=tmp[["Power","Ambient temperature","PastPower","PastAmbientTemp"]]
                    tmp4=tmp4.dropna()

                    rebuilt=m.predict(tmp4)
                    rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                    tmp5=pd.concat([tmp[["Stator temperature"]],rebuilt],axis=1)

                    tmp5["Timestamp"]=tmp5.index
                    tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                    tmp5=tmp5.rename(columns={"Stator temperature":'Actual'})
                    tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                    tmp5["p25"]=0.675*rmse
                    tmp5["p10"]=1.282*rmse
                    tmp5["p75"]=-0.675*rmse
                    tmp5["p90"]=-1.282*rmse
                    tmp5.index=range(len(tmp5.index))

                    kpis["Stator temperature data"].iloc[item]=str(tmp5.to_dict())

                    tmp5=tmp5.dropna()

                    percentage_residuals_above_p25=100*len(tmp5[tmp5['Residuals']>0.675*rmse])/tmp5.shape[0]
                    percentage_residuals_above_p10=100*len(tmp5[tmp5['Residuals']>1.282*rmse])/tmp5.shape[0]

                    if tmp5.shape[0]>10 and (percentage_residuals_above_p25>50 or percentage_residuals_above_p10>20):
                        kpis["Stator temperature KPI"].iloc[item]='🔴'
                        kpis["Stator temperature KPI Color"].iloc[item]='red'

                    elif tmp5.shape[0]>10 and (percentage_residuals_above_p25>37.5 or percentage_residuals_above_p10>15):
                        kpis["Stator temperature KPI"].iloc[item]='🟠'
                        kpis["Stator temperature KPI Color"].iloc[item]='orange'

                    elif tmp5.shape[0]>10 and percentage_residuals_above_p25<=37.5 and percentage_residuals_above_p10<=15:
                        kpis["Stator temperature KPI"].iloc[item]='🟢'
                        kpis["Stator temperature KPI Color"].iloc[item]='green'

                except:
                    pass
                    
    except:
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Stator temperature KPI"]='⚪'
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Stator temperature KPI Color"]='white'

    try:
        m=joblib.load(os.path.join('./ML',wt_type+'_GearboxHSSBearingTemperature.joblib'))
        rmse=confidence_intervals.at[wt_type,"Gearbox HSS bearing temperature"]
        
        for item in range(len(kpis.index)):
            if kpis["Wind turbine type"].iloc[item]==wt_type:

                try:
                    current_month=kpis["MonthAsDate"].iloc[item]
                    current_wt=kpis["Wind turbine"].iloc[item]

                    timestamps=pd.period_range(current_month,current_month+relativedelta(months=1),freq='10min')[:-1].to_timestamp().tolist()

                    tmp=pd.DataFrame(index=timestamps)

                    tmp3=kpis["Filtered data"].iloc[item]
                    tmp3=pd.DataFrame(tmp3)
                    tmp3=tmp3[["Timestamp","Power","Ambient temperature","Gearbox HSS bearing temperature"]]
                    tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                    tmp3.set_index("Timestamp",inplace=True)

                    tmp=tmp.merge(tmp3,how='left',left_index=True,right_index=True)

                    try:
                        timestamps=pd.period_range(current_month+relativedelta(months=-1),current_month,freq='10min')[:-1].to_timestamp().tolist()

                        tmp2=pd.DataFrame(index=timestamps)

                        tmp3=kpis.loc[(kpis["Wind turbine"]==current_wt) & (kpis["MonthAsDate"]==(current_month+relativedelta(months=-1)))]["Filtered data"].iloc[0]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3[["Timestamp","Power","Ambient temperature","Gearbox HSS bearing temperature"]]
                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.set_index("Timestamp",inplace=True)

                        tmp2=tmp2.merge(tmp3,how='left',left_index=True,right_index=True)

                        tmp=pd.concat([tmp2,tmp])

                    except:
                        pass

                    tmp["Timestamp"]=tmp.index
                    tmp["PastPower"]=tmp.rolling(18)["Power"].mean()
                    tmp["PastAmbientTemp"]=tmp.rolling(18)["Ambient temperature"].mean()
                    tmp["PastPower"]=tmp["PastPower"].ffill()
                    tmp["PastAmbientTemp"]=tmp["PastAmbientTemp"].ffill()
                    tmp=tmp.loc[tmp["Timestamp"]>=current_month]

                    tmp4=tmp[["Power","Ambient temperature","PastPower","PastAmbientTemp"]]
                    tmp4=tmp4.dropna()

                    rebuilt=m.predict(tmp4)
                    rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                    tmp5=pd.concat([tmp[["Gearbox HSS bearing temperature"]],rebuilt],axis=1)
                    tmp5["Timestamp"]=tmp5.index
                    tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                    tmp5=tmp5.rename(columns={"Gearbox HSS bearing temperature":'Actual'})
                    tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                    tmp5["p25"]=0.675*rmse
                    tmp5["p10"]=1.282*rmse
                    tmp5["p75"]=-0.675*rmse
                    tmp5["p90"]=-1.282*rmse
                    tmp5.index=range(len(tmp5.index))

                    kpis["Gearbox HSS bearing temperature data"].iloc[item]=str(tmp5.to_dict())

                    tmp5=tmp5.dropna()

                    percentage_residuals_above_p25=100*len(tmp5[tmp5['Residuals']>0.675*rmse])/tmp5.shape[0]
                    percentage_residuals_above_p10=100*len(tmp5[tmp5['Residuals']>1.282*rmse])/tmp5.shape[0]

                    if tmp5.shape[0]>10 and (percentage_residuals_above_p25>50 or percentage_residuals_above_p10>20):
                        kpis["Gearbox HSS bearing temperature KPI"].iloc[item]='🔴'
                        kpis["Gearbox HSS bearing temperature KPI Color"].iloc[item]='red'

                    elif tmp5.shape[0]>10 and (percentage_residuals_above_p25>37.5 or percentage_residuals_above_p10>15):
                        kpis["Gearbox HSS bearing temperature KPI"].iloc[item]='🟠'
                        kpis["Gearbox HSS bearing temperature KPI Color"].iloc[item]='orange'

                    elif tmp5.shape[0]>10 and percentage_residuals_above_p25<=37.5 and percentage_residuals_above_p10<=15:
                        kpis["Gearbox HSS bearing temperature KPI"].iloc[item]='🟢'
                        kpis["Gearbox HSS bearing temperature KPI Color"].iloc[item]='green'

                except:
                    pass
                    
    except:
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Gearbox HSS bearing temperature KPI"]='⚪'
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Gearbox HSS bearing temperature KPI Color"]='white'

    try:
        m=joblib.load(os.path.join('./ML',wt_type+'_GearboxIMSLSSBearingTemperature.joblib'))
        rmse=confidence_intervals.at[wt_type,"Gearbox IMS/LSS bearing temperature"]
        
        for item in range(len(kpis.index)):
            if kpis["Wind turbine type"].iloc[item]==wt_type:

                try:
                    current_month=kpis["MonthAsDate"].iloc[item]
                    current_wt=kpis["Wind turbine"].iloc[item]

                    timestamps=pd.period_range(current_month,current_month+relativedelta(months=1),freq='10min')[:-1].to_timestamp().tolist()

                    tmp=pd.DataFrame(index=timestamps)

                    tmp3=kpis["Filtered data"].iloc[item]
                    tmp3=pd.DataFrame(tmp3)
                    tmp3=tmp3[["Timestamp","Power","Ambient temperature","Gearbox IMS/LSS bearing temperature"]]
                    tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                    tmp3.set_index("Timestamp",inplace=True)

                    tmp=tmp.merge(tmp3,how='left',left_index=True,right_index=True)

                    try:
                        timestamps=pd.period_range(current_month+relativedelta(months=-1),current_month,freq='10min')[:-1].to_timestamp().tolist()

                        tmp2=pd.DataFrame(index=timestamps)

                        tmp3=kpis.loc[(kpis["Wind turbine"]==current_wt) & (kpis["MonthAsDate"]==(current_month+relativedelta(months=-1)))]["Filtered data"].iloc[0]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3[["Timestamp","Power","Ambient temperature","Gearbox IMS/LSS bearing temperature"]]
                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.set_index("Timestamp",inplace=True)

                        tmp2=tmp2.merge(tmp3,how='left',left_index=True,right_index=True)

                        tmp=pd.concat([tmp2,tmp])

                    except:
                        pass

                    tmp["Timestamp"]=tmp.index
                    tmp["PastPower"]=tmp.rolling(18)["Power"].mean()
                    tmp["PastAmbientTemp"]=tmp.rolling(18)["Ambient temperature"].mean()
                    tmp["PastPower"]=tmp["PastPower"].ffill()
                    tmp["PastAmbientTemp"]=tmp["PastAmbientTemp"].ffill()

                    tmp=tmp.loc[tmp["Timestamp"]>=current_month]

                    tmp4=tmp[["Power","Ambient temperature","PastPower","PastAmbientTemp"]]
                    tmp4=tmp4.dropna()

                    rebuilt=m.predict(tmp4)
                    rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                    tmp5=pd.concat([tmp[["Gearbox IMS/LSS bearing temperature"]],rebuilt],axis=1)

                    tmp5["Timestamp"]=tmp5.index
                    tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))

                    tmp5=tmp5.rename(columns={"Gearbox IMS/LSS bearing temperature":'Actual'})

                    tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                    tmp5["p25"]=0.675*rmse
                    tmp5["p10"]=1.282*rmse
                    tmp5["p75"]=-0.675*rmse
                    tmp5["p90"]=-1.282*rmse

                    tmp5.index=range(len(tmp5.index))

                    kpis["Gearbox IMS/LSS bearing temperature data"].iloc[item]=str(tmp5.to_dict())

                    tmp5=tmp5.dropna()

                    percentage_residuals_above_p25=100*len(tmp5[tmp5['Residuals']>0.675*rmse])/tmp5.shape[0]
                    percentage_residuals_above_p10=100*len(tmp5[tmp5['Residuals']>1.282*rmse])/tmp5.shape[0]

                    if tmp5.shape[0]>10 and (percentage_residuals_above_p25>50 or percentage_residuals_above_p10>20):
                        kpis["Gearbox IMS/LSS bearing temperature KPI"].iloc[item]='🔴'
                        kpis["Gearbox IMS/LSS bearing temperature KPI Color"].iloc[item]='red'

                    elif tmp5.shape[0]>10 and (percentage_residuals_above_p25>37.5 or percentage_residuals_above_p10>15):
                        kpis["Gearbox IMS/LSS bearing temperature KPI"].iloc[item]='🟠'
                        kpis["Gearbox IMS/LSS bearing temperature KPI Color"].iloc[item]='orange'

                    elif tmp5.shape[0]>10 and percentage_residuals_above_p25<=37.5 and percentage_residuals_above_p10<=15:
                        kpis["Gearbox IMS/LSS bearing temperature KPI"].iloc[item]='🟢'
                        kpis["Gearbox IMS/LSS bearing temperature KPI Color"].iloc[item]='green'

                except:
                    pass

    except:
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Gearbox IMS/LSS bearing temperature KPI"]='⚪'
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Gearbox IMS/LSS bearing temperature KPI Color"]='white'

    try:
        m=joblib.load(os.path.join('./ML',wt_type+'_GeneratorBearingFrontTemperature.joblib'))
        rmse=confidence_intervals.at[wt_type,"Generator bearing front temperature"]
        
        for item in range(len(kpis.index)):
            if kpis["Wind turbine type"].iloc[item]==wt_type:

                try:
                    current_month=kpis["MonthAsDate"].iloc[item]
                    current_wt=kpis["Wind turbine"].iloc[item]

                    timestamps=pd.period_range(current_month,current_month+relativedelta(months=1),freq='10min')[:-1].to_timestamp().tolist()

                    tmp=pd.DataFrame(index=timestamps)

                    tmp3=kpis["Filtered data"].iloc[item]
                    tmp3=pd.DataFrame(tmp3)
                    tmp3=tmp3[["Timestamp","Power","Ambient temperature","Generator bearing front temperature"]]
                    tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                    tmp3.set_index("Timestamp",inplace=True)

                    tmp=tmp.merge(tmp3,how='left',left_index=True,right_index=True)

                    try:
                        timestamps=pd.period_range(current_month+relativedelta(months=-1),current_month,freq='10min')[:-1].to_timestamp().tolist()

                        tmp2=pd.DataFrame(index=timestamps)

                        tmp3=kpis.loc[(kpis["Wind turbine"]==current_wt) & (kpis["MonthAsDate"]==(current_month+relativedelta(months=-1)))]["Filtered data"].iloc[0]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3[["Timestamp","Power","Ambient temperature","Generator bearing front temperature"]]
                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.set_index("Timestamp",inplace=True)

                        tmp2=tmp2.merge(tmp3,how='left',left_index=True,right_index=True)

                        tmp=pd.concat([tmp2,tmp])

                    except:
                        pass

                    tmp["Timestamp"]=tmp.index
                    tmp["PastPower"]=tmp.rolling(144)["Power"].mean()
                    tmp["PastAmbientTemp"]=tmp.rolling(144)["Ambient temperature"].mean()
                    tmp["PastPower"]=tmp["PastPower"].ffill()
                    tmp["PastAmbientTemp"]=tmp["PastAmbientTemp"].ffill()
                    tmp=tmp.loc[tmp["Timestamp"]>=current_month]

                    tmp4=tmp[["Power","Ambient temperature","PastPower","PastAmbientTemp"]]
                    tmp4=tmp4.dropna()

                    rebuilt=m.predict(tmp4)
                    rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                    tmp5=pd.concat([tmp[["Generator bearing front temperature"]],rebuilt],axis=1)

                    tmp5["Timestamp"]=tmp5.index
                    tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))

                    tmp5=tmp5.rename(columns={"Generator bearing front temperature":'Actual'})

                    tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                    tmp5["p25"]=0.675*rmse
                    tmp5["p10"]=1.282*rmse
                    tmp5["p75"]=-0.675*rmse
                    tmp5["p90"]=-1.282*rmse

                    tmp5.index=range(len(tmp5.index))

                    kpis["Generator bearing front temperature data"].iloc[item]=str(tmp5.to_dict())

                    tmp5=tmp5.dropna()

                    percentage_residuals_above_p25=100*len(tmp5[tmp5['Residuals']>0.675*rmse])/tmp5.shape[0]
                    percentage_residuals_above_p10=100*len(tmp5[tmp5['Residuals']>1.282*rmse])/tmp5.shape[0]

                    if tmp5.shape[0]>10 and (percentage_residuals_above_p25>50 or percentage_residuals_above_p10>20):
                        kpis["Generator bearing front temperature KPI"].iloc[item]='🔴'
                        kpis["Generator bearing front temperature KPI Color"].iloc[item]='red'

                    elif tmp5.shape[0]>10 and (percentage_residuals_above_p25>37.5 or percentage_residuals_above_p10>15):
                        kpis["Generator bearing front temperature KPI"].iloc[item]='🟠'
                        kpis["Generator bearing front temperature KPI Color"].iloc[item]='orange'

                    elif tmp5.shape[0]>10 and percentage_residuals_above_p25<=37.5 and percentage_residuals_above_p10<=15:
                        kpis["Generator bearing front temperature KPI"].iloc[item]='🟢'
                        kpis["Generator bearing front temperature KPI Color"].iloc[item]='green'

                except:
                    pass
                    
    except:
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Generator bearing front temperature KPI"]='⚪'
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Generator bearing front temperature KPI Color"]='white'

    try:
        m=joblib.load(os.path.join('./ML',wt_type+'_GeneratorBearingRearTemperature.joblib'))
        rmse=confidence_intervals.at[wt_type,"Generator bearing rear temperature"]
        
        for item in range(len(kpis.index)):
            if kpis["Wind turbine type"].iloc[item]==wt_type:

                try:
                    current_month=kpis["MonthAsDate"].iloc[item]
                    current_wt=kpis["Wind turbine"].iloc[item]

                    timestamps=pd.period_range(current_month,current_month+relativedelta(months=1),freq='10min')[:-1].to_timestamp().tolist()

                    tmp=pd.DataFrame(index=timestamps)

                    tmp3=kpis["Filtered data"].iloc[item]
                    tmp3=pd.DataFrame(tmp3)
                    tmp3=tmp3[["Timestamp","Power","Ambient temperature","Generator bearing rear temperature"]]
                    tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                    tmp3.set_index("Timestamp",inplace=True)

                    tmp=tmp.merge(tmp3,how='left',left_index=True,right_index=True)

                    try:
                        timestamps=pd.period_range(current_month+relativedelta(months=-1),current_month,freq='10min')[:-1].to_timestamp().tolist()

                        tmp2=pd.DataFrame(index=timestamps)

                        tmp3=kpis.loc[(kpis["Wind turbine"]==current_wt) & (kpis["MonthAsDate"]==(current_month+relativedelta(months=-1)))]["Filtered data"].iloc[0]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3[["Timestamp","Power","Ambient temperature","Generator bearing rear temperature"]]
                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.set_index("Timestamp",inplace=True)

                        tmp2=tmp2.merge(tmp3,how='left',left_index=True,right_index=True)

                        tmp=pd.concat([tmp2,tmp])

                    except:
                        pass

                    tmp["Timestamp"]=tmp.index
                    tmp["PastPower"]=tmp.rolling(144)["Power"].mean()
                    tmp["PastAmbientTemp"]=tmp.rolling(144)["Ambient temperature"].mean()
                    tmp["PastPower"]=tmp["PastPower"].ffill()
                    tmp["PastAmbientTemp"]=tmp["PastAmbientTemp"].ffill()

                    tmp=tmp.loc[tmp["Timestamp"]>=current_month]

                    tmp4=tmp[["Power","Ambient temperature","PastPower","PastAmbientTemp"]]
                    tmp4=tmp4.dropna()

                    rebuilt=m.predict(tmp4)
                    rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                    tmp5=pd.concat([tmp[["Generator bearing rear temperature"]],rebuilt],axis=1)

                    tmp5["Timestamp"]=tmp5.index
                    tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                    tmp5=tmp5.rename(columns={"Generator bearing rear temperature":'Actual'})
                    tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                    tmp5["p25"]=0.675*rmse
                    tmp5["p10"]=1.282*rmse
                    tmp5["p75"]=-0.675*rmse
                    tmp5["p90"]=-1.282*rmse
                    tmp5.index=range(len(tmp5.index))

                    kpis["Generator bearing rear temperature data"].iloc[item]=str(tmp5.to_dict())

                    tmp5=tmp5.dropna()

                    percentage_residuals_above_p25=100*len(tmp5[tmp5['Residuals']>0.675*rmse])/tmp5.shape[0]
                    percentage_residuals_above_p10=100*len(tmp5[tmp5['Residuals']>1.282*rmse])/tmp5.shape[0]

                    if tmp5.shape[0]>10 and (percentage_residuals_above_p25>50 or percentage_residuals_above_p10>20):
                        kpis["Generator bearing rear temperature KPI"].iloc[item]='🔴'
                        kpis["Generator bearing rear temperature KPI Color"].iloc[item]='red'

                    elif tmp5.shape[0]>10 and (percentage_residuals_above_p25>37.5 or percentage_residuals_above_p10>15):
                        kpis["Generator bearing rear temperature KPI"].iloc[item]='🟠'
                        kpis["Generator bearing rear temperature KPI Color"].iloc[item]='orange'

                    elif tmp5.shape[0]>10 and percentage_residuals_above_p25<=37.5 and percentage_residuals_above_p10<=15:
                        kpis["Generator bearing rear temperature KPI"].iloc[item]='🟢'
                        kpis["Generator bearing rear temperature KPI Color"].iloc[item]='green'

                except:
                    pass
                    
    except:
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Generator bearing rear temperature KPI"]='⚪'
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Generator bearing rear temperature KPI Color"]='white'

    try:
        m=joblib.load(os.path.join('./ML',wt_type+'_MainBearingTemperature.joblib'))
        rmse=confidence_intervals.at[wt_type,"Main bearing temperature"]
        
        for item in range(len(kpis.index)):
            if kpis["Wind turbine type"].iloc[item]==wt_type:

                try:
                    current_month=kpis["MonthAsDate"].iloc[item]
                    current_wt=kpis["Wind turbine"].iloc[item]

                    timestamps=pd.period_range(current_month,current_month+relativedelta(months=1),freq='10min')[:-1].to_timestamp().tolist()

                    tmp=pd.DataFrame(index=timestamps)

                    tmp3=kpis["Filtered data"].iloc[item]
                    tmp3=pd.DataFrame(tmp3)
                    tmp3=tmp3[["Timestamp","Power","Ambient temperature","Main bearing temperature"]]
                    tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                    tmp3.set_index("Timestamp",inplace=True)

                    tmp=tmp.merge(tmp3,how='left',left_index=True,right_index=True)

                    try:
                        timestamps=pd.period_range(current_month+relativedelta(months=-1),current_month,freq='10min')[:-1].to_timestamp().tolist()

                        tmp2=pd.DataFrame(index=timestamps)

                        tmp3=kpis.loc[(kpis["Wind turbine"]==current_wt) & (kpis["MonthAsDate"]==(current_month+relativedelta(months=-1)))]["Filtered data"].iloc[0]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3[["Timestamp","Power","Ambient temperature","Main bearing temperature"]]
                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.set_index("Timestamp",inplace=True)

                        tmp2=tmp2.merge(tmp3,how='left',left_index=True,right_index=True)

                        tmp=pd.concat([tmp2,tmp])

                    except:
                        pass

                    tmp["Timestamp"]=tmp.index
                    tmp["PastPower"]=tmp.rolling(144)["Power"].mean()
                    tmp["PastAmbientTemp"]=tmp.rolling(144)["Ambient temperature"].mean()
                    tmp["PastPower"]=tmp["PastPower"].ffill()
                    tmp["PastAmbientTemp"]=tmp["PastAmbientTemp"].ffill()
                    tmp=tmp.loc[tmp["Timestamp"]>=current_month]

                    tmp4=tmp[["Power","Ambient temperature","PastPower","PastAmbientTemp"]]
                    tmp4=tmp4.dropna()

                    rebuilt=m.predict(tmp4)
                    rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                    tmp5=pd.concat([tmp[["Main bearing temperature"]],rebuilt],axis=1)

                    tmp5["Timestamp"]=tmp5.index
                    tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                    tmp5=tmp5.rename(columns={"Main bearing temperature":'Actual'})
                    tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                    tmp5["p25"]=0.675*rmse
                    tmp5["p10"]=1.282*rmse
                    tmp5["p75"]=-0.675*rmse
                    tmp5["p90"]=-1.282*rmse
                    tmp5.index=range(len(tmp5.index))

                    kpis["Main bearing temperature data"].iloc[item]=str(tmp5.to_dict())

                    tmp5=tmp5.dropna()

                    percentage_residuals_above_p25=100*len(tmp5[tmp5['Residuals']>0.675*rmse])/tmp5.shape[0]
                    percentage_residuals_above_p10=100*len(tmp5[tmp5['Residuals']>1.282*rmse])/tmp5.shape[0]

                    if tmp5.shape[0]>10 and (percentage_residuals_above_p25>50 or percentage_residuals_above_p10>20):
                        kpis["Main bearing temperature KPI"].iloc[item]='🔴'
                        kpis["Main bearing temperature KPI Color"].iloc[item]='red'

                    elif tmp5.shape[0]>10 and (percentage_residuals_above_p25>37.5 or percentage_residuals_above_p10>15):
                        kpis["Main bearing temperature KPI"].iloc[item]='🟠'
                        kpis["Main bearing temperature KPI Color"].iloc[item]='orange'

                    elif tmp5.shape[0]>10 and percentage_residuals_above_p25<=37.5 and percentage_residuals_above_p10<=15:
                        kpis["Main bearing temperature KPI"].iloc[item]='🟢'
                        kpis["Main bearing temperature KPI Color"].iloc[item]='green'

                except:
                    pass
                    

    except:
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Main bearing temperature KPI"]='⚪'
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Main bearing temperature KPI Color"]='white'

    try:
        m=joblib.load(os.path.join('./ML',wt_type+'_GearboxOilTemperature.joblib'))
        rmse=confidence_intervals.at[wt_type,"Gearbox oil temperature"]
        
        for item in range(len(kpis.index)):
            if kpis["Wind turbine type"].iloc[item]==wt_type:
                
                try:
                    current_month=kpis["MonthAsDate"].iloc[item]
                    current_wt=kpis["Wind turbine"].iloc[item]

                    timestamps=pd.period_range(current_month,current_month+relativedelta(months=1),freq='10min')[:-1].to_timestamp().tolist()

                    tmp=pd.DataFrame(index=timestamps)

                    tmp3=kpis["Filtered data"].iloc[item]
                    tmp3=pd.DataFrame(tmp3)
                    tmp3=tmp3[["Timestamp","Power","Ambient temperature","Gearbox oil temperature"]]
                    tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                    tmp3.set_index("Timestamp",inplace=True)

                    tmp=tmp.merge(tmp3,how='left',left_index=True,right_index=True)

                    try:
                        timestamps=pd.period_range(current_month+relativedelta(months=-1),current_month,freq='10min')[:-1].to_timestamp().tolist()

                        tmp2=pd.DataFrame(index=timestamps)

                        tmp3=kpis.loc[(kpis["Wind turbine"]==current_wt) & (kpis["MonthAsDate"]==(current_month+relativedelta(months=-1)))]["Filtered data"].iloc[0]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3[["Timestamp","Power","Ambient temperature","Gearbox oil temperature"]]
                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.set_index("Timestamp",inplace=True)

                        tmp2=tmp2.merge(tmp3,how='left',left_index=True,right_index=True)

                        tmp=pd.concat([tmp2,tmp])

                    except:
                        pass

                    tmp["Timestamp"]=tmp.index
                    tmp["PastPower"]=tmp.rolling(18)["Power"].mean()
                    tmp["PastAmbientTemp"]=tmp.rolling(18)["Ambient temperature"].mean()
                    tmp["PastPower"]=tmp["PastPower"].ffill()
                    tmp["PastAmbientTemp"]=tmp["PastAmbientTemp"].ffill()
                    tmp=tmp.loc[tmp["Timestamp"]>=current_month]

                    tmp4=tmp[["Power","Ambient temperature","PastPower","PastAmbientTemp"]]
                    tmp4=tmp4.dropna()

                    rebuilt=m.predict(tmp4)
                    rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                    tmp5=pd.concat([tmp[["Gearbox oil temperature"]],rebuilt],axis=1)
                    tmp5["Timestamp"]=tmp5.index
                    tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                    tmp5=tmp5.rename(columns={"Gearbox oil temperature":'Actual'})
                    tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                    tmp5["p25"]=0.675*rmse
                    tmp5["p10"]=1.282*rmse
                    tmp5["p75"]=-0.675*rmse
                    tmp5["p90"]=-1.282*rmse
                    tmp5.index=range(len(tmp5.index))

                    kpis["Gearbox oil temperature data"].iloc[item]=str(tmp5.to_dict())

                    tmp5=tmp5.dropna()

                    percentage_residuals_above_p25=100*len(tmp5[tmp5['Residuals']>0.675*rmse])/tmp5.shape[0]
                    percentage_residuals_above_p10=100*len(tmp5[tmp5['Residuals']>1.282*rmse])/tmp5.shape[0]

                    if tmp5.shape[0]>10 and (percentage_residuals_above_p25>50 or percentage_residuals_above_p10>20):
                        kpis["Gearbox oil temperature KPI"].iloc[item]='🔴'
                        kpis["Gearbox oil temperature KPI Color"].iloc[item]='red'

                    elif tmp5.shape[0]>10 and (percentage_residuals_above_p25>37.5 or percentage_residuals_above_p10>15):
                        kpis["Gearbox oil temperature KPI"].iloc[item]='🟠'
                        kpis["Gearbox oil temperature KPI Color"].iloc[item]='orange'

                    elif tmp5.shape[0]>10 and percentage_residuals_above_p25<=37.5 and percentage_residuals_above_p10<=15:
                        kpis["Gearbox oil temperature KPI"].iloc[item]='🟢'
                        kpis["Gearbox oil temperature KPI Color"].iloc[item]='green'

                except:
                    pass
                    

    except:
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Gearbox oil temperature KPI"]='⚪'
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Gearbox oil temperature KPI Color"]='white'

    try:
        m=joblib.load(os.path.join('./ML',wt_type+'_MetalParticleCount.joblib'))
        
        for item in range(len(kpis.index)):
            if kpis["Wind turbine type"].iloc[item]==wt_type:

                try:
                    current_month=kpis["MonthAsDate"].iloc[item]
                    current_wt=kpis["Wind turbine"].iloc[item]

                    timestamps=pd.period_range(current_month,current_month+relativedelta(months=1),freq='10min')[:-1].to_timestamp().tolist()

                    tmp=pd.DataFrame(index=timestamps)

                    tmp3=kpis["Filtered data"].iloc[item]
                    tmp3=pd.DataFrame(tmp3)
                    tmp3=tmp3[["Timestamp","Power","Gearbox HSS bearing temperature","Metal particle count"]]
                    tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                    tmp3.set_index("Timestamp",inplace=True)

                    tmp=tmp.merge(tmp3,how='left',left_index=True,right_index=True)

                    try:
                        timestamps=pd.period_range(current_month+relativedelta(months=-1),current_month,freq='10min')[:-1].to_timestamp().tolist()

                        tmp2=pd.DataFrame(index=timestamps)

                        tmp3=kpis.loc[(kpis["Wind turbine"]==current_wt) & (kpis["MonthAsDate"]==(current_month+relativedelta(months=-1)))]["Filtered data"].iloc[0]
                        tmp3=pd.DataFrame(tmp3)
                        tmp3=tmp3[["Timestamp","Power","Gearbox HSS bearing temperature","Metal particle count"]]
                        tmp3["Timestamp"]=tmp3["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
                        tmp3.set_index("Timestamp",inplace=True)

                        tmp2=tmp2.merge(tmp3,how='left',left_index=True,right_index=True)

                        tmp=pd.concat([tmp2,tmp])

                    except:
                        pass
                    
                    tmp=tmp[["Power","Gearbox HSS bearing temperature","Metal particle count"]]

                    tmp4=tmp.resample('3H').sum()
                    tmp5=tmp.resample('3H').mean()
                    tmp6=tmp.resample('3H').count()
                    tmp6=tmp6[["Power"]].rename(columns={"Power":'Count'})
                    tmp=pd.concat([tmp4[["Metal particle count"]],tmp5[["Power","Gearbox HSS bearing temperature"]],tmp6[["Count"]]],axis=1)

                    tmp=tmp.loc[tmp.index>=current_month]

                    tmp.loc[tmp["Count"]<10,"Power"]=np.nan
                    
                    tmp=tmp[["Power","Gearbox HSS bearing temperature","Metal particle count"]]

                    tmp4=tmp[["Power","Gearbox HSS bearing temperature"]]
                    tmp4=tmp4.dropna()

                    rebuilt=m.predict(tmp4)
                    rebuilt=pd.DataFrame(rebuilt,index=tmp4.index,columns=["Predicted"])

                    tmp5=pd.concat([tmp[["Metal particle count"]],rebuilt],axis=1)

                    tmp5["Timestamp"]=tmp5.index
                    tmp5["Timestamp"]=tmp5["Timestamp"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                    tmp5=tmp5.rename(columns={"Metal particle count":'Actual'})
                    tmp5['Actual']=tmp5['Actual']
                    tmp5["Residuals"]=tmp5["Actual"]-tmp5["Predicted"]
                    tmp5.index=range(len(tmp5.index))
                    kpis["Metal particle count data"].iloc[item]=str(tmp5.to_dict())
                    tmp5=tmp5.dropna()
                    total_positive_residuals=tmp5.loc[tmp5['Residuals']>0]
                    total_positive_residuals=total_positive_residuals['Residuals'].sum()
                    if tmp5.shape[0]>50 and total_positive_residuals>20:
                        kpis["Metal particle count KPI"].iloc[item]='🔴'
                        kpis["Metal particle count KPI Color"].iloc[item]='red'
                    elif tmp5.shape[0]>50 and total_positive_residuals>10:
                        kpis["Metal particle count KPI"].iloc[item]='🟠'
                        kpis["Metal particle count KPI Color"].iloc[item]='orange'
                    elif tmp5.shape[0]>50 and total_positive_residuals<=10:
                        kpis["Metal particle count KPI"].iloc[item]='🟢'
                        kpis["Metal particle count KPI Color"].iloc[item]='green'
                except:
                    pass
                    

    except:
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Metal particle count KPI"]='⚪'
        kpis.loc[kpis["Wind turbine type"]==wt_type,"Metal particle count KPI Color"]='white'













kpis["Pitch vs Power"]=np.nan
kpis["Pitch vs Power KPI"]='⚫'
for i in range(len(kpis.index)):
    try:
        expected_relationship=wt_types_relationships.at[kpis["Wind turbine type"].iloc[i],"Pitch vs Power"]
        expected_relationship=pd.DataFrame(expected_relationship)
        bin_size=(expected_relationship.index[-1]-expected_relationship.index[0])/20
        bin_centers=[expected_relationship.index[0]+bin_size/2+j*bin_size for j in range(20)]
        tmp=kpis["Filtered data"].iloc[i]
        tmp=pd.DataFrame(tmp)
        check_if_yaw_misalignment_data=tmp["Yaw misalignment"].count()
        if check_if_yaw_misalignment_data>0:
            tmp=tmp.loc[(tmp["Yaw misalignment"]>=-5) & (tmp["Yaw misalignment"]<=5)]
        tmp=tmp[["Power","Blade angle"]].dropna()
        tmp["Blade angle"]=tmp["Blade angle"].apply(lambda x: ((x+180)%360)-180)
        tmp["Power"]=tmp["Power"].apply(lambda x: round(x,0))  
        tmp=tmp.loc[tmp["Power"].isin(expected_relationship.index)]
        try:
            tmp["flag"]=tmp.apply(lambda row: 0 if (row["Blade angle"]>=expected_relationship.at[row["Power"],"Avg"]-1.96*expected_relationship.at[row["Power"],"Stdev"])&(row["Blade angle"]<=expected_relationship.at[row["Power"],"Avg"]+1.96*expected_relationship.at[row["Power"],"Stdev"]) else 1,axis=1)
            tmp_dico={bin_center:tmp.loc[(tmp["Power"]>bin_center-bin_size/2)&(tmp["Power"]<=bin_center+bin_size/2),"flag"].tolist() for bin_center in bin_centers}
            tmp_dico={elem:100-100*sum(tmp_dico[elem])/len(tmp_dico[elem]) if len(tmp_dico[elem])>10 else np.nan for elem in tmp_dico}
            new_dico={elem:"red" if tmp_dico[elem]<80 else ("orange" if (tmp_dico[elem]>=80 and tmp_dico[elem]<90) else ("green" if tmp_dico[elem]>=90 else ("black"))) for elem in tmp_dico}
            kpis["Pitch vs Power"].iloc[i]=str(new_dico)
            blacks=sum((value=="black") for value in new_dico.values())
            orange_and_reds=3*sum((value=="red") for value in new_dico.values())+sum((value=="orange") for value in new_dico.values())
            if blacks==20:
                kpis["Pitch vs Power KPI"].iloc[i]='⚫'
            else:
                if orange_and_reds>10:
                    kpis["Pitch vs Power KPI"].iloc[i]='🔴'
                elif orange_and_reds<6:
                    kpis["Pitch vs Power KPI"].iloc[i]='🟢'
                elif orange_and_reds>=6 and orange_and_reds<=10:
                    kpis["Pitch vs Power KPI"].iloc[i]='🟠'
        except:
            tmp_dico={bin_center:"black" for bin_center in bin_centers}
            kpis["Pitch vs Power"].iloc[i]=str(tmp_dico)
    except:
        kpis["Pitch vs Power"].iloc[i]=str({})







        

kpis["Pitch vs RPM"]=np.nan
kpis["Pitch vs RPM KPI"]='⚫'
for i in range(len(kpis.index)):
    try:
        expected_relationship=wt_types_relationships.at[kpis["Wind turbine type"].iloc[i],"Pitch vs RPM"]
        expected_relationship=pd.DataFrame(expected_relationship)
        bin_size=(expected_relationship.index[-1]-expected_relationship.index[0])/20
        bin_centers=[expected_relationship.index[0]+bin_size/2+j*bin_size for j in range(20)]
        tmp=kpis["Filtered data"].iloc[i]
        tmp=pd.DataFrame(tmp)
        check_if_yaw_misalignment_data=tmp["Yaw misalignment"].count()
        if check_if_yaw_misalignment_data>0:
            tmp=tmp.loc[(tmp["Yaw misalignment"]>=-5) & (tmp["Yaw misalignment"]<=5)]
        tmp=tmp[["Rotor speed","Blade angle"]].dropna()
        tmp["Blade angle"]=tmp["Blade angle"].apply(lambda x: ((x+180)%360)-180)
        tmp["Rotor speed"]=tmp["Rotor speed"].apply(lambda x: round(x,1))  
        tmp=tmp.loc[tmp["Rotor speed"].isin(expected_relationship.index)]
        try:
            tmp["flag"]=tmp.apply(lambda row: 0 if (row["Blade angle"]>=expected_relationship.at[row["Rotor speed"],"Avg"]-1.96*expected_relationship.at[row["Rotor speed"],"Stdev"])&(row["Blade angle"]<=expected_relationship.at[row["Rotor speed"],"Avg"]+1.96*expected_relationship.at[row["Rotor speed"],"Stdev"]) else 1,axis=1)
            tmp_dico={bin_center:tmp.loc[(tmp["Rotor speed"]>bin_center-bin_size/2)&(tmp["Rotor speed"]<=bin_center+bin_size/2),"flag"].tolist() for bin_center in bin_centers}
            tmp_dico={elem:100-100*sum(tmp_dico[elem])/len(tmp_dico[elem]) if len(tmp_dico[elem])>10 else np.nan for elem in tmp_dico}
            new_dico={elem:"red" if tmp_dico[elem]<80 else ("orange" if (tmp_dico[elem]>=80 and tmp_dico[elem]<90) else ("green" if tmp_dico[elem]>=90 else ("black"))) for elem in tmp_dico}
            kpis["Pitch vs RPM"].iloc[i]=str(new_dico)
            blacks=sum((value=="black") for value in new_dico.values())
            orange_and_reds=3*sum((value=="red") for value in new_dico.values())+sum((value=="orange") for value in new_dico.values())
            if blacks==20:
                kpis["Pitch vs RPM KPI"].iloc[i]='⚫'
            else:
                if orange_and_reds>10:
                    kpis["Pitch vs RPM KPI"].iloc[i]='🔴'
                elif orange_and_reds<6:
                    kpis["Pitch vs RPM KPI"].iloc[i]='🟢'
                elif orange_and_reds>=6 and orange_and_reds<=10:
                    kpis["Pitch vs RPM KPI"].iloc[i]='🟠'
        except:
            tmp_dico={bin_center:"black" for bin_center in bin_centers}
            kpis["Pitch vs RPM"].iloc[i]=str(tmp_dico)
    except:
        kpis["Pitch vs RPM"].iloc[i]=str({})
        




kpis["RPM vs Power"]=np.nan
kpis["RPM vs Power KPI"]='⚫'
for i in range(len(kpis.index)):
    try:
        expected_relationship=wt_types_relationships.at[kpis["Wind turbine type"].iloc[i],"RPM vs Power"]
        expected_relationship=pd.DataFrame(expected_relationship)
        bin_size=(expected_relationship.index[-1]-expected_relationship.index[0])/20
        bin_centers=[expected_relationship.index[0]+bin_size/2+j*bin_size for j in range(20)]
        tmp=kpis["Filtered data"].iloc[i]
        tmp=pd.DataFrame(tmp)
        check_if_yaw_misalignment_data=tmp["Yaw misalignment"].count()
        if check_if_yaw_misalignment_data>0:
            tmp=tmp.loc[(tmp["Yaw misalignment"]>=-5) & (tmp["Yaw misalignment"]<=5)]
        tmp=tmp[["Power","Rotor speed"]].dropna()
        tmp["Power"]=tmp["Power"].apply(lambda x: round(x,0))  
        tmp=tmp.loc[tmp["Power"].isin(expected_relationship.index)]
        try:
            tmp["flag"]=tmp.apply(lambda row: 0 if (row["Rotor speed"]>=expected_relationship.at[row["Power"],"Avg"]-1.96*expected_relationship.at[row["Power"],"Stdev"])&(row["Rotor speed"]<=expected_relationship.at[row["Power"],"Avg"]+1.96*expected_relationship.at[row["Power"],"Stdev"]) else 1,axis=1)
            tmp_dico={bin_center:tmp.loc[(tmp["Power"]>bin_center-bin_size/2)&(tmp["Power"]<=bin_center+bin_size/2),"flag"].tolist() for bin_center in bin_centers}
            tmp_dico={elem:100-100*sum(tmp_dico[elem])/len(tmp_dico[elem]) if len(tmp_dico[elem])>10 else np.nan for elem in tmp_dico}
            new_dico={elem:"red" if tmp_dico[elem]<80 else ("orange" if (tmp_dico[elem]>=80 and tmp_dico[elem]<90) else ("green" if tmp_dico[elem]>=90 else ("black"))) for elem in tmp_dico}
            kpis["RPM vs Power"].iloc[i]=str(new_dico)
            blacks=sum((value=="black") for value in new_dico.values())
            orange_and_reds=3*sum((value=="red") for value in new_dico.values())+sum((value=="orange") for value in new_dico.values())
            if blacks==20:
                kpis["RPM vs Power KPI"].iloc[i]='⚫'
            else:
                if orange_and_reds>10:
                    kpis["RPM vs Power KPI"].iloc[i]='🔴'
                elif orange_and_reds<6:
                    kpis["RPM vs Power KPI"].iloc[i]='🟢'
                elif orange_and_reds>=6 and orange_and_reds<=10:
                    kpis["RPM vs Power KPI"].iloc[i]='🟠'
        except:
            tmp_dico={bin_center:"black" for bin_center in bin_centers}
            kpis["RPM vs Power"].iloc[i]=str(tmp_dico)
    except:
        kpis["RPM vs Power"].iloc[i]=str({})











kpis["Power vs Speed"]=np.nan
kpis["Power vs Speed KPI"]='⚫'
for i in range(len(kpis.index)):
    try:
        expected_relationship=wt_types_relationships.at[kpis["Wind turbine type"].iloc[i],"Power vs Speed"]
        expected_relationship=pd.DataFrame(expected_relationship)
        bin_size=(expected_relationship.index[-1]-expected_relationship.index[0])/20
        bin_centers=[expected_relationship.index[0]+bin_size/2+j*bin_size for j in range(20)]
        tmp=kpis["Filtered data"].iloc[i]
        tmp=pd.DataFrame(tmp)
        check_if_yaw_misalignment_data=tmp["Yaw misalignment"].count()
        if check_if_yaw_misalignment_data>0:
            tmp=tmp.loc[(tmp["Yaw misalignment"]>=-5) & (tmp["Yaw misalignment"]<=5)]
        tmp=tmp[["Corrected wind speed","Power"]].dropna()
        tmp["Corrected wind speed"]=tmp["Corrected wind speed"].apply(lambda x: round(x,1))  
        tmp=tmp.loc[tmp["Corrected wind speed"].isin(expected_relationship.index)]
        try:
            tmp["flag"]=tmp.apply(lambda row: 0 if (row["Power"]>=expected_relationship.at[row["Corrected wind speed"],"Avg"]-1.96*expected_relationship.at[row["Corrected wind speed"],"Stdev"])&(row["Power"]<=expected_relationship.at[row["Corrected wind speed"],"Avg"]+1.96*expected_relationship.at[row["Corrected wind speed"],"Stdev"]) else 1,axis=1)
            tmp_dico={bin_center:tmp.loc[(tmp["Corrected wind speed"]>bin_center-bin_size/2)&(tmp["Corrected wind speed"]<=bin_center+bin_size/2),"flag"].tolist() for bin_center in bin_centers}
            tmp_dico={elem:100-100*sum(tmp_dico[elem])/len(tmp_dico[elem]) if len(tmp_dico[elem])>10 else np.nan for elem in tmp_dico}
            new_dico={elem:"red" if tmp_dico[elem]<80 else ("orange" if (tmp_dico[elem]>=80 and tmp_dico[elem]<90) else ("green" if tmp_dico[elem]>=90 else ("black"))) for elem in tmp_dico}
            kpis["Power vs Speed"].iloc[i]=str(new_dico)
            blacks=sum((value=="black") for value in new_dico.values())
            orange_and_reds=3*sum((value=="red") for value in new_dico.values())+sum((value=="orange") for value in new_dico.values())
            if blacks==20:
                kpis["Power vs Speed KPI"].iloc[i]='⚫'
            else:
                if orange_and_reds>10:
                    kpis["Power vs Speed KPI"].iloc[i]='🔴'
                elif orange_and_reds<6:
                    kpis["Power vs Speed KPI"].iloc[i]='🟢'
                elif orange_and_reds>=6 and orange_and_reds<=10:
                    kpis["Power vs Speed KPI"].iloc[i]='🟠'
        except:
            tmp_dico={bin_center:"black" for bin_center in bin_centers}
            kpis["Power vs Speed"].iloc[i]=str(tmp_dico)
    except:
        kpis["Power vs Speed"].iloc[i]=str({})
    

kpis["Pitch vs Speed"]=np.nan
kpis["Pitch vs Speed KPI"]='⚫'
for i in range(len(kpis.index)):
    try:
        expected_relationship=wt_types_relationships.at[kpis["Wind turbine type"].iloc[i],"Pitch vs Speed"]
        expected_relationship=pd.DataFrame(expected_relationship)
        bin_size=(expected_relationship.index[-1]-expected_relationship.index[0])/20
        bin_centers=[expected_relationship.index[0]+bin_size/2+j*bin_size for j in range(20)]
        tmp=kpis["Filtered data"].iloc[i]
        tmp=pd.DataFrame(tmp)
        check_if_yaw_misalignment_data=tmp["Yaw misalignment"].count()
        if check_if_yaw_misalignment_data>0:
            tmp=tmp.loc[(tmp["Yaw misalignment"]>=-5) & (tmp["Yaw misalignment"]<=5)]
        tmp=tmp[["Wind speed","Blade angle"]].dropna()
        tmp["Blade angle"]=tmp["Blade angle"].apply(lambda x: ((x+180)%360)-180)
        tmp["Wind speed"]=tmp["Wind speed"].apply(lambda x: round(x,1))  
        tmp=tmp.loc[tmp["Wind speed"].isin(expected_relationship.index)]
        try:
            tmp["flag"]=tmp.apply(lambda row: 0 if (row["Blade angle"]>=expected_relationship.at[row["Wind speed"],"Avg"]-1.96*expected_relationship.at[row["Wind speed"],"Stdev"])&(row["Blade angle"]<=expected_relationship.at[row["Wind speed"],"Avg"]+1.96*expected_relationship.at[row["Wind speed"],"Stdev"]) else 1,axis=1)
            tmp_dico={bin_center:tmp.loc[(tmp["Wind speed"]>bin_center-bin_size/2)&(tmp["Wind speed"]<=bin_center+bin_size/2),"flag"].tolist() for bin_center in bin_centers}
            tmp_dico={elem:100-100*sum(tmp_dico[elem])/len(tmp_dico[elem]) if len(tmp_dico[elem])>10 else np.nan for elem in tmp_dico}
            new_dico={elem:"red" if tmp_dico[elem]<80 else ("orange" if (tmp_dico[elem]>=80 and tmp_dico[elem]<90) else ("green" if tmp_dico[elem]>=90 else ("black"))) for elem in tmp_dico}
            kpis["Pitch vs Speed"].iloc[i]=str(new_dico)
            blacks=sum((value=="black") for value in new_dico.values())
            orange_and_reds=3*sum((value=="red") for value in new_dico.values())+sum((value=="orange") for value in new_dico.values())
            if blacks==20:
                kpis["Pitch vs Speed KPI"].iloc[i]='⚫'
            else:
                if orange_and_reds>10:
                    kpis["Pitch vs Speed KPI"].iloc[i]='🔴'
                elif orange_and_reds<6:
                    kpis["Pitch vs Speed KPI"].iloc[i]='🟢'
                elif orange_and_reds>=6 and orange_and_reds<=10:
                    kpis["Pitch vs Speed KPI"].iloc[i]='🟠'
        except:
            tmp_dico={bin_center:"black" for bin_center in bin_centers}
            kpis["Pitch vs Speed"].iloc[i]=str(tmp_dico)
    except:
        kpis["Pitch vs Speed"].iloc[i]=str({})


kpis["RPM vs Speed"]=np.nan
kpis["RPM vs Speed KPI"]='⚫'
for i in range(len(kpis.index)):
    try:
        expected_relationship=wt_types_relationships.at[kpis["Wind turbine type"].iloc[i],"RPM vs Speed"]
        expected_relationship=pd.DataFrame(expected_relationship)
        bin_size=(expected_relationship.index[-1]-expected_relationship.index[0])/20
        bin_centers=[expected_relationship.index[0]+bin_size/2+j*bin_size for j in range(20)]
        tmp=kpis["Filtered data"].iloc[i]
        tmp=pd.DataFrame(tmp)
        check_if_yaw_misalignment_data=tmp["Yaw misalignment"].count()
        if check_if_yaw_misalignment_data>0:
            tmp=tmp.loc[(tmp["Yaw misalignment"]>=-5) & (tmp["Yaw misalignment"]<=5)]
        tmp=tmp[["Wind speed","Rotor speed"]].dropna()
        tmp["Wind speed"]=tmp["Wind speed"].apply(lambda x: round(x,1))  
        tmp=tmp.loc[tmp["Wind speed"].isin(expected_relationship.index)]
        try:
            tmp["flag"]=tmp.apply(lambda row: 0 if (row["Rotor speed"]>=expected_relationship.at[row["Wind speed"],"Avg"]-1.96*expected_relationship.at[row["Wind speed"],"Stdev"])&(row["Rotor speed"]<=expected_relationship.at[row["Wind speed"],"Avg"]+1.96*expected_relationship.at[row["Wind speed"],"Stdev"]) else 1,axis=1)
            tmp_dico={bin_center:tmp.loc[(tmp["Wind speed"]>bin_center-bin_size/2)&(tmp["Wind speed"]<=bin_center+bin_size/2),"flag"].tolist() for bin_center in bin_centers}
            tmp_dico={elem:100-100*sum(tmp_dico[elem])/len(tmp_dico[elem]) if len(tmp_dico[elem])>10 else np.nan for elem in tmp_dico}
            new_dico={elem:"red" if tmp_dico[elem]<80 else ("orange" if (tmp_dico[elem]>=80 and tmp_dico[elem]<90) else ("green" if tmp_dico[elem]>=90 else ("black"))) for elem in tmp_dico}
            kpis["RPM vs Speed"].iloc[i]=str(new_dico)
            blacks=sum((value=="black") for value in new_dico.values())
            orange_and_reds=3*sum((value=="red") for value in new_dico.values())+sum((value=="orange") for value in new_dico.values())
            if blacks==20:
                kpis["RPM vs Speed KPI"].iloc[i]='⚫'
            else:
                if orange_and_reds>10:
                    kpis["RPM vs Speed KPI"].iloc[i]='🔴'
                elif orange_and_reds<6:
                    kpis["RPM vs Speed KPI"].iloc[i]='🟢'
                elif orange_and_reds>=6 and orange_and_reds<=10:
                    kpis["RPM vs Speed KPI"].iloc[i]='🟠'
        except:
            tmp_dico={bin_center:"black" for bin_center in bin_centers}
            kpis["RPM vs Speed"].iloc[i]=str(tmp_dico)
    except:
        kpis["RPM vs Speed"].iloc[i]=str({})




kpis["Power vs RPM"]=np.nan
kpis["Power vs RPM KPI"]='⚫'
for i in range(len(kpis.index)):
    try:
        expected_relationship=wt_types_relationships.at[kpis["Wind turbine type"].iloc[i],"Power vs RPM"]
        expected_relationship=pd.DataFrame(expected_relationship)
        bin_size=(expected_relationship.index[-1]-expected_relationship.index[0])/20
        bin_centers=[expected_relationship.index[0]+bin_size/2+j*bin_size for j in range(20)]
        tmp=kpis["Filtered data"].iloc[i]
        tmp=pd.DataFrame(tmp)
        check_if_yaw_misalignment_data=tmp["Yaw misalignment"].count()
        if check_if_yaw_misalignment_data>0:
            tmp=tmp.loc[(tmp["Yaw misalignment"]>=-5) & (tmp["Yaw misalignment"]<=5)]
        tmp=tmp[["Rotor speed","Power"]].dropna()
        tmp["Rotor speed"]=tmp["Rotor speed"].apply(lambda x: round(x,1))  
        tmp=tmp.loc[tmp["Rotor speed"].isin(expected_relationship.index)]
        try:
            tmp["flag"]=tmp.apply(lambda row: 0 if (row["Power"]>=expected_relationship.at[row["Rotor speed"],"Avg"]-1.96*expected_relationship.at[row["Rotor speed"],"Stdev"])&(row["Power"]<=expected_relationship.at[row["Rotor speed"],"Avg"]+1.96*expected_relationship.at[row["Rotor speed"],"Stdev"]) else 1,axis=1)
            tmp_dico={bin_center:tmp.loc[(tmp["Rotor speed"]>bin_center-bin_size/2)&(tmp["Rotor speed"]<=bin_center+bin_size/2),"flag"].tolist() for bin_center in bin_centers}
            tmp_dico={elem:100-100*sum(tmp_dico[elem])/len(tmp_dico[elem]) if len(tmp_dico[elem])>10 else np.nan for elem in tmp_dico}
            new_dico={elem:"red" if tmp_dico[elem]<80 else ("orange" if (tmp_dico[elem]>=80 and tmp_dico[elem]<90) else ("green" if tmp_dico[elem]>=90 else ("black"))) for elem in tmp_dico}
            kpis["Power vs RPM"].iloc[i]=str(new_dico)
            blacks=sum((value=="black") for value in new_dico.values())
            orange_and_reds=3*sum((value=="red") for value in new_dico.values())+sum((value=="orange") for value in new_dico.values())
            if blacks==20:
                kpis["Power vs RPM KPI"].iloc[i]='⚫'
            else:
                if orange_and_reds>10:
                    kpis["Power vs RPM KPI"].iloc[i]='🔴'
                elif orange_and_reds<6:
                    kpis["Power vs RPM KPI"].iloc[i]='🟢'
                elif orange_and_reds>=6 and orange_and_reds<=10:
                    kpis["Power vs RPM KPI"].iloc[i]='🟠'
        except:
            tmp_dico={bin_center:"black" for bin_center in bin_centers}
            kpis["Power vs RPM"].iloc[i]=str(tmp_dico)
    except:
        kpis["Power vs RPM"].iloc[i]=str({})























kpis["Dynamic yaw misalignment KPI"]='⚫'
kpis["Dynamic yaw misalignment KPI Color"]='black'
for i in range(len(kpis.index)):
    try:
        expected_relationship=wt_types_dymb.at[kpis["Wind turbine type"].iloc[i],"Dynamic yaw misalignment"]
        expected_relationship=pd.DataFrame(expected_relationship)
        expected_relationship=expected_relationship.dropna()
        tmp=kpis["Filtered data"].iloc[i]
        tmp=pd.DataFrame(tmp)
        tmp=tmp[["Wind speed","Yaw misalignment"]].dropna()
        bin_value=0.5
        binnedtable=pd.DataFrame()
        tempnewdf=tmp.copy()
        tempnewdf["bin"]=(tempnewdf["Wind speed"]-(bin_value/2))/bin_value
        tempnewdf["bin"]=tempnewdf["bin"].astype("int64")
        ultratempone=tempnewdf[["bin","Wind speed"]]
        ultratemptwo=tempnewdf[["bin","Yaw misalignment"]]
        tempbinnedtable1=ultratempone.groupby(["bin"]).mean()
        tempbinnedtable2=ultratemptwo.groupby(["bin"]).apply(median_angle)
        tempnewdf2=pd.concat([tempbinnedtable1, tempbinnedtable2[["Yaw misalignment"]]], axis=1)
        tempnewdf2=tempnewdf2.rename(columns={"Yaw misalignment":'Avg'})
        tempbinnedtable3=ultratempone.groupby(["bin"]).mean()
        tempbinnedtable4=ultratemptwo.groupby(["bin"]).std()
        tempnewdf3 = pd.concat([tempbinnedtable3,tempbinnedtable4], axis=1)
        tempnewdf3=tempnewdf3.rename(columns={"Yaw misalignment":'Stdev'})
        tempnewdf4 = pd.concat([tempnewdf2,tempnewdf3], axis=1)
        tempbinnedtable5=ultratempone.groupby(["bin"]).mean()
        tempbinnedtable6=ultratemptwo.groupby(["bin"]).count()
        tempnewdf5 = pd.concat([tempbinnedtable5,tempbinnedtable6], axis=1)
        tempnewdf5=tempnewdf5.rename(columns={"Yaw misalignment":'Count'})
        tempnewdf6 = pd.concat([tempnewdf4,tempnewdf5], axis=1)
        tempnewdf6=tempnewdf6.loc[tempnewdf6["Count"]>25]
        tempnewdf4=tempnewdf6.drop(columns=["Count"])
        tempnewdf4 = tempnewdf4.loc[:,~tempnewdf4.columns.duplicated()]
        tempnewdf4.index=tempnewdf4["Wind speed"]
        if tempnewdf4.empty==False:
            steps=np.around(np.arange(0,tempnewdf["Wind speed"].max(),0.1),1).tolist()
            steps_tmp=pd.DataFrame(index=steps,columns=tempnewdf4.columns)
            tempnewdf4=tempnewdf4._append(steps_tmp)
            tempnewdf4.sort_index(inplace=True)
            tempnewdf4=tempnewdf4.interpolate(method="index")
            tempnewdf4=tempnewdf4.loc[steps]
            tempnewdf4=tempnewdf4.dropna()
            tempnewdf4=tempnewdf4.loc[~tempnewdf4.index.duplicated(keep='first')]
        binnedtable = tempnewdf4[["Avg","Stdev"]]
        binnedtable.sort_index(inplace=True)
        tmp["HiLim"]=tmp.apply(lambda row: binnedtable.at[round(row["Wind speed"],1),"Avg"]+1.96*binnedtable.at[round(row["Wind speed"],1),"Stdev"] if (round(row["Wind speed"],1) in binnedtable.index and pd.isnull(row["Wind speed"])==False) else np.nan, axis=1)
        tmp["LoLim"]=tmp.apply(lambda row: binnedtable.at[round(row["Wind speed"],1),"Avg"]-1.96*binnedtable.at[round(row["Wind speed"],1),"Stdev"] if (round(row["Wind speed"],1) in binnedtable.index and pd.isnull(row["Wind speed"])==False) else np.nan, axis=1)
        tmp["Filtered yaw misalignment"]=tmp.apply(lambda row: row["Yaw misalignment"] if (row["Yaw misalignment"]>row["LoLim"] and row["Yaw misalignment"]<row["HiLim"] ) else np.nan, axis=1)
        tmp=tmp[["Wind speed","Filtered yaw misalignment"]].dropna()
        tmp["flag"]=tmp.apply(lambda row: 1 if (round(row["Wind speed"],1) in expected_relationship.index and (row["Filtered yaw misalignment"]>expected_relationship.at[round(row["Wind speed"],1),"Max"] or row["Filtered yaw misalignment"]<expected_relationship.at[round(row["Wind speed"],1),"Min"])) else 0,axis=1)
        if tmp["flag"].sum()/tmp["flag"].count()>0.1:
            kpis["Dynamic yaw misalignment KPI"].iloc[i]='🔴'
            kpis["Dynamic yaw misalignment KPI Color"].iloc[i]='red'
        elif tmp["flag"].sum()/tmp["flag"].count()<=0.05:
            kpis["Dynamic yaw misalignment KPI"].iloc[i]='🟢'
            kpis["Dynamic yaw misalignment KPI Color"].iloc[i]='green'
        elif tmp["flag"].sum()/tmp["flag"].count()>0.05 and tmp["flag"].sum()/tmp["flag"].count()<=0.1:
            kpis["Dynamic yaw misalignment KPI"].iloc[i]='🟠'
            kpis["Dynamic yaw misalignment KPI Color"].iloc[i]='orange'
    except:
        kpis["Dynamic yaw misalignment KPI"].iloc[i]='⚫'
        kpis["Dynamic yaw misalignment KPI Color"].iloc[i]='black'

      




current_kpis=pd.DataFrame(index=sorted(list(set(kpis["Wind turbine"].tolist()))),columns=["Manufacturer","Type","Asset Manager","SCADA wind direction KPI","Ambient temperature KPI","Static yaw misalignment KPI","Dynamic yaw misalignment KPI","Pitch vs Power KPI","Pitch vs RPM KPI","RPM vs Power KPI","Power vs RPM KPI","Power vs Speed KPI","Pitch vs Speed KPI","RPM vs Speed KPI","Aerodynamic rotor imbalance KPI","Mass rotor imbalance KPI","Global rotor imbalance KPI","Front bearing temperature KPI","Rear bearing temperature KPI","Main bearing temperature KPI","Gearbox HSS bearing temperature KPI","Gearbox IMS/LSS bearing temperature KPI","Metal particle count KPI","Gearbox oil temperature KPI","Generator bearing front temperature KPI","Generator bearing rear temperature KPI","Rotor temperature KPI","Stator temperature KPI"])
current_kpis["Wind turbine"]=current_kpis.index
current_kpis["Wind farm"]=current_kpis["Wind turbine"].apply(lambda x: wt_info_frame.loc[wt_info_frame['Wind turbine']==x]['Wind farm'].iloc[0])
current_kpis["Country"]=current_kpis["Wind farm"].apply(lambda x: wf_coordinates_df.at[x,"Country"] if x in wf_coordinates_df.index else "Unknown")
current_kpis=current_kpis.sort_values(by=['Country',"Wind farm","Wind turbine"],key=natsort_keygen())

current_kpis.index=range(len(current_kpis.index))


for i in range(len(current_kpis.index)):
    wt=current_kpis["Wind turbine"].iloc[i]
    current_kpis["Manufacturer"].iloc[i]=kpis.loc[kpis["Wind turbine"]==wt]["Manufacturer"].iloc[0]
    current_kpis["Type"].iloc[i]=kpis.loc[kpis["Wind turbine"]==wt]["Wind turbine type"].iloc[0]
    current_kpis["Asset Manager"].iloc[i]=kpis.loc[kpis["Wind turbine"]==wt]["Asset Manager"].iloc[0]
    tmp=kpis.loc[(kpis["Wind turbine"]==wt) & (kpis["Month"]==(dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)).strftime('%B %Y'))]
    if tmp.shape[0]>0:
        current_kpis["SCADA wind direction KPI"].iloc[i]=tmp["SCADA wind direction KPI"].iloc[0]
        current_kpis["Ambient temperature KPI"].iloc[i]=tmp["Ambient temperature KPI"].iloc[0]
        current_kpis["Static yaw misalignment KPI"].iloc[i]=tmp["Static yaw misalignment KPI"].iloc[0]
        current_kpis["Dynamic yaw misalignment KPI"].iloc[i]=tmp["Dynamic yaw misalignment KPI"].iloc[0]
        current_kpis["Pitch vs Power KPI"].iloc[i]=tmp["Pitch vs Power KPI"].iloc[0]
        current_kpis["Pitch vs RPM KPI"].iloc[i]=tmp["Pitch vs RPM KPI"].iloc[0]
        current_kpis["RPM vs Power KPI"].iloc[i]=tmp["RPM vs Power KPI"].iloc[0]
        current_kpis["Power vs RPM KPI"].iloc[i]=tmp["Power vs RPM KPI"].iloc[0]
        current_kpis["Power vs Speed KPI"].iloc[i]=tmp["Power vs Speed KPI"].iloc[0]
        current_kpis["Pitch vs Speed KPI"].iloc[i]=tmp["Pitch vs Speed KPI"].iloc[0]
        current_kpis["RPM vs Speed KPI"].iloc[i]=tmp["RPM vs Speed KPI"].iloc[0]
        current_kpis["Aerodynamic rotor imbalance KPI"].iloc[i]=tmp["Aerodynamic rotor imbalance KPI"].iloc[0]
        current_kpis["Mass rotor imbalance KPI"].iloc[i]=tmp["Mass rotor imbalance KPI"].iloc[0]
        current_kpis["Global rotor imbalance KPI"].iloc[i]=tmp["Global rotor imbalance KPI"].iloc[0]
        current_kpis["Front bearing temperature KPI"].iloc[i]=tmp["Front bearing temperature KPI"].iloc[0]
        current_kpis["Rear bearing temperature KPI"].iloc[i]=tmp["Rear bearing temperature KPI"].iloc[0]
        current_kpis["Rotor temperature KPI"].iloc[i]=tmp["Rotor temperature KPI"].iloc[0]
        current_kpis["Stator temperature KPI"].iloc[i]=tmp["Stator temperature KPI"].iloc[0]
        current_kpis["Gearbox HSS bearing temperature KPI"].iloc[i]=tmp["Gearbox HSS bearing temperature KPI"].iloc[0]
        current_kpis["Gearbox IMS/LSS bearing temperature KPI"].iloc[i]=tmp["Gearbox IMS/LSS bearing temperature KPI"].iloc[0]
        current_kpis["Generator bearing front temperature KPI"].iloc[i]=tmp["Generator bearing front temperature KPI"].iloc[0]
        current_kpis["Generator bearing rear temperature KPI"].iloc[i]=tmp["Generator bearing rear temperature KPI"].iloc[0]
        current_kpis["Main bearing temperature KPI"].iloc[i]=tmp["Main bearing temperature KPI"].iloc[0]
        current_kpis["Metal particle count KPI"].iloc[i]=tmp["Metal particle count KPI"].iloc[0]
        current_kpis["Gearbox oil temperature KPI"].iloc[i]=tmp["Gearbox oil temperature KPI"].iloc[0]

    else:
        current_kpis["SCADA wind direction KPI"].iloc[i]='⚫'
        current_kpis["Ambient temperature KPI"].iloc[i]='⚫'
        current_kpis["Static yaw misalignment KPI"].iloc[i]='⚫'
        current_kpis["Dynamic yaw misalignment KPI"].iloc[i]='⚫'
        current_kpis["Pitch vs Power KPI"].iloc[i]='⚫'
        current_kpis["Pitch vs RPM KPI"].iloc[i]='⚫'
        current_kpis["RPM vs Power KPI"].iloc[i]='⚫'
        current_kpis["Power vs RPM KPI"].iloc[i]='⚫'
        current_kpis["Power vs Speed KPI"].iloc[i]='⚫'
        current_kpis["Pitch vs Speed KPI"].iloc[i]='⚫'
        current_kpis["RPM vs Speed KPI"].iloc[i]='⚫'
        current_kpis["Aerodynamic rotor imbalance KPI"].iloc[i]='⚫'
        current_kpis["Mass rotor imbalance KPI"].iloc[i]='⚫'
        current_kpis["Global rotor imbalance KPI"].iloc[i]='⚫'
        current_kpis["Front bearing temperature KPI"].iloc[i]='⚫'
        current_kpis["Rear bearing temperature KPI"].iloc[i]='⚫'
        current_kpis["Rotor temperature KPI"].iloc[i]='⚫'
        current_kpis["Stator temperature KPI"].iloc[i]='⚫'
        current_kpis["Gearbox HSS bearing temperature KPI"].iloc[i]='⚫'
        current_kpis["Gearbox IMS/LSS bearing temperature KPI"].iloc[i]='⚫'
        current_kpis["Generator bearing front temperature KPI"].iloc[i]='⚫'
        current_kpis["Generator bearing rear temperature KPI"].iloc[i]='⚫'
        current_kpis["Main bearing temperature KPI"].iloc[i]='⚫'
        current_kpis["Metal particle count KPI"].iloc[i]='⚫'
        current_kpis["Gearbox oil temperature KPI"].iloc[i]='⚫'

current_kpis=current_kpis[["Country","Wind farm","Wind turbine","Manufacturer","Type","Asset Manager","SCADA wind direction KPI","Ambient temperature KPI","Static yaw misalignment KPI","Dynamic yaw misalignment KPI","Pitch vs Power KPI","Pitch vs RPM KPI","RPM vs Power KPI","Power vs RPM KPI","Power vs Speed KPI","Pitch vs Speed KPI","RPM vs Speed KPI","Aerodynamic rotor imbalance KPI","Mass rotor imbalance KPI","Global rotor imbalance KPI","Front bearing temperature KPI","Rear bearing temperature KPI","Main bearing temperature KPI","Gearbox HSS bearing temperature KPI","Gearbox IMS/LSS bearing temperature KPI","Metal particle count KPI","Gearbox oil temperature KPI","Generator bearing front temperature KPI","Generator bearing rear temperature KPI","Rotor temperature KPI","Stator temperature KPI"]]
current_kpis["id"]=range(len(current_kpis.index))

print("["+dt.today().strftime('%d/%m/%Y %H:%M')+"] [INFO] Successfully computed performance KPIs")    





kpis["Unclassified curtailment periods"]=kpis["Unclassified curtailment periods"].apply(lambda x: x.replace("nan",'np.nan'))
kpis["Unclassified curtailment periods"]=kpis["Unclassified curtailment periods"].astype(str)
kpis["Unclassified curtailment periods"]=kpis["Unclassified curtailment periods"].apply(lambda x: eval(x))

kpis["Unclassified curtailments KPI"]=np.nan
kpis["Unclassified curtailments KPI"]='🟢'
kpis["Unclassified curtailments KPI Colors"]='green'
for i in range(len(kpis.index)):
    try:
        tmp=kpis["Unclassified curtailment periods"].iloc[i]
        tmp=pd.DataFrame(tmp)
        tmp["Start"]=tmp["Start"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
        tmp["End"]=tmp["End"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
        tmp["Duration"]=tmp["End"]-tmp["Start"]
        tmp["Duration"]=tmp["Duration"].apply(lambda x: x.total_seconds())
        total_duration=tmp["Duration"].sum()/60/60
        if total_duration>24:
            kpis["Unclassified curtailments KPI"].iloc[i]='🔴'
            kpis["Unclassified curtailments KPI Colors"].iloc[i]='red'
        elif total_duration>0 and total_duration<=24:
            kpis["Unclassified curtailments KPI"].iloc[i]='🟠'
            kpis["Unclassified curtailments KPI Colors"].iloc[i]='orange'
        else:
            kpis["Unclassified curtailments KPI"].iloc[i]='🟢'
            kpis["Unclassified curtailments KPI Colors"].iloc[i]='green'
            
    except:
        pass


current_availability_kpis=pd.DataFrame(index=sorted(list(set(availability_kpis["Wind turbine"].tolist()))),columns=["Monthly time-based System Avail. KPI","Monthly production-based System Avail. KPI","Monthly Year-To-Date time-based System Avail. KPI","Monthly Year-To-Date production-based System Avail. KPI","Monthly own stops KPI","Monthly scheduled maintenance stops KPI","Monthly Year-To-Date own stops KPI","Monthly Year-To-Date scheduled maintenance stops KPI","Status Codes KPI","Warnings KPI","Unclassified curtailments KPI"])
current_availability_kpis["Wind turbine"]=current_availability_kpis.index
current_availability_kpis["Wind farm"]=current_availability_kpis["Wind turbine"].apply(lambda x: wt_info_frame.loc[wt_info_frame['Wind turbine']==x]['Wind farm'].iloc[0])
current_availability_kpis["Country"]=current_availability_kpis["Wind farm"].apply(lambda x: wf_coordinates_df.at[x,"Country"] if x in wf_coordinates_df.index else "Unknown")
current_availability_kpis["Asset Manager"]=current_availability_kpis["Wind farm"].apply(lambda x: metadata.at[x,"AM"])


current_availability_kpis["Manufacturer"]=current_availability_kpis["Wind turbine"].apply(lambda x: wt_info_frame.loc[wt_info_frame['Wind turbine']==x]['Manufacturer'].iloc[0])
current_availability_kpis.index=range(len(current_availability_kpis.index))
for i in range(len(current_availability_kpis.index)):
    wt=current_availability_kpis["Wind turbine"].iloc[i]
    tmp=availability_kpis.loc[(availability_kpis["Wind turbine"]==wt) & (availability_kpis["Date"]==(dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)).strftime('%B %Y'))]
    if tmp.shape[0]>0:
        current_availability_kpis["Monthly time-based System Avail. KPI"].iloc[i]=tmp["Time-based System Avail. KPI"].iloc[0]
        current_availability_kpis["Monthly production-based System Avail. KPI"].iloc[i]=tmp["Production-based System Avail. KPI"].iloc[0]
        current_availability_kpis["Monthly own stops KPI"].iloc[i]=tmp["Monthly own stops KPI"].iloc[0]
        current_availability_kpis["Monthly scheduled maintenance stops KPI"].iloc[i]=tmp["Monthly scheduled maintenance stops KPI"].iloc[0]
        current_availability_kpis["Status Codes KPI"].iloc[i]=tmp["Status Codes KPI"].iloc[0]
        current_availability_kpis["Warnings KPI"].iloc[i]=tmp["Warnings KPI"].iloc[0]
    else:
        current_availability_kpis["Monthly time-based System Avail. KPI"].iloc[i]='⚫'
        current_availability_kpis["Monthly production-based System Avail. KPI"].iloc[i]='⚫'
        current_availability_kpis["Monthly own stops KPI"].iloc[i]='⚫'
        current_availability_kpis["Monthly scheduled maintenance stops KPI"].iloc[i]='⚫'
        current_availability_kpis["Status Codes KPI"].iloc[i]='⚫'
        current_availability_kpis["Warnings KPI"].iloc[i]='⚫'
    tmp=restructured_ytd_availability_kpis.loc[(restructured_ytd_availability_kpis["Wind turbine"]==wt) & (restructured_ytd_availability_kpis["Date"]==(dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)).strftime('%B %Y'))]
    if tmp.shape[0]>0:
        current_availability_kpis["Monthly Year-To-Date time-based System Avail. KPI"].iloc[i]=tmp["Time-based System Avail. KPI"].iloc[0]
        current_availability_kpis["Monthly Year-To-Date production-based System Avail. KPI"].iloc[i]=tmp["Production-based System Avail. KPI"].iloc[0]
        current_availability_kpis["Monthly Year-To-Date own stops KPI"].iloc[i]=tmp["Monthly own stops KPI"].iloc[0]
        current_availability_kpis["Monthly Year-To-Date scheduled maintenance stops KPI"].iloc[i]=tmp["Monthly scheduled maintenance stops KPI"].iloc[0]
    else:
        current_availability_kpis["Monthly Year-To-Date time-based System Avail. KPI"].iloc[i]='⚫'
        current_availability_kpis["Monthly Year-To-Date production-based System Avail. KPI"].iloc[i]='⚫'
        current_availability_kpis["Monthly Year-To-Date own stops KPI"].iloc[i]='⚫'
        current_availability_kpis["Monthly Year-To-Date scheduled maintenance stops KPI"].iloc[i]='⚫'



for i in range(len(current_availability_kpis.index)):
    wt=current_availability_kpis["Wind turbine"].iloc[i]
    tmp=kpis.loc[(kpis["Wind turbine"]==wt) & (kpis["Month"]==(dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)).strftime('%B %Y'))]
    if tmp.shape[0]>0:
        current_availability_kpis["Unclassified curtailments KPI"].iloc[i]=tmp["Unclassified curtailments KPI"].iloc[0]
    else:
        current_availability_kpis["Unclassified curtailments KPI"].iloc[i]='⚫'



current_availability_kpis=current_availability_kpis[['Country',"Wind farm","Wind turbine","Manufacturer","Asset Manager","Monthly time-based System Avail. KPI","Monthly production-based System Avail. KPI","Monthly Year-To-Date time-based System Avail. KPI","Monthly Year-To-Date production-based System Avail. KPI","Monthly own stops KPI","Monthly scheduled maintenance stops KPI","Monthly Year-To-Date own stops KPI","Monthly Year-To-Date scheduled maintenance stops KPI","Status Codes KPI","Warnings KPI","Unclassified curtailments KPI"]]
#current_availability_kpis=current_availability_kpis.sort_values(by=['Country',"Wind farm"])
current_availability_kpis=current_availability_kpis.sort_values(by=['Country',"Wind farm","Wind turbine"],key=natsort_keygen())

current_availability_kpis["id"]=range(len(current_availability_kpis.index))


























with open('WindIndexes.csv') as f:
    resource_kpis = pd.read_csv(f,sep=',',header=0,names=["Wind farm","Wind turbine","Wind Indexes","WTLongitude","WTLatitude"])
resource_kpis.index=resource_kpis["Wind farm"]
resource_kpis=resource_kpis.loc[~resource_kpis.index.duplicated(keep='first')]
resource_kpis.index=range(len(resource_kpis.index))

resource_kpis["Wind Indexes"] = resource_kpis["Wind Indexes"].astype(str)
resource_kpis["Wind Indexes"]=resource_kpis["Wind Indexes"].apply(lambda x: ast.literal_eval(x))
resource_kpis["Wind Indexes"]=resource_kpis["Wind Indexes"].apply(lambda x: x["Wind Indexes"])

resource_kpis["Country"]=resource_kpis["Wind farm"].apply(lambda x: wf_coordinates_df.at[x,"Country"] if x in wf_coordinates_df.index else "Unknown")

resource_indexes=resource_kpis["Wind Indexes"].iloc[0]
resource_indexes=[dt.strptime(k,'%d/%m/%Y %H:%M') for k, v in resource_indexes.items()]
first_index=min(resource_indexes)
last_index=max(resource_indexes)



end_date=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)
production_months=pd.period_range(start_date,end_date,freq='1M').to_timestamp().tolist()

months=pd.period_range(first_index,end_date,freq='1M').to_timestamp().tolist()
resource_kpis["Wind Indexes"]=resource_kpis["Wind Indexes"].apply(lambda x: {dt.strptime(el,'%d/%m/%Y %H:%M'):x[el] for el in x})
resource_kpis["Curent Monthly Wind Index"]=resource_kpis["Wind Indexes"].apply(lambda x: x[end_date] if end_date in x.keys() else np.nan)
resource_kpis["Monthly P50s"]=resource_kpis["Wind Indexes"].apply(lambda x: {el:np.nanmean([x[k] for k in x if k.month==el.month]) for el in months})
resource_kpis["Curent Monthly P50"]=resource_kpis["Monthly P50s"].apply(lambda x: x[end_date] if end_date in x.keys() else np.nan)
resource_kpis["Monthly P75s"]=resource_kpis["Wind Indexes"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])-0.675*np.nanstd([x[k] for k in x if k.month==el.month])) for el in months})
resource_kpis["Monthly P25s"]=resource_kpis["Wind Indexes"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])+0.675*np.nanstd([x[k] for k in x if k.month==el.month])) for el in months})
resource_kpis["Monthly P90s"]=resource_kpis["Wind Indexes"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])-1.282*np.nanstd([x[k] for k in x if k.month==el.month])) for el in months})
resource_kpis["Monthly P10s"]=resource_kpis["Wind Indexes"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])+1.282*np.nanstd([x[k] for k in x if k.month==el.month])) for el in months})

resource_kpis["Monthly Wind Resource KPI"]=resource_kpis.apply(lambda row: '⚫' if end_date not in row["Wind Indexes"].keys() else '🟢'+" (+"+str(round(100*((row["Wind Indexes"][end_date]-row["Monthly P50s"][end_date])/row["Monthly P50s"][end_date]),1))+"%)",axis=1)
resource_kpis["Monthly Wind Resource KPI"]=resource_kpis.apply(lambda row: '🟠'+" ("+str(round(100*((row["Wind Indexes"][end_date]-row["Monthly P50s"][end_date])/row["Monthly P50s"][end_date]),1))+"%)" if (end_date in row["Wind Indexes"].keys()) and (row["Wind Indexes"][end_date]>=row["Monthly P75s"][end_date]) and (row["Wind Indexes"][end_date]<row["Monthly P50s"][end_date]) else row["Monthly Wind Resource KPI"],axis=1)
resource_kpis["Monthly Wind Resource KPI"]=resource_kpis.apply(lambda row: '🔴'+" ("+str(round(100*((row["Wind Indexes"][end_date]-row["Monthly P50s"][end_date])/row["Monthly P50s"][end_date]),1))+"%)" if (end_date in row["Wind Indexes"].keys()) and (row["Wind Indexes"][end_date]<row["Monthly P75s"][end_date]) else row["Monthly Wind Resource KPI"],axis=1)

resource_kpis["Monthly WI Graph Colors"]=resource_kpis.apply(lambda row: {el:'green' if ((el in row["Wind Indexes"].keys()) and (row["Wind Indexes"][el]>=row["Monthly P50s"][el])) else ('red' if ((el in row["Wind Indexes"].keys()) and (row["Wind Indexes"][el]<row["Monthly P75s"][el])) else ("orange" if el in row["Wind Indexes"].keys() else ("black"))) for el in months},axis=1)




resource_kpis["Monthly YTD Wind Indexes"]=resource_kpis["Wind Indexes"].apply(lambda x: {el:np.nanmean([x[k] for k in x if k.year==el.year and k.month<=el.month]) for el in x})
resource_kpis["Wind Indexes for YTD stats"]=resource_kpis["Wind Indexes"]
years_in=[k.year for k in resource_kpis["Wind Indexes for YTD stats"].iloc[0]]
min_year_in=min(years_in)
first_year_months_count=len([p for p in years_in if p == min_year_in])
max_year_in=max(years_in)
#last_year_months_count=len([p for p in years_in if p == max_year_in])
resource_kpis["Curent Monthly YTD Wind Index"]=resource_kpis["Wind Indexes"].apply(lambda x: np.nanmean([x[k] for k in x if k.year==max_year_in]))
if first_year_months_count<12:
    resource_kpis["Wind Indexes for YTD stats"]=resource_kpis["Wind Indexes for YTD stats"].apply(lambda x: {el:x[el] for el in x if el.year!=min_year_in})
    resource_kpis["Monthly YTD Wind Indexes"]=resource_kpis["Monthly YTD Wind Indexes"].apply(lambda x: {el:x[el] for el in x if el.year!=min_year_in})
    months=[el for el in months if el.year!=min_year_in]
#if last_year_months_count<12:
#    resource_kpis["Wind Indexes for YTD stats"]=resource_kpis["Wind Indexes for YTD stats"].apply(lambda x: {el:x[el] for el in x if el.year!=max_year_in})

resource_kpis["Monthly YTD P50s"]=resource_kpis["Wind Indexes for YTD stats"].apply(lambda x: {el:np.nanmean([x[k] for k in x if k.year==el.year and k.month<=el.month]) for el in months})
resource_kpis["Monthly YTD P75s"]=resource_kpis["Monthly YTD P50s"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])-0.675*np.nanstd([x[k] for k in x if k.month==el.month])) for el in months})
resource_kpis["Monthly YTD P25s"]=resource_kpis["Monthly YTD P50s"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])+0.675*np.nanstd([x[k] for k in x if k.month==el.month])) for el in months})
resource_kpis["Monthly YTD P90s"]=resource_kpis["Monthly YTD P50s"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])-1.282*np.nanstd([x[k] for k in x if k.month==el.month])) for el in months})
resource_kpis["Monthly YTD P10s"]=resource_kpis["Monthly YTD P50s"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])+1.282*np.nanstd([x[k] for k in x if k.month==el.month])) for el in months})
resource_kpis["Monthly YTD P50s"]=resource_kpis["Monthly YTD P50s"].apply(lambda x: {el:np.nanmean([x[k] for k in x if k.month==el.month]) for el in months})
resource_kpis["Curent Monthly YTD P50"]=resource_kpis["Monthly YTD P50s"].apply(lambda x: x[last_index])

resource_kpis["Monthly Year-To-Date Wind Resource KPI"]=resource_kpis.apply(lambda row: '⚫' if end_date not in row["Monthly YTD Wind Indexes"].keys() else '🟢'+" (+"+str(round(100*((row["Monthly YTD Wind Indexes"][end_date]-row["Monthly YTD P50s"][end_date])/row["Monthly YTD P50s"][end_date]),1))+"%)",axis=1)
resource_kpis["Monthly Year-To-Date Wind Resource KPI"]=resource_kpis.apply(lambda row: '🟠'+" ("+str(round(100*((row["Monthly YTD Wind Indexes"][end_date]-row["Monthly YTD P50s"][end_date])/row["Monthly YTD P50s"][end_date]),1))+"%)" if (end_date in row["Monthly YTD Wind Indexes"].keys()) and (row["Monthly YTD Wind Indexes"][end_date]>=row["Monthly YTD P75s"][end_date]) and (row["Monthly YTD Wind Indexes"][end_date]<row["Monthly YTD P50s"][end_date]) else row["Monthly Year-To-Date Wind Resource KPI"],axis=1)
resource_kpis["Monthly Year-To-Date Wind Resource KPI"]=resource_kpis.apply(lambda row: '🔴'+" ("+str(round(100*((row["Monthly YTD Wind Indexes"][end_date]-row["Monthly YTD P50s"][end_date])/row["Monthly YTD P50s"][end_date]),1))+"%)" if (end_date in row["Monthly YTD Wind Indexes"].keys()) and (row["Monthly YTD Wind Indexes"][end_date]<row["Monthly YTD P75s"][end_date]) else row["Monthly Year-To-Date Wind Resource KPI"],axis=1)

resource_kpis["Monthly YTD WI Graph Colors"]=resource_kpis.apply(lambda row: {el:'green' if ((el in row["Monthly YTD Wind Indexes"].keys()) and (row["Monthly YTD Wind Indexes"][el]>=row["Monthly YTD P50s"][el])) else ('red' if ((el in row["Monthly YTD Wind Indexes"].keys()) and (row["Monthly YTD Wind Indexes"][el]<row["Monthly YTD P75s"][el])) else ("orange" if el in row["Monthly YTD Wind Indexes"].keys() else ("black"))) for el in months},axis=1)

#resource_kpis=resource_kpis.sort_values(by=['Country', 'Wind farm'])
resource_kpis=resource_kpis.sort_values(by=['Country',"Wind farm"],key=natsort_keygen())





with open('ClusteredWindIndexes.csv') as f:
    clustered_resource_kpis = pd.read_csv(f,sep=',',header=0,names=["WTLongitude","WTLatitude","Wind Indexes","Cluster"])
clustered_resource_kpis["Wind Indexes"] = clustered_resource_kpis["Wind Indexes"].astype(str)
clustered_resource_kpis["Wind Indexes"]=clustered_resource_kpis["Wind Indexes"].apply(lambda x: ast.literal_eval(x))

clustered_resource_indexes=clustered_resource_kpis["Wind Indexes"].iloc[0]
clustered_resource_indexes=[dt.strptime(k,'%d/%m/%Y %H:%M') for k, v in clustered_resource_indexes.items()]
clustered_first_index=min(clustered_resource_indexes)
clustered_last_index=max(clustered_resource_indexes)
clustered_last_closed_month=clustered_last_index.month

clustered_months=pd.period_range(clustered_first_index,end_date,freq='1M').to_timestamp().tolist()
clustered_resource_kpis["Wind Indexes"]=clustered_resource_kpis["Wind Indexes"].apply(lambda x: {dt.strptime(el,'%d/%m/%Y %H:%M'):x[el] for el in x})
clustered_resource_kpis["Curent Monthly Wind Index"]=clustered_resource_kpis["Wind Indexes"].apply(lambda x: x[clustered_last_index])
clustered_resource_kpis["Monthly P50s"]=clustered_resource_kpis["Wind Indexes"].apply(lambda x: {el:np.nanmean([x[k] for k in x if k.month==el.month]) for el in clustered_months})
clustered_resource_kpis["Curent Monthly P50"]=clustered_resource_kpis["Monthly P50s"].apply(lambda x: x[clustered_last_index])
clustered_resource_kpis["Monthly P75s"]=clustered_resource_kpis["Wind Indexes"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])-0.675*np.nanstd([x[k] for k in x if k.month==el.month])) for el in clustered_months})
clustered_resource_kpis["Monthly P25s"]=clustered_resource_kpis["Wind Indexes"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])+0.675*np.nanstd([x[k] for k in x if k.month==el.month])) for el in clustered_months})
clustered_resource_kpis["Monthly P90s"]=clustered_resource_kpis["Wind Indexes"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])-1.282*np.nanstd([x[k] for k in x if k.month==el.month])) for el in clustered_months})
clustered_resource_kpis["Monthly P10s"]=clustered_resource_kpis["Wind Indexes"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])+1.282*np.nanstd([x[k] for k in x if k.month==el.month])) for el in clustered_months})
clustered_resource_kpis["Monthly Wind Resource KPI"]=clustered_resource_kpis.apply(lambda row: '🔴' if row["Wind Indexes"][clustered_last_index]<row["Monthly P75s"][clustered_last_index] else '🟠',axis=1)
clustered_resource_kpis["Monthly Wind Resource KPI"]=clustered_resource_kpis.apply(lambda row: '🟢' if row["Wind Indexes"][clustered_last_index]>=row["Monthly P50s"][clustered_last_index] else row["Monthly Wind Resource KPI"],axis=1)

clustered_resource_kpis["Monthly WI Graph Colors"]=clustered_resource_kpis.apply(lambda row: ['green' if (el in row["Wind Indexes"].keys() and row["Wind Indexes"][el]>=row["Monthly P50s"][el]) else ('red' if (el in row["Wind Indexes"].keys() and row["Wind Indexes"][el]<row["Monthly P75s"][el]) else ("orange" if el in row["Wind Indexes"].keys() else ("black"))) for el in clustered_months],axis=1)
clustered_resource_kpis["Monthly WI Graph Colors"]=clustered_resource_kpis["Monthly WI Graph Colors"].apply(lambda x: {clustered_months[i]:x[i] if i<=len(x) else np.nan for i in range(len(clustered_months))})
clustered_resource_kpis["Monthly YTD Wind Indexes"]=clustered_resource_kpis["Wind Indexes"].apply(lambda x: {el:np.nanmean([x[k] for k in x if k.year==el.year and k.month<=el.month]) for el in clustered_months})
clustered_resource_kpis["Wind Indexes for YTD stats"]=clustered_resource_kpis["Wind Indexes"]
clustered_years_in=[k.year for k in clustered_resource_kpis["Wind Indexes for YTD stats"].iloc[0]]
clustered_min_year_in=min(clustered_years_in)
clustered_first_year_months_count=len([p for p in clustered_years_in if p == clustered_min_year_in])
clustered_max_year_in=max(clustered_years_in)
#last_year_months_count=len([p for p in years_in if p == max_year_in])
clustered_resource_kpis["Curent Monthly YTD Wind Index"]=clustered_resource_kpis["Wind Indexes"].apply(lambda x: np.nanmean([x[k] for k in x if k.year==clustered_max_year_in]))
if clustered_first_year_months_count<12:
    clustered_resource_kpis["Wind Indexes for YTD stats"]=clustered_resource_kpis["Wind Indexes for YTD stats"].apply(lambda x: {el:x[el] for el in x if el.year!=clustered_min_year_in})
    clustered_resource_kpis["Monthly YTD Wind Indexes"]=clustered_resource_kpis["Monthly YTD Wind Indexes"].apply(lambda x: {el:x[el] for el in x if el.year!=clustered_min_year_in})
    clustered_months=[el for el in clustered_months if el.year!=clustered_min_year_in]



clustered_resource_kpis["Monthly YTD P50s"]=clustered_resource_kpis["Wind Indexes for YTD stats"].apply(lambda x: {el:np.nanmean([x[k] for k in x if k.year==el.year and k.month<=el.month]) for el in clustered_months})
clustered_resource_kpis["Monthly YTD P75s"]=clustered_resource_kpis["Monthly YTD P50s"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])-0.675*np.nanstd([x[k] for k in x if k.month==el.month])) for el in clustered_months})
clustered_resource_kpis["Monthly YTD P25s"]=clustered_resource_kpis["Monthly YTD P50s"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])+0.675*np.nanstd([x[k] for k in x if k.month==el.month])) for el in clustered_months})
clustered_resource_kpis["Monthly YTD P90s"]=clustered_resource_kpis["Monthly YTD P50s"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])-1.282*np.nanstd([x[k] for k in x if k.month==el.month])) for el in clustered_months})
clustered_resource_kpis["Monthly YTD P10s"]=clustered_resource_kpis["Monthly YTD P50s"].apply(lambda x: {el:(np.nanmean([x[k] for k in x if k.month==el.month])+1.282*np.nanstd([x[k] for k in x if k.month==el.month])) for el in clustered_months})
clustered_resource_kpis["Monthly YTD P50s"]=clustered_resource_kpis["Monthly YTD P50s"].apply(lambda x: {el:np.nanmean([x[k] for k in x if k.month==el.month]) for el in clustered_months})
clustered_resource_kpis["Curent Monthly YTD P50"]=clustered_resource_kpis["Monthly YTD P50s"].apply(lambda x: x[clustered_last_index])
clustered_resource_kpis["Monthly Year-To-Date Wind Resource KPI"]=clustered_resource_kpis.apply(lambda row: '🔴' if row["Monthly YTD Wind Indexes"][clustered_last_index]<row["Monthly YTD P75s"][clustered_last_index] else '🟠',axis=1)
clustered_resource_kpis["Monthly Year-To-Date Wind Resource KPI"]=clustered_resource_kpis.apply(lambda row: '🟢' if row["Monthly YTD Wind Indexes"][clustered_last_index]>=row["Monthly YTD P50s"][clustered_last_index] else row["Monthly Year-To-Date Wind Resource KPI"],axis=1)
clustered_resource_kpis["Monthly YTD WI Graph Colors"]=clustered_resource_kpis.apply(lambda row: ['green' if row["Monthly YTD Wind Indexes"][el]>=row["Monthly YTD P50s"][el] else ('red' if row["Monthly YTD Wind Indexes"][el]<row["Monthly YTD P75s"][el] else "orange") for el in clustered_months],axis=1)
clustered_resource_kpis["Monthly YTD WI Graph Colors"]=clustered_resource_kpis["Monthly YTD WI Graph Colors"].apply(lambda x: {clustered_months[i]:x[i] for i in range(len(clustered_months))})


current_resource_kpis=resource_kpis[["Country","Wind farm","Monthly Wind Resource KPI","Monthly Year-To-Date Wind Resource KPI"]]
current_resource_kpis["id"]=range(len(current_resource_kpis.index))

print("["+dt.today().strftime('%d/%m/%Y %H:%M')+"] [INFO] Successfully computed wind resource KPIs")    


app = dash.Dash(__name__)

app.title = TITLE
server = app.server
app.config['suppress_callback_exceptions']=True
app.scripts.config.serve_locally=True



kpis_datatable_widget = html.Div([
         html.Br(),
         html.H6('Results displayed in below overview table are valid for '+end_date.strftime("%B %Y")+". If all KPIs are black, it means that results are not yet available. Performance KPIs usually get available around 7 days after month end",style={'color': '#404756','text-align': 'center'}),
        dcc.Clipboard(id="kpis_datatable_copy", style={"fontSize":20}),
                       dashtable.DataTable(
                               id="kpis-datatable",
                               style_table={'maxHeight': '600','overflowY': 'auto'},
                               style_cell={'textAlign': 'center','padding': '5px','width': '140px','whiteSpace': 'normal','height': 'auto'},
                               page_action="none",
                               data=current_kpis.to_dict('records'),
                               columns=[{"name": i, "id": i} for i in ["Country","Wind farm","Wind turbine","Manufacturer","Type","Asset Manager","SCADA wind direction KPI","Ambient temperature KPI","Static yaw misalignment KPI","Dynamic yaw misalignment KPI","Pitch vs Power KPI","Pitch vs RPM KPI","RPM vs Power KPI","Power vs RPM KPI","Power vs Speed KPI","Pitch vs Speed KPI","RPM vs Speed KPI"]],
                               fixed_rows={'headers':True},
                               filter_action="native",
                               sort_action='native',
                               style_header={'backgroundColor': 'white','fontWeight': 'bold'},
                               )
                       ],style={"margin": "10px"})


reliability_kpis_datatable_widget = html.Div([
         html.Br(),
         html.H6('Results displayed in below overview table are valid for '+end_date.strftime("%B %Y")+". If all KPIs are black, it means that results are not yet available. Performance KPIs usually get available around 7 days after month end",style={'color': '#404756','text-align': 'center'}),
        dcc.Clipboard(id="reliability_kpis_datatable_copy", style={"fontSize":20}),
                       dashtable.DataTable(
                               id="reliability-kpis-datatable",
                               style_table={'maxHeight': '600','overflowY': 'auto'},
                               style_cell={'textAlign': 'center','padding': '5px','width': '140px','whiteSpace': 'normal','height': 'auto'},
                               page_action="none",
                               data=current_kpis.to_dict('records'),
                               columns=[{"name": i, "id": i} for i in ["Country","Wind farm","Wind turbine","Manufacturer","Type","Asset Manager","Aerodynamic rotor imbalance KPI","Mass rotor imbalance KPI","Global rotor imbalance KPI","Front bearing temperature KPI","Rear bearing temperature KPI","Main bearing temperature KPI","Gearbox HSS bearing temperature KPI","Gearbox IMS/LSS bearing temperature KPI","Metal particle count KPI","Gearbox oil temperature KPI","Generator bearing front temperature KPI","Generator bearing rear temperature KPI","Rotor temperature KPI","Stator temperature KPI"]],
                               fixed_rows={'headers':True},
                               filter_action="native",
                               sort_action='native',
                               style_header={'backgroundColor': 'white','fontWeight': 'bold'},
                               )
                       ],style={"margin": "10px"})


resource_kpis_datatable_widget = html.Div([
         html.Br(),
         html.H6('Results displayed in below overview table are valid for '+end_date.strftime("%B %Y")+". If all KPIs are black, it means that results are not yet available. Wind resource KPIs usually get available around 7 days after month end",style={'color': '#404756','text-align': 'center'}),
        dcc.Clipboard(id="resource_kpis_datatable_copy", style={"fontSize":20}),
                       dashtable.DataTable(
                               id="resource-kpis-datatable",
                               style_table={'maxHeight': '600','overflowY': 'auto'},
                               style_cell={'textAlign': 'center','padding': '5px','width': '140px','whiteSpace': 'normal','height': 'auto'},
                               page_action="none",
                               data=current_resource_kpis.to_dict('records'),
                               columns=[{"name": i, "id": i} for i in ["Country","Wind farm","Monthly Wind Resource KPI","Monthly Year-To-Date Wind Resource KPI"]],
                               fixed_rows={'headers':True},
                               filter_action="native",
                               sort_action='native',
                               style_header={'backgroundColor': 'white','fontWeight': 'bold'},
                               )
                       ],style={"margin": "10px"})


availability_kpis_datatable_widget = html.Div([
         html.Br(),
         html.H6('Results displayed in below overview table are valid for '+end_date.strftime("%B %Y"),style={'color': '#404756','text-align': 'center'}),
        dcc.Clipboard(id="availability_kpis_datatable_copy", style={"fontSize":20}),
                       dashtable.DataTable(
                               id="availability-kpis-datatable",
                               style_table={'maxHeight': '600','overflowY': 'auto'},
                               style_cell={'textAlign': 'center','padding': '5px','width': '140px','whiteSpace': 'normal','height': 'auto'},
                               page_action="none",
                               data=current_availability_kpis.to_dict('records'),
                               columns=[{"name": i, "id": i} for i in ['Country',"Wind farm","Wind turbine","Manufacturer","Asset Manager","Monthly time-based System Avail. KPI","Monthly production-based System Avail. KPI","Monthly Year-To-Date time-based System Avail. KPI","Monthly Year-To-Date production-based System Avail. KPI","Monthly own stops KPI","Monthly scheduled maintenance stops KPI","Monthly Year-To-Date own stops KPI","Monthly Year-To-Date scheduled maintenance stops KPI","Status Codes KPI","Warnings KPI","Unclassified curtailments KPI"]],
                               fixed_rows={'headers':True},
                               filter_action="native",
                               sort_action='native',
                               style_header={'backgroundColor': 'white','fontWeight': 'bold'},
                               )
                       ],style={"margin": "10px"})


production_kpis_datatable_widget = html.Div([
         html.Br(),
         html.H6('Results displayed in below overview table are valid for '+end_date.strftime("%B %Y")+". If all KPIs are black, it means that results are not yet available. Production KPIs get available as soon as OCC has exported monthly net metered productions to Greenbyte",style={'color': '#404756','text-align': 'center'}),
        dcc.Clipboard(id="production_kpis_datatable_copy", style={"fontSize":20}),
                       dashtable.DataTable(
                               id="production-kpis-datatable",
                               style_table={'maxHeight': '600','overflowY': 'auto'},
                               style_cell={'textAlign': 'center','padding': '5px','width': '140px','whiteSpace': 'normal','height': 'auto'},
                               page_action="none",
                               data=current_production_kpis.to_dict('records'),
                               columns=[{"name": i, "id": i} for i in ["Country","Wind farm","Monthly production KPI","Monthly Year-To-Date production KPI"]],
                               fixed_rows={'headers':True},
                               filter_action="native",
                               sort_action='native',
                               style_header={'backgroundColor': 'white','fontWeight': 'bold'},
                               )
                       ],style={"margin": "10px"})




kpis_plot_widget = html.Div([
        html.Div(id="kpis-graph-container",children=[
        dcc.Graph(id="kpis-graph"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

kpis_plot_widget_2 = html.Div([
        html.Div(id="kpis-graph-2-container",children=[
        dcc.Graph(id="kpis-graph-2"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

kpis_plot_widget_3 = html.Div([
        html.Div(id="kpis-graph-3-container",children=[
        dcc.Graph(id="kpis-graph-3"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

kpis_plot_widget_4 = html.Div([
        html.Div(id="kpis-graph-4-container",children=[
        dcc.Graph(id="kpis-graph-4"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

kpis_plot_widget_5 = html.Div([
        html.Div(id="kpis-graph-5-container",children=[
        dcc.Graph(id="kpis-graph-5"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})


reliability_kpis_plot_widget = html.Div([
        html.Div(id="reliability-kpis-graph-container",children=[
        dcc.Graph(id="reliability-kpis-graph"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

reliability_kpis_plot_widget_2 = html.Div([
        html.Div(id="reliability-kpis-graph-2-container",children=[
        dcc.Graph(id="reliability-kpis-graph-2"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

    
    
resource_kpis_plot_widget = html.Div([
        html.Div(id="resource-kpis-graph-container",children=[
        dcc.Graph(id="resource-kpis-graph"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

resource_kpis_plot_widget_2 = html.Div([
        html.Div(id="resource-kpis-graph-2-container",children=[
        dcc.Graph(id="resource-kpis-graph-2"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

resource_kpis_plot_widget_3 = html.Div([
        html.Div(id="resource-kpis-graph-3-container",children=[
        dcc.Graph(id="resource-kpis-graph-3"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

    
availability_kpis_plot_widget = html.Div([
        html.Div(id="availability-kpis-graph-container",children=[
        dcc.Graph(id="availability-kpis-graph"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

availability_kpis_datatable_2_widget = html.Div([
        html.Div(id="period-container",children=[
                html.Div(id='period'),
                ],style={'display': 'none'}),
        html.Div(id="availability-kpis-datatable-2-container",children=[
#                html.H6(end_date.strftime("%B %Y")+" table",style={'color': '#404756','text-align': 'center'}),
                dcc.Clipboard(id="availability_kpis_datatable_2_copy", style={"fontSize":20}),
                       dashtable.DataTable(
                               id="availability-kpis-datatable-2",
                               style_table={'maxHeight': '600','overflowY': 'auto'},
                               style_cell={'textAlign': 'center','padding': '5px','width': '140px','whiteSpace': 'normal','height': 'auto'},
                               page_action="none",
                               fixed_rows={'headers':True},
                               filter_action="native",
                               sort_action='native',
                               style_header={'backgroundColor': 'white','fontWeight': 'bold'},
                               )
                       ],style={'display': 'none'})
                       ],style={"margin": "10px"})

availability_kpis_plot_widget_3 = html.Div([
        html.Div(id="availability-kpis-graph-3-container",children=[
        dcc.Graph(id="availability-kpis-graph-3"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

availability_kpis_plot_widget_4 = html.Div([
        html.Div(id="availability-kpis-graph-4-container",children=[
        dcc.Graph(id="availability-kpis-graph-4"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

availability_kpis_plot_widget_5 = html.Div([
        html.Div(id="availability-kpis-graph-5-container",children=[
        dcc.Graph(id="availability-kpis-graph-5"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

production_kpis_plot_widget = html.Div([
        html.Div(id="production-kpis-graph-container",children=[
        dcc.Graph(id="production-kpis-graph"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

production_kpis_plot_widget_2 = html.Div([
        html.Div(id="production-kpis-graph-2-container",children=[
        dcc.Graph(id="production-kpis-graph-2"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

production_kpis_plot_widget_3 = html.Div([
        html.Div(id="production-kpis-graph-3-container",children=[
        dcc.Graph(id="production-kpis-graph-3"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

production_kpis_plot_widget_4 = html.Div([
        html.Div(id="production-kpis-graph-4-container",children=[
        dcc.Graph(id="production-kpis-graph-4"),
        ],style={'display': 'none'})
        ],style={"margin": "10px"})

    
print('5447')

    
row1 = html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': "10%",'padding-top': '20px','color': 'white','text-align': 'center',}),
#        html.H1("", style={'width': "5%",'padding-top': '20px','color': 'white','text-align': 'center',}),
        html.H1(TITLE, style={'width': "80%",'padding-top': '30px','color': 'white','text-align': 'center',}),
        html.A("Manual", style={'width': "10%",'padding-top': '45px','color': 'white','text-align': 'center',},href='https://bte737364.sharepoint.com/:b:/s/Technicalsupportmeeting/EQQMiWSFpk1MsXdUU_L9160B0zCUq4fC7N2S_Wmv3iKgjg?e=ivbBZ4', target="_blank")
        ],
    className='page-header',
    style={
            'height': 'auto',
            'margin-top': '-20px',
            'background': 'rgb(64, 71, 86)',
            'display': 'flex',
            'justify-content': 'space-between',
            'text-align': 'center',
            })

    
row2 = html.Div([
                 html.Div([kpis_datatable_widget], className='col-md-12'),
                 ], className='row shaded')

row3 = html.Div([
                 html.Div([kpis_plot_widget], className='col-md-6'),
                 html.Div([kpis_plot_widget_2], className='col-md-6'),
                 ], className='row shaded')

row4 = html.Div([
                 html.Div([kpis_plot_widget_3], className='col-md-6'),
                 html.Div([kpis_plot_widget_4], className='col-md-6'),
                 ], className='row shaded')

row4b = html.Div([
                 html.Div([kpis_plot_widget_5], className='col-md-6'),
                 ], className='row shaded')
 
row11 = html.Div([
                 html.Div([reliability_kpis_datatable_widget], className='col-md-12'),
                 ], className='row shaded')

row12 = html.Div([
                 html.Div([reliability_kpis_plot_widget], className='col-md-6'),
                 html.Div([reliability_kpis_plot_widget_2], className='col-md-6'),
                 ], className='row shaded')


        
row5 = html.Div([
                 html.Div([resource_kpis_datatable_widget], className='col-md-12'),
                 ], className='row shaded')

row6 = html.Div([
                 html.Div([resource_kpis_plot_widget], className='col-md-6'),
                 html.Div([resource_kpis_plot_widget_3], className='col-md-6'),
                 ], className='row shaded')

row6b = html.Div([
                 html.Div([resource_kpis_plot_widget_2], className='col-md-4'),
#                 html.Div([resource_kpis_plot_widget_4], className='col-md-6'),
                 ], className='row shaded')

row7 = html.Div([
                 html.Div([availability_kpis_datatable_widget], className='col-md-12'),
                 ], className='row shaded')

row8 = html.Div([
                 html.Div([availability_kpis_plot_widget], className='col-md-6'),
                 html.Div([availability_kpis_datatable_2_widget], className='col-md-6'),
                 ], className='row shaded')

row8b = html.Div([
                 html.Div([availability_kpis_plot_widget_3], className='col-md-6'),
                 html.Div([availability_kpis_plot_widget_4], className='col-md-6'),
                 ], className='row shaded')

row8c = html.Div([
                 html.Div([availability_kpis_plot_widget_5], className='col-md-6'),
                 ], className='row shaded')

row9 = html.Div([
                 html.Div([production_kpis_datatable_widget], className='col-md-12'),
                 ], className='row shaded')

row10 = html.Div([
                 html.Div([production_kpis_plot_widget], className='col-md-6'),
                 html.Div([production_kpis_plot_widget_2], className='col-md-6'),
                 ], className='row shaded')

row10b = html.Div([
                 html.Div([production_kpis_plot_widget_3], className='col-md-6'),
                 html.Div([production_kpis_plot_widget_4], className='col-md-6'),
                 ], className='row shaded')



app.layout = html.Div([row1,
    html.Div(
            dcc.Tabs([
                dcc.Tab(label='Wind resource KPIs', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div([row5,row6,row6b],className='container-fluid')]),
                dcc.Tab(label='Availability KPIs', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div([row7,row8,row8b,row8c],className='container-fluid')]),
                dcc.Tab(label='Production KPIs', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div([row9,row10,row10b],className='container-fluid')]),
                dcc.Tab(label='Performance KPIs', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div([row2,row3,row4,row4b],className='container-fluid')]),
                dcc.Tab(label='Reliability KPIs', style=tab_style, selected_style=tab_selected_style, children=[
                        html.Div([row11,row12],className='container-fluid')]),
    ])
    )
])

    
    
@app.callback(
    Output("kpis_datatable_copy", "content"),
    Input("kpis_datatable_copy", "n_clicks"),
    State("kpis-datatable", "data"),
)
def custom_copy_1(_, data):
    dff = pd.DataFrame(data)
    for col in dff.columns:
        try:
            dff[col]=dff[col].str.replace('⚫', 'black')
            dff[col]=dff[col].str.replace('🔴', 'red')
            dff[col]=dff[col].str.replace('🟠', 'orange')
            dff[col]=dff[col].str.replace('🟢', 'green')
        except:
            pass
    dff=dff[["Country","Wind farm","Wind turbine","Manufacturer","Type","Asset Manager","SCADA wind direction KPI","Ambient temperature KPI","Static yaw misalignment KPI","Dynamic yaw misalignment KPI","Pitch vs Power KPI","Pitch vs RPM KPI","RPM vs Power KPI","Power vs RPM KPI","Power vs Speed KPI","Pitch vs Speed KPI","RPM vs Speed KPI"]]
    # See options for .to_csv() or .to_excel() or .to_string() in the  pandas documentation
    return dff.to_csv(index=False)  # includes headers


@app.callback(
    Output("reliability_kpis_datatable_copy", "content"),
    Input("reliability_kpis_datatable_copy", "n_clicks"),
    State("reliability-kpis-datatable", "data"),
)
def custom_copy_1b(_, data):
    dff = pd.DataFrame(data)
    for col in dff.columns:
        try:
            dff[col]=dff[col].str.replace('⚫', 'black')
            dff[col]=dff[col].str.replace('🔴', 'red')
            dff[col]=dff[col].str.replace('🟠', 'orange')
            dff[col]=dff[col].str.replace('🟢', 'green')
        except:
            pass
    dff=dff[["Country","Wind farm","Wind turbine","Manufacturer","Type","Asset Manager","Front bearing temperature KPI","Rear bearing temperature KPI","Main bearing temperature KPI","Gearbox HSS bearing temperature KPI","Gearbox IMS/LSS bearing temperature KPI","Metal particle count KPI","Gearbox oil temperature KPI","Generator bearing front temperature KPI","Generator bearing rear temperature KPI","Rotor temperature KPI","Stator temperature KPI"]]
    # See options for .to_csv() or .to_excel() or .to_string() in the  pandas documentation
    return dff.to_csv(index=False)  # includes headers


    
@app.callback(
    Output("resource_kpis_datatable_copy", "content"),
    Input("resource_kpis_datatable_copy", "n_clicks"),
    State("resource-kpis-datatable", "data"),
)
def custom_copy_2(_, data):
    dff = pd.DataFrame(data)
    for col in dff.columns:
        try:
            dff[col]=dff[col].str.replace('⚫', 'black')
            dff[col]=dff[col].str.replace('🔴', 'red')
            dff[col]=dff[col].str.replace('🟠', 'orange')
            dff[col]=dff[col].str.replace('🟢', 'green')
        except:
            pass
    dff=dff[["Country","Wind farm","Monthly Wind Resource KPI","Monthly Year-To-Date Wind Resource KPI"]]
    # See options for .to_csv() or .to_excel() or .to_string() in the  pandas documentation
    return dff.to_csv(index=False)  # includes headers
    
@app.callback(
    Output("availability_kpis_datatable_copy", "content"),
    Input("availability_kpis_datatable_copy", "n_clicks"),
    State("availability-kpis-datatable", "data"),
)
def custom_copy_3(_, data):
    dff = pd.DataFrame(data)
    for col in dff.columns:
        try:
            dff[col]=dff[col].str.replace('⚫', 'black')
            dff[col]=dff[col].str.replace('🔴', 'red')
            dff[col]=dff[col].str.replace('🟠', 'orange')
            dff[col]=dff[col].str.replace('🟢', 'green')
        except:
            pass
    dff=dff[['Country',"Wind farm","Wind turbine","Manufacturer","Monthly time-based System Avail. KPI","Monthly production-based System Avail. KPI","Monthly Year-To-Date time-based System Avail. KPI","Monthly Year-To-Date production-based System Avail. KPI","Monthly own stops KPI","Monthly scheduled maintenance stops KPI","Monthly Year-To-Date own stops KPI","Monthly Year-To-Date scheduled maintenance stops KPI","Status Codes KPI","Warnings KPI"]]
    # See options for .to_csv() or .to_excel() or .to_string() in the  pandas documentation
    return dff.to_csv(index=False)  # includes headers

@app.callback(
    Output("production_kpis_datatable_copy", "content"),
    Input("production_kpis_datatable_copy", "n_clicks"),
    State("production-kpis-datatable", "data"),
)
def custom_copy_4(_, data):
    dff = pd.DataFrame(data)
    for col in dff.columns:
        try:
            dff[col]=dff[col].str.replace('⚫', 'black')
            dff[col]=dff[col].str.replace('🔴', 'red')
            dff[col]=dff[col].str.replace('🟠', 'orange')
            dff[col]=dff[col].str.replace('🟢', 'green')
        except:
            pass
    dff=dff[["Country","Wind farm","Monthly production KPI","Monthly Year-To-Date production KPI"]]
    # See options for .to_csv() or .to_excel() or .to_string() in the  pandas documentation
    return dff.to_csv(index=False)  # includes headers

@app.callback(
    Output("availability_kpis_datatable_2_copy", "content"),
    Input("availability_kpis_datatable_2_copy", "n_clicks"),
    State("availability-kpis-datatable-2", "data"),
)
def custom_copy_5(_, data):
    dff = pd.DataFrame(data)
    for col in dff.columns:
        try:
            dff[col]=dff[col].str.replace('⚫', 'black')
            dff[col]=dff[col].str.replace('🔴', 'red')
            dff[col]=dff[col].str.replace('🟠', 'orange')
            dff[col]=dff[col].str.replace('🟢', 'green')
        except:
            pass
    if "Duration KPI" in dff.columns:
        dff=dff[["Duration KPI","Start","End","Duration","% Time","Code","Message","Category","Global contract category"]]
    elif "Loss KPI" in dff.columns:
        dff=dff[["Loss KPI","Start","End","Duration","% Loss","Code","Message","Category","Global contract category"]]
    elif "Own stop KPI" in dff.columns:
        dff=dff[["Own stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
    elif "Scheduled maintenance stop KPI" in dff.columns:
        dff=dff[["Scheduled maintenance stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
    elif "Status Code KPI" in dff.columns:
        dff=dff[["Status Code KPI","Code","Message","Category","Global contract category","Duration","Count","P95","P99"]]
    elif "Warning KPI" in dff.columns:
        dff=dff[["Warning KPI","Code","Message","Category","Global contract category","Duration","Count","P95","P99"]]
    else:
         dff=dff[["Start","End","Duration"]]
    # See options for .to_csv() or .to_excel() or .to_string() in the  pandas documentation
    return dff.to_csv(index=False)  # includes headers
    
    
    
    
@app.callback(
        [Output('kpis-graph', 'figure'),
         Output("kpis-graph-container","style")
         ],[Input('kpis-datatable', 'active_cell')])
def kpis_graph(active_cell):
    
    selected_wt=current_kpis["Wind turbine"].iloc[active_cell["row_id"]]
    figure_appearance={'display': 'none'}



    if active_cell["column_id"]=="Static yaw misalignment KPI":
        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]

        kpis_figure = go.Figure()
        
        kpis_figure.add_trace(go.Scatter(
                name='P50',
                legendgroup = 'a',
                x=data_over_time["Month"],
                y=data_over_time["Yaw misalignment P50 (°)"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))

        kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'b',
                x=data_over_time["Month"],
                y=data_over_time["Yaw misalignment P25 (°)"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':"#444"
#                        'width':0
                        },
#                marker=dict(color="#444"),
                showlegend=True
            ))

        kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'b',
                x=data_over_time["Month"],
                y=data_over_time["Yaw misalignment P75 (°)"],
#                marker=dict(color="#444"),
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':"#444"
#                        'width':0
                        },
#                fillcolor='rgba(68, 68, 68, 0.3)',
#                fill='tonexty',
                showlegend=False
            ))

        kpis_figure.add_trace(go.Scatter(
                name="Best guess",
                legendgroup = 'c',
                x=data_over_time["Month"],
                y=data_over_time["Static yaw misalignment best guess"],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Black'
                        },
                showlegend=True
            ))


        kpis_figure.add_trace(go.Scatter(
                name="0",
                legendgroup = 'd',
                x=data_over_time["Month"],
                y=[0 for i in data_over_time.index],
                mode='lines',
                line=dict(color="Green",dash="dashdot"),
                showlegend=True
            ))


        kpis_figure.add_trace(go.Scatter(
                name="-5/+5",
                legendgroup = 'e',
                x=data_over_time["Month"],
                y=[-5 for i in data_over_time.index],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=True
            ))

        kpis_figure.add_trace(go.Scatter(
                name="-5/+5",
                legendgroup = 'e',
                x=data_over_time["Month"],
                y=[5 for i in data_over_time.index],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=False
            ))

        kpis_figure.add_trace(go.Scatter(
                name="-10/+10",
                legendgroup = 'f',
                x=data_over_time["Month"],
                y=[-10 for i in data_over_time.index],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=True
            ))

        kpis_figure.add_trace(go.Scatter(
                name="-10/+10",
                legendgroup = 'f',
                x=data_over_time["Month"],
                y=[10 for i in data_over_time.index],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=False
            ))


        kpis_figure.update_yaxes(title="Static yaw misalignment (°)",range=[-15,15],showgrid=False)
        kpis_figure.update_xaxes(showgrid=False)
        kpis_figure.update_layout(title=dict(text=selected_wt+' - Historical static yaw misalignment',x=0.5))
        figure_appearance={'display': 'block'}
        




    elif active_cell["column_id"]=="Ambient temperature KPI":
        
        current_wf=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind farm"].iloc[0]
        tmp=kpis.loc[kpis["Wind farm"]==current_wf]

        kpis_figure = go.Figure()
        
        for wt in natsorted(list(set(tmp["Wind turbine"].tolist()))):
        
            data_over_time=tmp.loc[tmp["Wind turbine"]==wt]

        
        
            kpis_figure.add_trace(go.Scatter(
                    name=wt,
                    x=data_over_time["Month"],
                    y=data_over_time["Monthly ambient temperature"],
                    mode='lines+markers',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            },
                    showlegend=True
                ))



        kpis_figure.update_yaxes(title="Ambient temperature (°)",showgrid=False)
        kpis_figure.update_xaxes(showgrid=False)
        kpis_figure.update_layout(title=dict(text=current_wf+' - Historical ambient temperatures',x=0.5))
        figure_appearance={'display': 'block'}




    

    elif active_cell["column_id"]=="SCADA wind direction KPI":
        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]

        kpis_figure = go.Figure()
        
        
        kpis_figure.add_trace(go.Scatter(
                name='Offset',
                legendgroup = 'a',
                x=data_over_time["Month"],
                y=data_over_time["Current wind direction offset to true north (°)"],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))


        kpis_figure.add_trace(go.Scatter(
                name="0",
                legendgroup = 'b',
                x=data_over_time["Month"],
                y=[0 for i in data_over_time.index],
                mode='lines',
                line=dict(color="Green",dash="dashdot"),
                showlegend=True
            ))


        kpis_figure.add_trace(go.Scatter(
                name="-15/+15",
                legendgroup = 'c',
                x=data_over_time["Month"],
                y=[-15 for i in data_over_time.index],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=True
            ))

        kpis_figure.add_trace(go.Scatter(
                name="-15/+15",
                legendgroup = 'c',
                x=data_over_time["Month"],
                y=[15 for i in data_over_time.index],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=False
            ))

        kpis_figure.add_trace(go.Scatter(
                name="-30/+30",
                legendgroup = 'd',
                x=data_over_time["Month"],
                y=[-30 for i in data_over_time.index],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=True
            ))

        kpis_figure.add_trace(go.Scatter(
                name="-30/+30",
                legendgroup = 'd',
                x=data_over_time["Month"],
                y=[30 for i in data_over_time.index],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=False
            ))

        kpis_figure.update_yaxes(title="SCADA wind direction offset (°)",showgrid=False)
        kpis_figure.update_xaxes(showgrid=False)
        kpis_figure.update_layout(title=dict(text=selected_wt+' - Historical SCADA wind direction offset',x=0.5))
        figure_appearance={'display': 'block'}
        







    elif active_cell["column_id"]=="Dynamic yaw misalignment KPI":
                
        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of dynamic yaw misalignment',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Dynamic yaw misalignment KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of dynamic yaw misalignment',x=0.5))
        figure_appearance={'display': 'block'}







    elif active_cell["column_id"]=="Pitch vs Power KPI":
        
        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
#        data_over_time=data_over_time.loc[data_over_time["DateAsDate"]>dt(2019,12,1)]
        data_over_time=data_over_time[["Month","Pitch vs Power"]]
        data_over_time["Pitch vs Power"]=data_over_time["Pitch vs Power"].apply(lambda x: eval(x))
        data_over_time["bins"]=data_over_time["Pitch vs Power"].apply(lambda x: x.keys())
        bins=data_over_time["bins"].tolist()
        bins=[item for sublist in bins for item in sublist]
        bins=sorted(list(set(bins)))
        
        data_over_time = pd.concat([data_over_time["Month"], data_over_time["Pitch vs Power"].apply(pd.Series)], axis=1)
        
        data_over_time.index=data_over_time["Month"]
        data_over_time=data_over_time.drop(columns=["Month"])
        dummy=pd.DataFrame(index=data_over_time.index)
        dummy["val"]=1

        kpis_figure = go.Figure()
        
        for col in data_over_time.columns:
            kpis_figure.add_trace(go.Bar(
                name=str(col),
                x=data_over_time.index,
                y=dummy["val"],
                marker_color=data_over_time[col].tolist(),
                showlegend=False
            ))
        kpis_figure.update_layout(barmode='stack')

        kpis_figure.update_yaxes(
                title="Power bin",
                showticklabels=False,
                showgrid=False
                )
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Historical Pitch vs Power behavior',x=0.5))
        figure_appearance={'display': 'block'}




    elif active_cell["column_id"]=="Pitch vs RPM KPI":
        
        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
#        data_over_time=data_over_time.loc[data_over_time["DateAsDate"]>dt(2019,12,1)]
        data_over_time=data_over_time[["Month","Pitch vs RPM"]]
        data_over_time["Pitch vs RPM"]=data_over_time["Pitch vs RPM"].apply(lambda x: eval(x))
        data_over_time["bins"]=data_over_time["Pitch vs RPM"].apply(lambda x: x.keys())
        bins=data_over_time["bins"].tolist()
        bins=[item for sublist in bins for item in sublist]
        bins=sorted(list(set(bins)))
        
        data_over_time = pd.concat([data_over_time["Month"], data_over_time["Pitch vs RPM"].apply(pd.Series)], axis=1)
        
        data_over_time.index=data_over_time["Month"]
        data_over_time=data_over_time.drop(columns=["Month"])
        dummy=pd.DataFrame(index=data_over_time.index)
        dummy["val"]=1

        kpis_figure = go.Figure()
        
        for col in data_over_time.columns:
            kpis_figure.add_trace(go.Bar(
                name=str(col),
                x=data_over_time.index,
                y=dummy["val"],
                marker_color=data_over_time[col].tolist(),
                showlegend=False
            ))
        kpis_figure.update_layout(barmode='stack')

        kpis_figure.update_yaxes(
                title="Rotor speed bin",
                showticklabels=False,
                showgrid=False
                )
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Historical Pitch vs RPM behavior',x=0.5))
        figure_appearance={'display': 'block'}




    elif active_cell["column_id"]=="RPM vs Power KPI":
        
        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
#        data_over_time=data_over_time.loc[data_over_time["DateAsDate"]>dt(2019,12,1)]
        data_over_time=data_over_time[["Month","RPM vs Power"]]
        data_over_time["RPM vs Power"]=data_over_time["RPM vs Power"].apply(lambda x: eval(x))
        data_over_time["bins"]=data_over_time["RPM vs Power"].apply(lambda x: x.keys())
        bins=data_over_time["bins"].tolist()
        bins=[item for sublist in bins for item in sublist]
        bins=sorted(list(set(bins)))
        
        data_over_time = pd.concat([data_over_time["Month"], data_over_time["RPM vs Power"].apply(pd.Series)], axis=1)
        
        data_over_time.index=data_over_time["Month"]
        data_over_time=data_over_time.drop(columns=["Month"])
        dummy=pd.DataFrame(index=data_over_time.index)
        dummy["val"]=1

        kpis_figure = go.Figure()
        
        for col in data_over_time.columns:
            kpis_figure.add_trace(go.Bar(
                name=str(col),
                x=data_over_time.index,
                y=dummy["val"],
                marker_color=data_over_time[col].tolist(),
                showlegend=False
            ))
        kpis_figure.update_layout(barmode='stack')

        kpis_figure.update_yaxes(
                title="Power bin",
                showticklabels=False,
                showgrid=False
                )
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Historical RPM vs Power behavior',x=0.5))
        figure_appearance={'display': 'block'}




    elif active_cell["column_id"]=="Power vs Speed KPI":
        
        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
#        data_over_time=data_over_time.loc[data_over_time["DateAsDate"]>dt(2019,12,1)]
        data_over_time=data_over_time[["Month","Power vs Speed"]]
        data_over_time["Power vs Speed"]=data_over_time["Power vs Speed"].apply(lambda x: eval(x))
        data_over_time["bins"]=data_over_time["Power vs Speed"].apply(lambda x: x.keys())
        bins=data_over_time["bins"].tolist()
        bins=[item for sublist in bins for item in sublist]
        bins=sorted(list(set(bins)))
        
        data_over_time = pd.concat([data_over_time["Month"], data_over_time["Power vs Speed"].apply(pd.Series)], axis=1)
        
        data_over_time.index=data_over_time["Month"]
        data_over_time=data_over_time.drop(columns=["Month"])
        dummy=pd.DataFrame(index=data_over_time.index)
        dummy["val"]=1

        kpis_figure = go.Figure()
        
        for col in data_over_time.columns:
            kpis_figure.add_trace(go.Bar(
                name=str(col),
                x=data_over_time.index,
                y=dummy["val"],
                marker_color=data_over_time[col].tolist(),
                showlegend=False
            ))
        kpis_figure.update_layout(barmode='stack')

        kpis_figure.update_yaxes(
                title="Wind speed bin",
                showticklabels=False,
                showgrid=False
                )
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Historical Power vs Speed behavior',x=0.5))
        figure_appearance={'display': 'block'}






    elif active_cell["column_id"]=="Pitch vs Speed KPI":
        
        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
#        data_over_time=data_over_time.loc[data_over_time["DateAsDate"]>dt(2019,12,1)]
        data_over_time=data_over_time[["Month","Pitch vs Speed"]]
        data_over_time["Pitch vs Speed"]=data_over_time["Pitch vs Speed"].apply(lambda x: eval(x))
        data_over_time["bins"]=data_over_time["Pitch vs Speed"].apply(lambda x: x.keys())
        bins=data_over_time["bins"].tolist()
        bins=[item for sublist in bins for item in sublist]
        bins=sorted(list(set(bins)))
        
        data_over_time = pd.concat([data_over_time["Month"], data_over_time["Pitch vs Speed"].apply(pd.Series)], axis=1)
        
        data_over_time.index=data_over_time["Month"]
        data_over_time=data_over_time.drop(columns=["Month"])
        dummy=pd.DataFrame(index=data_over_time.index)
        dummy["val"]=1

        kpis_figure = go.Figure()
        
        for col in data_over_time.columns:
            kpis_figure.add_trace(go.Bar(
                name=str(col),
                x=data_over_time.index,
                y=dummy["val"],
                marker_color=data_over_time[col].tolist(),
                showlegend=False
            ))
        kpis_figure.update_layout(barmode='stack')

        kpis_figure.update_yaxes(
                title="Wind speed bin",
                showticklabels=False,
                showgrid=False
                )
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Historical Pitch vs Speed behavior',x=0.5))
        figure_appearance={'display': 'block'}










    elif active_cell["column_id"]=="RPM vs Speed KPI":
        
        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
#        data_over_time=data_over_time.loc[data_over_time["DateAsDate"]>dt(2019,12,1)]
        data_over_time=data_over_time[["Month","RPM vs Speed"]]
        data_over_time["RPM vs Speed"]=data_over_time["RPM vs Speed"].apply(lambda x: eval(x))
        data_over_time["bins"]=data_over_time["RPM vs Speed"].apply(lambda x: x.keys())
        bins=data_over_time["bins"].tolist()
        bins=[item for sublist in bins for item in sublist]
        bins=sorted(list(set(bins)))
        
        data_over_time = pd.concat([data_over_time["Month"], data_over_time["RPM vs Speed"].apply(pd.Series)], axis=1)
        
        data_over_time.index=data_over_time["Month"]
        data_over_time=data_over_time.drop(columns=["Month"])
        dummy=pd.DataFrame(index=data_over_time.index)
        dummy["val"]=1

        kpis_figure = go.Figure()
        
        for col in data_over_time.columns:
            kpis_figure.add_trace(go.Bar(
                name=str(col),
                x=data_over_time.index,
                y=dummy["val"],
                marker_color=data_over_time[col].tolist(),
                showlegend=False
            ))
        kpis_figure.update_layout(barmode='stack')

        kpis_figure.update_yaxes(
                title="Wind speed bin",
                showticklabels=False,
                showgrid=False
                )
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Historical RPM vs Speed behavior',x=0.5))
        figure_appearance={'display': 'block'}








    elif active_cell["column_id"]=="Power vs RPM KPI":
        
        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
#        data_over_time=data_over_time.loc[data_over_time["DateAsDate"]>dt(2019,12,1)]
        data_over_time=data_over_time[["Month","Power vs RPM"]]
        data_over_time["Power vs RPM"]=data_over_time["Power vs RPM"].apply(lambda x: eval(x))
        data_over_time["bins"]=data_over_time["Power vs RPM"].apply(lambda x: x.keys())
        bins=data_over_time["bins"].tolist()
        bins=[item for sublist in bins for item in sublist]
        bins=sorted(list(set(bins)))
        
        data_over_time = pd.concat([data_over_time["Month"], data_over_time["Power vs RPM"].apply(pd.Series)], axis=1)
        
        data_over_time.index=data_over_time["Month"]
        data_over_time=data_over_time.drop(columns=["Month"])
        dummy=pd.DataFrame(index=data_over_time.index)
        dummy["val"]=1

        kpis_figure = go.Figure()
        
        for col in data_over_time.columns:
            kpis_figure.add_trace(go.Bar(
                name=str(col),
                x=data_over_time.index,
                y=dummy["val"],
                marker_color=data_over_time[col].tolist(),
                showlegend=False
            ))
        kpis_figure.update_layout(barmode='stack')

        kpis_figure.update_yaxes(
                title="Rotor speed bin",
                showticklabels=False,
                showgrid=False
                )
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Historical Power vs RPM behavior',x=0.5))
        figure_appearance={'display': 'block'}




    else:
        kpis_figure=default_figure
    
    return kpis_figure,figure_appearance



    












@app.callback(
        [Output('reliability-kpis-graph', 'figure'),
         Output("reliability-kpis-graph-container","style")
         ],[Input('reliability-kpis-datatable', 'active_cell')])
def reliability_kpis_graph(active_cell):
    
    selected_wt=current_kpis["Wind turbine"].iloc[active_cell["row_id"]]
    figure_appearance={'display': 'none'}


    if active_cell["column_id"]=="Aerodynamic rotor imbalance KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of Aerodynamic rotor imbalance',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Aerodynamic rotor imbalance KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of Aerodynamic rotor imbalance',x=0.5))
        figure_appearance={'display': 'block'}


    elif active_cell["column_id"]=="Mass rotor imbalance KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of Mass rotor imbalance',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Mass rotor imbalance KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of Mass rotor imbalance',x=0.5))
        figure_appearance={'display': 'block'}

    elif active_cell["column_id"]=="Global rotor imbalance KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of Global rotor imbalance',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Global rotor imbalance KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of Global rotor imbalance',x=0.5))
        figure_appearance={'display': 'block'}






    elif active_cell["column_id"]=="Front bearing temperature KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of front bearing temperature',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Front bearing temperature KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of front bearing temperature',x=0.5))
        figure_appearance={'display': 'block'}



    elif active_cell["column_id"]=="Rear bearing temperature KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of rear bearing temperature',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Rear bearing temperature KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of rear bearing temperature',x=0.5))
        figure_appearance={'display': 'block'}




    elif active_cell["column_id"]=="Rotor temperature KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of rotor temperature',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Rotor temperature KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of rotor temperature',x=0.5))
        figure_appearance={'display': 'block'}




    elif active_cell["column_id"]=="Stator temperature KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of stator temperature',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Stator temperature KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of stator temperature',x=0.5))
        figure_appearance={'display': 'block'}








    elif active_cell["column_id"]=="Gearbox HSS bearing temperature KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of gearbox HSS bearing temperature',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Gearbox HSS bearing temperature KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of gearbox HSS bearing temperature',x=0.5))
        figure_appearance={'display': 'block'}


    elif active_cell["column_id"]=="Gearbox IMS/LSS bearing temperature KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of gearbox IMS/LSS bearing temperature',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Gearbox IMS/LSS bearing temperature KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of gearbox IMS/LSS bearing temperature',x=0.5))
        figure_appearance={'display': 'block'}



    elif active_cell["column_id"]=="Generator bearing front temperature KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of generator bearing front temperature',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Generator bearing front temperature KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of generator bearing front temperature',x=0.5))
        figure_appearance={'display': 'block'}


    elif active_cell["column_id"]=="Generator bearing rear temperature KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of generator bearing rear temperature',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Generator bearing rear temperature KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of generator bearing rear temperature',x=0.5))
        figure_appearance={'display': 'block'}




    elif active_cell["column_id"]=="Main bearing temperature KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of main bearing temperature',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Main bearing temperature KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of main bearing temperature',x=0.5))
        figure_appearance={'display': 'block'}



    elif active_cell["column_id"]=="Metal particle count KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of metal particle count',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Metal particle count KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of metal particle count',x=0.5))
        figure_appearance={'display': 'block'}



    elif active_cell["column_id"]=="Gearbox oil temperature KPI":
        

        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]
        
        kpis_figure = go.Figure()
                
        kpis_figure.add_trace(go.Scatter(
                name='Assessment of gearbox oil temperature',
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Gearbox oil temperature KPI Color"].tolist(),'size': 12},
                showlegend=False
            ))

        kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        kpis_figure.update_xaxes(showgrid=False)

        kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of gearbox oil temperature',x=0.5))
        figure_appearance={'display': 'block'}




    else:
        kpis_figure=default_figure
    
    return kpis_figure,figure_appearance












@app.callback(
        [Output('kpis-graph-2', 'figure'),
         Output("kpis-graph-2-container","style")
         ],[Input('kpis-datatable', 'active_cell'),
         Input('kpis-graph','clickData')])
def kpis_graph_2(active_cell,clickData):
    
    selected_wt=current_kpis["Wind turbine"].iloc[active_cell["row_id"]]

    if clickData is not None:
        selected_time=clickData['points'][0]['x']
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)
        selected_time=selected_time.strftime("%B %Y")

    graph_month=dt.strptime(selected_time,'%B %Y').strftime("%B")
    graph_year=dt.strptime(selected_time,'%B %Y').year
    figure_appearance={'display': 'none'}

    if active_cell["column_id"]=="Static yaw misalignment KPI":

        current_wf=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind farm"].iloc[0]
        wf_yaw_misalignments=kpis.loc[(kpis["Wind farm"]==current_wf ) & (kpis["Month"]==selected_time)]
        wf_yaw_misalignments=wf_yaw_misalignments[["Static yaw misalignment best guess","Wind turbine"]]
        wf_yaw_misalignments=wf_yaw_misalignments.sort_values(by="Wind turbine",key=lambda x: np.argsort(index_natsorted(wf_yaw_misalignments["Wind turbine"])),ascending=False)
        wf_yaw_misalignments.index=range(len(wf_yaw_misalignments.index))

        kpis_figure_2 = go.Figure()
        
        kpis_figure_2.add_trace(go.Scatter(
                name="Best guess",
                legendgroup = 'a',
                text=wf_yaw_misalignments["Static yaw misalignment best guess"],
                textposition='top center',
                x=wf_yaw_misalignments["Static yaw misalignment best guess"],
                y=wf_yaw_misalignments["Wind turbine"],
                mode="markers+text",
                marker=dict(color="Black"),
                showlegend=True
            ))

        kpis_figure_2.add_trace(go.Scatter(
                name="0",
                legendgroup = 'b',
                x=[0 for i in wf_yaw_misalignments.index],
                y=wf_yaw_misalignments["Wind turbine"],
                mode='lines',
                line=dict(color="Green",dash="dashdot"),
                showlegend=True
            ))

        kpis_figure_2.add_trace(go.Scatter(
                name="-5/+5",
                legendgroup = 'c',
                x=[-5 for i in wf_yaw_misalignments.index],
                y=wf_yaw_misalignments["Wind turbine"],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=True
            ))

        kpis_figure_2.add_trace(go.Scatter(
                name="-5/+5",
                legendgroup = 'c',
                x=[5 for i in wf_yaw_misalignments.index],
                y=wf_yaw_misalignments["Wind turbine"],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=False
            ))

        kpis_figure_2.add_trace(go.Scatter(
                name="-10/+10",
                legendgroup = 'd',
                x=[-10 for i in wf_yaw_misalignments.index],
                y=wf_yaw_misalignments["Wind turbine"],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=True
            ))

        kpis_figure_2.add_trace(go.Scatter(
                name="-10/+10",
                legendgroup = 'd',
                x=[10 for i in wf_yaw_misalignments.index],
                y=wf_yaw_misalignments["Wind turbine"],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=False
            ))

        kpis_figure_2.update_yaxes(showgrid=False)
        kpis_figure_2.update_xaxes(title="Static yaw misalignment (°)",range=[-15,15],showgrid=False)
        kpis_figure_2.update_layout(title=dict(text=str(current_wf)+" - "+str(graph_month)+" "+str(graph_year)+' - Static yaw misalignments',x=0.5))
        figure_appearance={'display': 'block'}


    

    elif active_cell["column_id"]=="SCADA wind direction KPI":
        current_wf=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind farm"].iloc[0]
        wf_wind_direction_offsets=kpis.loc[(kpis["Wind farm"]==current_wf ) & (kpis["Month"]==selected_time)]
        wf_wind_direction_offsets=wf_wind_direction_offsets[["Current wind direction offset to true north (°)","Wind turbine"]]
        wf_wind_direction_offsets=wf_wind_direction_offsets.sort_values(by="Wind turbine",key=lambda x: np.argsort(index_natsorted(wf_wind_direction_offsets["Wind turbine"])),ascending=False)
        wf_wind_direction_offsets.index=range(len(wf_wind_direction_offsets.index))
        wf_wind_direction_offsets["Current wind direction offset to true north (°)"]=wf_wind_direction_offsets["Current wind direction offset to true north (°)"].apply(lambda x: round(x,0))




        kpis_figure_2 = go.Figure()
        
        kpis_figure_2.add_trace(go.Scatter(
                name="Offset",
                legendgroup = 'a',
                text=wf_wind_direction_offsets["Current wind direction offset to true north (°)"],
                textposition='top center',
                x=wf_wind_direction_offsets["Current wind direction offset to true north (°)"],
                y=wf_wind_direction_offsets["Wind turbine"],
                mode="markers+text",
                marker=dict(color='rgb(31, 119, 180)'),
                showlegend=True
            ))

        kpis_figure_2.add_trace(go.Scatter(
                name="0",
                legendgroup = 'b',
                x=[0 for i in wf_wind_direction_offsets.index],
                y=wf_wind_direction_offsets["Wind turbine"],
                mode='lines',
                line=dict(color="Green",dash="dashdot"),
                showlegend=True
            ))

        kpis_figure_2.add_trace(go.Scatter(
                name="-15/+15",
                legendgroup = 'c',
                x=[-15 for i in wf_wind_direction_offsets.index],
                y=wf_wind_direction_offsets["Wind turbine"],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=True
            ))

        kpis_figure_2.add_trace(go.Scatter(
                name="-15/+15",
                legendgroup = 'c',
                x=[15 for i in wf_wind_direction_offsets.index],
                y=wf_wind_direction_offsets["Wind turbine"],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=False
            ))

        kpis_figure_2.add_trace(go.Scatter(
                name="-30/+30",
                legendgroup = 'd',
                x=[-30 for i in wf_wind_direction_offsets.index],
                y=wf_wind_direction_offsets["Wind turbine"],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=True
            ))

        kpis_figure_2.add_trace(go.Scatter(
                name="-30/+30",
                legendgroup = 'd',
                x=[30 for i in wf_wind_direction_offsets.index],
                y=wf_wind_direction_offsets["Wind turbine"],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=False
            ))

        kpis_figure_2.update_yaxes(showgrid=False)
        kpis_figure_2.update_xaxes(title="SCADA wind direction offset (°)",showgrid=False)
        kpis_figure_2.update_layout(title=dict(text=str(current_wf)+" - "+str(graph_month)+" "+str(graph_year)+' - SCADA wind direction offsets',x=0.5))
        figure_appearance={'display': 'block'}






    elif active_cell["column_id"]=="Pitch vs Power KPI":

        try:
            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"Pitch vs Power"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Power","Blade angle"]].dropna()
            actual_data["Blade angle"]=actual_data["Blade angle"].apply(lambda x: ((x+180)%360)-180)
#            actual_data=actual_data.loc[actual_data["Blade angle"]<20]
            actual_data["Power"]=actual_data["Power"].apply(lambda x: round(x,0))  
            actual_data=actual_data.loc[actual_data["Power"].isin(expected_relationship.index)]
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Blade angle"]>=expected_relationship.at[row["Power"],"Avg"]-1.96*expected_relationship.at[row["Power"],"Stdev"])&(row["Blade angle"]<=expected_relationship.at[row["Power"],"Avg"]+1.96*expected_relationship.at[row["Power"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='P50',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]+1.96*expected_relationship["Stdev"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P95/P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]-1.96*expected_relationship["Stdev"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Power"],
                    y=actual_data["Blade angle"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
    
    
            kpis_figure_2.update_yaxes(title="Blade angle (°)",showgrid=False)
            kpis_figure_2.update_xaxes(title="Power (kW)",showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Pitch vs Power Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure









    elif active_cell["column_id"]=="Pitch vs RPM KPI":

        try:
            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"Pitch vs RPM"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Rotor speed","Blade angle"]].dropna()
            actual_data["Blade angle"]=actual_data["Blade angle"].apply(lambda x: ((x+180)%360)-180)
#            actual_data=actual_data.loc[actual_data["Blade angle"]<20]
            actual_data["Rotor speed"]=actual_data["Rotor speed"].apply(lambda x: round(x,1))  
            actual_data=actual_data.loc[actual_data["Rotor speed"].isin(expected_relationship.index)]
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Blade angle"]>=expected_relationship.at[row["Rotor speed"],"Avg"]-1.96*expected_relationship.at[row["Rotor speed"],"Stdev"])&(row["Blade angle"]<=expected_relationship.at[row["Rotor speed"],"Avg"]+1.96*expected_relationship.at[row["Rotor speed"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='P50',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]+1.96*expected_relationship["Stdev"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P95/P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]-1.96*expected_relationship["Stdev"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Rotor speed"],
                    y=actual_data["Blade angle"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
    
    
            kpis_figure_2.update_yaxes(title="Blade angle (°)",showgrid=False)
            kpis_figure_2.update_xaxes(title="Rotor speed (RPM)",showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Pitch vs RPM Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure







    elif active_cell["column_id"]=="RPM vs Power KPI":
        


        try:

            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"RPM vs Power"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Power","Rotor speed"]].dropna()
            actual_data["Power"]=actual_data["Power"].apply(lambda x: round(x,0))  
            actual_data=actual_data.loc[actual_data["Power"].isin(expected_relationship.index)]
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Rotor speed"]>=expected_relationship.at[row["Power"],"Avg"]-1.96*expected_relationship.at[row["Power"],"Stdev"])&(row["Rotor speed"]<=expected_relationship.at[row["Power"],"Avg"]+1.96*expected_relationship.at[row["Power"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
    
            kpis_figure_2 = go.Figure()
            
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='P50',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]+1.96*expected_relationship["Stdev"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P95/P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]-1.96*expected_relationship["Stdev"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Power"],
                    y=actual_data["Rotor speed"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
    
    
            kpis_figure_2.update_yaxes(title="Rotor speed (RPM)",showgrid=False)
            kpis_figure_2.update_xaxes(title="Power (kW)",showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - RPM vs Power Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure











    elif active_cell["column_id"]=="Power vs Speed KPI":
        




        try:

            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"Power vs Speed"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Corrected wind speed","Power"]].dropna()
            actual_data["Corrected wind speed"]=actual_data["Corrected wind speed"].apply(lambda x: round(x,1))  
            actual_data=actual_data.loc[actual_data["Corrected wind speed"].isin(expected_relationship.index)]
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Power"]>=expected_relationship.at[row["Corrected wind speed"],"Avg"]-1.96*expected_relationship.at[row["Corrected wind speed"],"Stdev"])&(row["Power"]<=expected_relationship.at[row["Corrected wind speed"],"Avg"]+1.96*expected_relationship.at[row["Corrected wind speed"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
    
            kpis_figure_2 = go.Figure()
            
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='P50',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]+1.96*expected_relationship["Stdev"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P95/P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]-1.96*expected_relationship["Stdev"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Corrected wind speed"],
                    y=actual_data["Power"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
    
    
            kpis_figure_2.update_yaxes(title="Power (kW)",showgrid=False)
            kpis_figure_2.update_xaxes(title="Wind speed (m/s)",showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Power vs Speed Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure















    elif active_cell["column_id"]=="Pitch vs Speed KPI":
        



        try:

            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"Pitch vs Speed"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Wind speed","Blade angle"]].dropna()
            actual_data["Blade angle"]=actual_data["Blade angle"].apply(lambda x: ((x+180)%360)-180)
#            actual_data=actual_data.loc[actual_data["Blade angle"]<20]
            actual_data["Wind speed"]=actual_data["Wind speed"].apply(lambda x: round(x,1))  
            actual_data=actual_data.loc[actual_data["Wind speed"].isin(expected_relationship.index)]
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Blade angle"]>=expected_relationship.at[row["Wind speed"],"Avg"]-1.96*expected_relationship.at[row["Wind speed"],"Stdev"])&(row["Blade angle"]<=expected_relationship.at[row["Wind speed"],"Avg"]+1.96*expected_relationship.at[row["Wind speed"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
    
            kpis_figure_2 = go.Figure()
            
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='P50',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]+1.96*expected_relationship["Stdev"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P95/P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]-1.96*expected_relationship["Stdev"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Wind speed"],
                    y=actual_data["Blade angle"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
    
    
            kpis_figure_2.update_yaxes(title="Blade angle (°)",showgrid=False)
            kpis_figure_2.update_xaxes(title="Wind speed (m/s)",showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Pitch vs Speed Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure
















    elif active_cell["column_id"]=="RPM vs Speed KPI":
        




        try:

            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"RPM vs Speed"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Wind speed","Rotor speed"]].dropna()
            actual_data["Wind speed"]=actual_data["Wind speed"].apply(lambda x: round(x,1))  
            actual_data=actual_data.loc[actual_data["Wind speed"].isin(expected_relationship.index)]
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Rotor speed"]>=expected_relationship.at[row["Wind speed"],"Avg"]-1.96*expected_relationship.at[row["Wind speed"],"Stdev"])&(row["Rotor speed"]<=expected_relationship.at[row["Wind speed"],"Avg"]+1.96*expected_relationship.at[row["Wind speed"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
    
            kpis_figure_2 = go.Figure()
            
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='P50',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]+1.96*expected_relationship["Stdev"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P95/P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]-1.96*expected_relationship["Stdev"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Wind speed"],
                    y=actual_data["Rotor speed"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
    
    
            kpis_figure_2.update_yaxes(title="Rotor speed (RPM)",showgrid=False)
            kpis_figure_2.update_xaxes(title="Wind speed (m/s)",showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - RPM vs Speed Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure













    elif active_cell["column_id"]=="Power vs RPM KPI":
        
        try:
            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"Power vs RPM"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Rotor speed","Power"]].dropna()
            actual_data["Rotor speed"]=actual_data["Rotor speed"].apply(lambda x: round(x,1))  
            actual_data=actual_data.loc[actual_data["Rotor speed"].isin(expected_relationship.index)]
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Power"]>=expected_relationship.at[row["Rotor speed"],"Avg"]-1.96*expected_relationship.at[row["Rotor speed"],"Stdev"])&(row["Power"]<=expected_relationship.at[row["Rotor speed"],"Avg"]+1.96*expected_relationship.at[row["Rotor speed"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='P50',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]+1.96*expected_relationship["Stdev"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P95/P5',
                    x=expected_relationship.index,
                    y=expected_relationship["Avg"]-1.96*expected_relationship["Stdev"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Rotor speed"],
                    y=actual_data["Power"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
            kpis_figure_2.update_yaxes(title="Power (kW)",showgrid=False)
            kpis_figure_2.update_xaxes(title="Rotor speed (RPM)",showgrid=False)
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Power vs RPM Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure


















































    elif active_cell["column_id"]=="Dynamic yaw misalignment KPI":

        try:

            wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_dymb.at[wt_type,"Dynamic yaw misalignment"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            
            bin_value=0.5
            binnedtable=pd.DataFrame()
            tempnewdf=actual_data[["Wind speed","Yaw misalignment"]].dropna()
            tempnewdf["bin"]=(tempnewdf["Wind speed"]-(bin_value/2))/bin_value
            tempnewdf["bin"]=tempnewdf["bin"].astype("int64")
            ultratempone=tempnewdf[["bin","Wind speed"]]
            ultratemptwo=tempnewdf[["bin","Yaw misalignment"]]
            tempbinnedtable1=ultratempone.groupby(["bin"]).mean()
            tempbinnedtable2=ultratemptwo.groupby(["bin"]).apply(median_angle)
            tempnewdf2=pd.concat([tempbinnedtable1, tempbinnedtable2[["Yaw misalignment"]]], axis=1)
            tempnewdf2=tempnewdf2.rename(columns={"Yaw misalignment":'Avg'})
            tempbinnedtable3=ultratempone.groupby(["bin"]).mean()
            tempbinnedtable4=ultratemptwo.groupby(["bin"]).std()
            tempnewdf3 = pd.concat([tempbinnedtable3,tempbinnedtable4], axis=1)
            tempnewdf3=tempnewdf3.rename(columns={"Yaw misalignment":'Stdev'})
            tempnewdf4 = pd.concat([tempnewdf2,tempnewdf3], axis=1)
            tempbinnedtable5=ultratempone.groupby(["bin"]).mean()
            tempbinnedtable6=ultratemptwo.groupby(["bin"]).count()
            tempnewdf5 = pd.concat([tempbinnedtable5,tempbinnedtable6], axis=1)
            tempnewdf5=tempnewdf5.rename(columns={"Yaw misalignment":'Count'})
            tempnewdf6 = pd.concat([tempnewdf4,tempnewdf5], axis=1)
            tempnewdf6=tempnewdf6.loc[tempnewdf6["Count"]>25]
            tempnewdf4=tempnewdf6.drop(columns=["Count"])
            tempnewdf4 = tempnewdf4.loc[:,~tempnewdf4.columns.duplicated()]
            tempnewdf4.index=tempnewdf4["Wind speed"]
            if tempnewdf4.empty==False:
                steps=np.around(np.arange(0,tempnewdf["Wind speed"].max(),0.1),1).tolist()
                steps_tmp=pd.DataFrame(index=steps,columns=tempnewdf4.columns)
                tempnewdf4=tempnewdf4._append(steps_tmp)
                tempnewdf4.sort_index(inplace=True)
                tempnewdf4=tempnewdf4.interpolate(method="index")
                tempnewdf4=tempnewdf4.loc[steps]
                tempnewdf4=tempnewdf4.dropna()
                tempnewdf4=tempnewdf4.loc[~tempnewdf4.index.duplicated(keep='first')]
            binnedtable = tempnewdf4[["Avg","Stdev"]]
            binnedtable.sort_index(inplace=True)
            actual_data["HiLim"]=actual_data.apply(lambda row: binnedtable.at[round(row["Wind speed"],1),"Avg"]+1.96*binnedtable.at[round(row["Wind speed"],1),"Stdev"] if (round(row["Wind speed"],1) in binnedtable.index and pd.isnull(row["Wind speed"])==False) else np.nan, axis=1)
            actual_data["LoLim"]=actual_data.apply(lambda row: binnedtable.at[round(row["Wind speed"],1),"Avg"]-1.96*binnedtable.at[round(row["Wind speed"],1),"Stdev"] if (round(row["Wind speed"],1) in binnedtable.index and pd.isnull(row["Wind speed"])==False) else np.nan, axis=1)
            actual_data["Filtered yaw misalignment"]=actual_data.apply(lambda row: row["Yaw misalignment"] if (row["Yaw misalignment"]>row["LoLim"] and row["Yaw misalignment"]<row["HiLim"]) else np.nan, axis=1)
            actual_data=actual_data[["Wind speed","Filtered yaw misalignment"]].dropna()
            actual_data["flag"]=actual_data.apply(lambda row: "red" if (round(row["Wind speed"],1) in expected_relationship.index and (row["Filtered yaw misalignment"]>expected_relationship.at[round(row["Wind speed"],1),"Max"] or row["Filtered yaw misalignment"]<expected_relationship.at[round(row["Wind speed"],1),"Min"])) else "black",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='0',
                    x=expected_relationship.index,
                    y=[0 for i in expected_relationship.index],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='Expected maximum dynamic yaw misalignment',
                    x=expected_relationship.index,
                    y=expected_relationship["Max"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))



        

            kpis_figure_2.add_trace(go.Scatter(
                    name='Expected bounds',
                    x=expected_relationship.index,
                    y=expected_relationship["Min"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))



    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Wind speed"],
                    y=actual_data["Filtered yaw misalignment"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
    
    
            kpis_figure_2.update_yaxes(title="Dynamic yaw misalignment (°)",showgrid=False)
            kpis_figure_2.update_xaxes(title="Wind speed (m/s)",showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Dynamic yaw misalignment analysis',x=0.5))
            figure_appearance={'display': 'block'}




        
        except:
            kpis_figure_2=default_figure



        
    else:
        kpis_figure_2=default_figure

    
    
    return kpis_figure_2,figure_appearance




























































@app.callback(
        [Output('reliability-kpis-graph-2', 'figure'),
         Output("reliability-kpis-graph-2-container","style")
         ],[Input('reliability-kpis-datatable', 'active_cell'),
         Input('reliability-kpis-graph','clickData')])
def reliability_kpis_graph_2(active_cell,clickData):
    
    selected_wt=current_kpis["Wind turbine"].iloc[active_cell["row_id"]]

    if clickData is not None:
        selected_time=clickData['points'][0]['x']
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)
        selected_time=selected_time.strftime("%B %Y")

    graph_month=dt.strptime(selected_time,'%B %Y').strftime("%B")
    graph_year=dt.strptime(selected_time,'%B %Y').year
    figure_appearance={'display': 'none'}


    if active_cell["column_id"]=="Aerodynamic rotor imbalance KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Aerodynamic rotor imbalance data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            
            confidence_intervals=actual_data[["Rounded rotor speed","p5","p10","p25","p75","p90","p95"]]
            confidence_intervals.index=confidence_intervals["Rounded rotor speed"]
            confidence_intervals=confidence_intervals.loc[~confidence_intervals.index.duplicated(keep='first')]
            confidence_intervals.sort_index(inplace=True)
            confidence_intervals.index=range(len(confidence_intervals.index))
            new_row = {"Rounded rotor speed":actual_data["Rotor speed"].max(),"p5":confidence_intervals["p5"].iloc[-1],"p10":confidence_intervals["p10"].iloc[-1],"p25":confidence_intervals["p25"].iloc[-1],"p75":confidence_intervals["p75"].iloc[-1],"p90":confidence_intervals["p90"].iloc[-1],"p95":confidence_intervals["p95"].iloc[-1]}
            confidence_intervals = confidence_intervals._append(new_row,ignore_index=True)
            new_row = {"Rounded rotor speed":actual_data["Rotor speed"].min(),"p5":confidence_intervals["p5"].iloc[0],"p10":confidence_intervals["p10"].iloc[0],"p25":confidence_intervals["p25"].iloc[0],"p75":confidence_intervals["p75"].iloc[0],"p90":confidence_intervals["p90"].iloc[0],"p95":confidence_intervals["p95"].iloc[0]}
            confidence_intervals = confidence_intervals._append(new_row,ignore_index=True)
            confidence_intervals.index=confidence_intervals["Rounded rotor speed"]
            confidence_intervals.sort_index(inplace=True)

    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data["Rotor speed"],
                    y=actual_data["Predicted"],
                    mode='markers',
                    opacity=0.7,
                    marker={"color":"rgb(31, 119, 180)","size":3},
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data["Rotor speed"],
                    y=actual_data["Actual"],
                    mode='markers',
                    opacity=0.7,
                    marker={"color":"black","size":3},
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data["Rotor speed"],
                    y=actual_data["Residuals"],
                    mode='markers',
                    opacity=0.7,
                    marker={"color":"grey","size":3},
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p25"],
                    mode='lines',
                    line=dict(color="Yellow",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p10"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P5/P95',
                    legendgroup = 'f',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p5"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p75"],
                    mode='lines',
                    line=dict(color="Yellow",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p90"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P5/P95',
                    legendgroup = 'f',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p95"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

    
            kpis_figure_2.update_yaxes(title="Normal tower acceleration (mm/ss)",showgrid=False)
            kpis_figure_2.update_xaxes(title="Rotor speed (RPM)",showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Aerodynamic rotor imbalance Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure





    elif active_cell["column_id"]=="Mass rotor imbalance KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Mass rotor imbalance data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            
            confidence_intervals=actual_data[["Rounded rotor speed","p5","p10","p25","p75","p90","p95"]]
            confidence_intervals.index=confidence_intervals["Rounded rotor speed"]
            confidence_intervals=confidence_intervals.loc[~confidence_intervals.index.duplicated(keep='first')]
            confidence_intervals.sort_index(inplace=True)
            confidence_intervals.index=range(len(confidence_intervals.index))
            new_row = {"Rounded rotor speed":actual_data["Rotor speed"].max(),"p5":confidence_intervals["p5"].iloc[-1],"p10":confidence_intervals["p10"].iloc[-1],"p25":confidence_intervals["p25"].iloc[-1],"p75":confidence_intervals["p75"].iloc[-1],"p90":confidence_intervals["p90"].iloc[-1],"p95":confidence_intervals["p95"].iloc[-1]}
            confidence_intervals = confidence_intervals._append(new_row,ignore_index=True)
            new_row = {"Rounded rotor speed":actual_data["Rotor speed"].min(),"p5":confidence_intervals["p5"].iloc[0],"p10":confidence_intervals["p10"].iloc[0],"p25":confidence_intervals["p25"].iloc[0],"p75":confidence_intervals["p75"].iloc[0],"p90":confidence_intervals["p90"].iloc[0],"p95":confidence_intervals["p95"].iloc[0]}
            confidence_intervals = confidence_intervals._append(new_row,ignore_index=True)
            confidence_intervals.index=confidence_intervals["Rounded rotor speed"]
            confidence_intervals.sort_index(inplace=True)

    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data["Rotor speed"],
                    y=actual_data["Predicted"],
                    mode='markers',
                    opacity=0.7,
                    marker={"color":"rgb(31, 119, 180)","size":3},
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data["Rotor speed"],
                    y=actual_data["Actual"],
                    mode='markers',
                    opacity=0.7,
                    marker={"color":"black","size":3},
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data["Rotor speed"],
                    y=actual_data["Residuals"],
                    mode='markers',
                    opacity=0.7,
                    marker={"color":"grey","size":3},
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p25"],
                    mode='lines',
                    line=dict(color="Yellow",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p10"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P5/P95',
                    legendgroup = 'f',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p5"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p75"],
                    mode='lines',
                    line=dict(color="Yellow",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p90"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P5/P95',
                    legendgroup = 'f',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p95"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

    
            kpis_figure_2.update_yaxes(title="Lateral tower acceleration (mm/ss)",showgrid=False)
            kpis_figure_2.update_xaxes(title="Rotor speed (RPM)",showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Mass rotor imbalance Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure



    elif active_cell["column_id"]=="Global rotor imbalance KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Global rotor imbalance data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            
            confidence_intervals=actual_data[["Rounded rotor speed","p5","p10","p25","p75","p90","p95"]]
            confidence_intervals.index=confidence_intervals["Rounded rotor speed"]
            confidence_intervals=confidence_intervals.loc[~confidence_intervals.index.duplicated(keep='first')]
            confidence_intervals.sort_index(inplace=True)
            confidence_intervals.index=range(len(confidence_intervals.index))
            new_row = {"Rounded rotor speed":actual_data["Rotor speed"].max(),"p5":confidence_intervals["p5"].iloc[-1],"p10":confidence_intervals["p10"].iloc[-1],"p25":confidence_intervals["p25"].iloc[-1],"p75":confidence_intervals["p75"].iloc[-1],"p90":confidence_intervals["p90"].iloc[-1],"p95":confidence_intervals["p95"].iloc[-1]}
            confidence_intervals = confidence_intervals._append(new_row,ignore_index=True)
            new_row = {"Rounded rotor speed":actual_data["Rotor speed"].min(),"p5":confidence_intervals["p5"].iloc[0],"p10":confidence_intervals["p10"].iloc[0],"p25":confidence_intervals["p25"].iloc[0],"p75":confidence_intervals["p75"].iloc[0],"p90":confidence_intervals["p90"].iloc[0],"p95":confidence_intervals["p95"].iloc[0]}
            confidence_intervals = confidence_intervals._append(new_row,ignore_index=True)
            confidence_intervals.index=confidence_intervals["Rounded rotor speed"]
            confidence_intervals.sort_index(inplace=True)

    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data["Rotor speed"],
                    y=actual_data["Predicted"],
                    mode='markers',
                    opacity=0.7,
                    marker={"color":"rgb(31, 119, 180)","size":3},
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data["Rotor speed"],
                    y=actual_data["Actual"],
                    mode='markers',
                    opacity=0.7,
                    marker={"color":"black","size":3},
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data["Rotor speed"],
                    y=actual_data["Residuals"],
                    mode='markers',
                    opacity=0.7,
                    marker={"color":"grey","size":3},
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p25"],
                    mode='lines',
                    line=dict(color="Yellow",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p10"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P5/P95',
                    legendgroup = 'f',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p5"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p75"],
                    mode='lines',
                    line=dict(color="Yellow",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p90"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P5/P95',
                    legendgroup = 'f',
                    x=confidence_intervals.index,
                    y=confidence_intervals["p95"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

    
            kpis_figure_2.update_yaxes(title="Combined tower acceleration (mm/ss)",showgrid=False)
            kpis_figure_2.update_xaxes(title="Rotor speed (RPM)",showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Global rotor imbalance Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure







    elif active_cell["column_id"]=="Front bearing temperature KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Front bearing temperature data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data.index=actual_data["Timestamp"]
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data.index,
                    y=actual_data["Predicted"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data.index,
                    y=actual_data["Actual"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'black'
                            },
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data.index,
                    y=actual_data["Residuals"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'grey'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p25"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p10"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p75"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p90"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

            # try:
            #     events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
            #     events_data=pd.DataFrame(events_data)
            #     events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            #     events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            #     events_data=events_data.loc[events_data["Category"]=="stop"]
            #     events_data=events_data[["Start","End","Message"]]
            #     events_data.index=range(len(events_data.index))   
                
            #     if events_data.shape[0]<100:
    
            #         for event in range(len(events_data.index)):
                        
            #             min_y=min(actual_data["Actual"].min(),actual_data["Predicted"].min(),actual_data["Residuals"].min(),actual_data["p90"].min())
            #             max_y=max(actual_data["Actual"].max(),actual_data["Predicted"].max(),actual_data["Residuals"].max(),actual_data["p10"].max())
                        
            #             kpis_figure_2.add_shape(
            #                 type="rect",
            #                 x0=events_data["Start"].iloc[event],
            #                 y0=min_y,
            #                 x1=events_data["End"].iloc[event],
            #                 y1=max_y,
            #                 fillcolor='orange',
            #                 line_color='orange',
            #                 opacity=0.5
            #                 )
                    
            #             kpis_figure_2.add_trace(
            #                 go.Scatter(
            #                     x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
            #                     y=[min_y,max_y,max_y,min_y,min_y], 
            #                     legendgroup = 'f',
            #                     fill="toself",
            #                     mode='lines',
            #                     name='',
            #                     text=str(events_data["Message"].iloc[event]),
            #                     opacity=0,
            #                     showlegend=False,
            #                     hoverlabel=dict(bgcolor="white")
            #                     )
            #                 )
            # except:
            #     pass




    
            kpis_figure_2.update_yaxes(title="Temperature (°C)",showgrid=False)
            kpis_figure_2.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Front bearing temperature Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure









    elif active_cell["column_id"]=="Rear bearing temperature KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Rear bearing temperature data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data.index=actual_data["Timestamp"]
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data.index,
                    y=actual_data["Predicted"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data.index,
                    y=actual_data["Actual"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'black'
                            },
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data.index,
                    y=actual_data["Residuals"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'grey'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p25"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p10"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p75"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p90"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

            # try:
            #     events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
            #     events_data=pd.DataFrame(events_data)
            #     events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            #     events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            #     events_data=events_data.loc[events_data["Category"]=="stop"]
            #     events_data=events_data[["Start","End","Message"]]
            #     events_data.index=range(len(events_data.index))    
                
            #     if events_data.shape[0]<100:
    
            #         for event in range(len(events_data.index)):
                        
            #             min_y=min(actual_data["Actual"].min(),actual_data["Predicted"].min(),actual_data["Residuals"].min(),actual_data["p90"].min())
            #             max_y=max(actual_data["Actual"].max(),actual_data["Predicted"].max(),actual_data["Residuals"].max(),actual_data["p10"].max())
                        
            #             kpis_figure_2.add_shape(
            #                 type="rect",
            #                 x0=events_data["Start"].iloc[event],
            #                 y0=min_y,
            #                 x1=events_data["End"].iloc[event],
            #                 y1=max_y,
            #                 fillcolor='orange',
            #                 line_color='orange',
            #                 opacity=0.5
            #                 )
                    
            #             kpis_figure_2.add_trace(
            #                 go.Scatter(
            #                     x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
            #                     y=[min_y,max_y,max_y,min_y,min_y], 
            #                     legendgroup = 'f',
            #                     fill="toself",
            #                     mode='lines',
            #                     name='',
            #                     text=str(events_data["Message"].iloc[event]),
            #                     opacity=0,
            #                     showlegend=False,
            #                     hoverlabel=dict(bgcolor="white")
            #                     )
            #                 )
            # except:
            #     pass




    
            kpis_figure_2.update_yaxes(title="Temperature (°C)",showgrid=False)
            kpis_figure_2.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Rear bearing temperature Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure













    elif active_cell["column_id"]=="Rotor temperature KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Rotor temperature data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data.index=actual_data["Timestamp"]
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data.index,
                    y=actual_data["Predicted"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data.index,
                    y=actual_data["Actual"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'black'
                            },
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data.index,
                    y=actual_data["Residuals"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'grey'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p25"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p10"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p75"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p90"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

            # try:
            #     events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
            #     events_data=pd.DataFrame(events_data)
            #     events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            #     events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            #     events_data=events_data.loc[events_data["Category"]=="stop"]
            #     events_data=events_data[["Start","End","Message"]]
            #     events_data.index=range(len(events_data.index))        
    
            #     if events_data.shape[0]<100:

            #         for event in range(len(events_data.index)):
                        
            #             min_y=min(actual_data["Actual"].min(),actual_data["Predicted"].min(),actual_data["Residuals"].min(),actual_data["p90"].min())
            #             max_y=max(actual_data["Actual"].max(),actual_data["Predicted"].max(),actual_data["Residuals"].max(),actual_data["p10"].max())
                        
            #             kpis_figure_2.add_shape(
            #                 type="rect",
            #                 x0=events_data["Start"].iloc[event],
            #                 y0=min_y,
            #                 x1=events_data["End"].iloc[event],
            #                 y1=max_y,
            #                 fillcolor='orange',
            #                 line_color='orange',
            #                 opacity=0.5
            #                 )
                    
            #             kpis_figure_2.add_trace(
            #                 go.Scatter(
            #                     x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
            #                     y=[min_y,max_y,max_y,min_y,min_y], 
            #                     legendgroup = 'f',
            #                     fill="toself",
            #                     mode='lines',
            #                     name='',
            #                     text=str(events_data["Message"].iloc[event]),
            #                     opacity=0,
            #                     showlegend=False,
            #                     hoverlabel=dict(bgcolor="white")
            #                     )
            #                 )
            # except:
            #     pass




    
            kpis_figure_2.update_yaxes(title="Temperature (°C)",showgrid=False)
            kpis_figure_2.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Rotor temperature Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure
















    elif active_cell["column_id"]=="Stator temperature KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Stator temperature data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data.index=actual_data["Timestamp"]
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data.index,
                    y=actual_data["Predicted"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data.index,
                    y=actual_data["Actual"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'black'
                            },
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data.index,
                    y=actual_data["Residuals"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'grey'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p25"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p10"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p75"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p90"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

            # try:
            #     events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
            #     events_data=pd.DataFrame(events_data)
            #     events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            #     events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            #     events_data=events_data.loc[events_data["Category"]=="stop"]
            #     events_data=events_data[["Start","End","Message"]]
            #     events_data.index=range(len(events_data.index))        
    
            #     if events_data.shape[0]<100:

            #         for event in range(len(events_data.index)):
                        
            #             min_y=min(actual_data["Actual"].min(),actual_data["Predicted"].min(),actual_data["Residuals"].min(),actual_data["p90"].min())
            #             max_y=max(actual_data["Actual"].max(),actual_data["Predicted"].max(),actual_data["Residuals"].max(),actual_data["p10"].max())
                        
            #             kpis_figure_2.add_shape(
            #                 type="rect",
            #                 x0=events_data["Start"].iloc[event],
            #                 y0=min_y,
            #                 x1=events_data["End"].iloc[event],
            #                 y1=max_y,
            #                 fillcolor='orange',
            #                 line_color='orange',
            #                 opacity=0.5
            #                 )
                    
            #             kpis_figure_2.add_trace(
            #                 go.Scatter(
            #                     x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
            #                     y=[min_y,max_y,max_y,min_y,min_y], 
            #                     legendgroup = 'f',
            #                     fill="toself",
            #                     mode='lines',
            #                     name='',
            #                     text=str(events_data["Message"].iloc[event]),
            #                     opacity=0,
            #                     showlegend=False,
            #                     hoverlabel=dict(bgcolor="white")
            #                     )
            #                 )
            # except:
            #     pass




    
            kpis_figure_2.update_yaxes(title="Temperature (°C)",showgrid=False)
            kpis_figure_2.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Stator temperature Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure







    elif active_cell["column_id"]=="Gearbox HSS bearing temperature KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Gearbox HSS bearing temperature data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data.index=actual_data["Timestamp"]
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data.index,
                    y=actual_data["Predicted"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data.index,
                    y=actual_data["Actual"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'black'
                            },
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data.index,
                    y=actual_data["Residuals"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'grey'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p25"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            print('8958')

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p10"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p75"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p90"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

            # try:
            #     events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
            #     events_data=pd.DataFrame(events_data)
            #     events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            #     events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            #     events_data=events_data.loc[events_data["Category"]=="stop"]
            #     events_data=events_data[["Start","End","Message"]]
            #     events_data.index=range(len(events_data.index))        
    
            #     if events_data.shape[0]<100:

            #         for event in range(len(events_data.index)):
                        
            #             min_y=min(actual_data["Actual"].min(),actual_data["Predicted"].min(),actual_data["Residuals"].min(),actual_data["p90"].min())
            #             max_y=max(actual_data["Actual"].max(),actual_data["Predicted"].max(),actual_data["Residuals"].max(),actual_data["p10"].max())
                        
            #             kpis_figure_2.add_shape(
            #                 type="rect",
            #                 x0=events_data["Start"].iloc[event],
            #                 y0=min_y,
            #                 x1=events_data["End"].iloc[event],
            #                 y1=max_y,
            #                 fillcolor='orange',
            #                 line_color='orange',
            #                 opacity=0.5
            #                 )
                    
            #             kpis_figure_2.add_trace(
            #                 go.Scatter(
            #                     x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
            #                     y=[min_y,max_y,max_y,min_y,min_y], 
            #                     legendgroup = 'f',
            #                     fill="toself",
            #                     mode='lines',
            #                     name='',
            #                     text=str(events_data["Message"].iloc[event]),
            #                     opacity=0,
            #                     showlegend=False,
            #                     hoverlabel=dict(bgcolor="white")
            #                     )
            #                 )
            # except:
            #     pass




    
            kpis_figure_2.update_yaxes(title="Temperature (°C)",showgrid=False)
            kpis_figure_2.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Gearbox HSS bearing temperature Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure






    elif active_cell["column_id"]=="Gearbox IMS/LSS bearing temperature KPI":
        


        try:
            print('9057')
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Gearbox IMS/LSS bearing temperature data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data.index=actual_data["Timestamp"]
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data.index,
                    y=actual_data["Predicted"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data.index,
                    y=actual_data["Actual"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'black'
                            },
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data.index,
                    y=actual_data["Residuals"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'grey'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p25"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p10"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p75"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p90"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

            # try:
            #     events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
            #     events_data=pd.DataFrame(events_data)
            #     events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            #     events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            #     events_data=events_data.loc[events_data["Category"]=="stop"]
            #     events_data=events_data[["Start","End","Message"]]
            #     events_data.index=range(len(events_data.index))        
    
            #     if events_data.shape[0]<100:

            #         for event in range(len(events_data.index)):
                        
            #             min_y=min(actual_data["Actual"].min(),actual_data["Predicted"].min(),actual_data["Residuals"].min(),actual_data["p90"].min())
            #             max_y=max(actual_data["Actual"].max(),actual_data["Predicted"].max(),actual_data["Residuals"].max(),actual_data["p10"].max())
                        
            #             kpis_figure_2.add_shape(
            #                 type="rect",
            #                 x0=events_data["Start"].iloc[event],
            #                 y0=min_y,
            #                 x1=events_data["End"].iloc[event],
            #                 y1=max_y,
            #                 fillcolor='orange',
            #                 line_color='orange',
            #                 opacity=0.5
            #                 )
                    
            #             kpis_figure_2.add_trace(
            #                 go.Scatter(
            #                     x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
            #                     y=[min_y,max_y,max_y,min_y,min_y], 
            #                     legendgroup = 'f',
            #                     fill="toself",
            #                     mode='lines',
            #                     name='',
            #                     text=str(events_data["Message"].iloc[event]),
            #                     opacity=0,
            #                     showlegend=False,
            #                     hoverlabel=dict(bgcolor="white")
            #                     )
            #                 )
            # except:
            #     pass




    
            kpis_figure_2.update_yaxes(title="Temperature (°C)",showgrid=False)
            kpis_figure_2.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Gearbox IMS/LSS bearing temperature Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure








    elif active_cell["column_id"]=="Generator bearing front temperature KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Generator bearing front temperature data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data.index=actual_data["Timestamp"]
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data.index,
                    y=actual_data["Predicted"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data.index,
                    y=actual_data["Actual"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'black'
                            },
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data.index,
                    y=actual_data["Residuals"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'grey'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p25"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p10"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p75"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p90"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

            # try:
            #     events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
            #     events_data=pd.DataFrame(events_data)
            #     events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            #     events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            #     events_data=events_data.loc[events_data["Category"]=="stop"]
            #     events_data=events_data[["Start","End","Message"]]
            #     events_data.index=range(len(events_data.index))        
    
            #     if events_data.shape[0]<100:
                    
            #         for event in range(len(events_data.index)):
                        
            #             min_y=min(actual_data["Actual"].min(),actual_data["Predicted"].min(),actual_data["Residuals"].min(),actual_data["p90"].min())
            #             max_y=max(actual_data["Actual"].max(),actual_data["Predicted"].max(),actual_data["Residuals"].max(),actual_data["p10"].max())
                        
            #             kpis_figure_2.add_shape(
            #                 type="rect",
            #                 x0=events_data["Start"].iloc[event],
            #                 y0=min_y,
            #                 x1=events_data["End"].iloc[event],
            #                 y1=max_y,
            #                 fillcolor='orange',
            #                 line_color='orange',
            #                 opacity=0.5
            #                 )
                    
            #             kpis_figure_2.add_trace(
            #                 go.Scatter(
            #                     x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
            #                     y=[min_y,max_y,max_y,min_y,min_y], 
            #                     legendgroup = 'f',
            #                     fill="toself",
            #                     mode='lines',
            #                     name='',
            #                     text=str(events_data["Message"].iloc[event]),
            #                     opacity=0,
            #                     showlegend=False,
            #                     hoverlabel=dict(bgcolor="white")
            #                     )
            #                 )
            # except:
            #     pass




    
            kpis_figure_2.update_yaxes(title="Temperature (°C)",showgrid=False)
            kpis_figure_2.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Generator bearing front temperature Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure








    elif active_cell["column_id"]=="Generator bearing rear temperature KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Generator bearing rear temperature data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data.index=actual_data["Timestamp"]
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data.index,
                    y=actual_data["Predicted"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data.index,
                    y=actual_data["Actual"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'black'
                            },
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data.index,
                    y=actual_data["Residuals"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'grey'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p25"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p10"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p75"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p90"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

            # try:
            #     events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
            #     events_data=pd.DataFrame(events_data)
            #     events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            #     events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            #     events_data=events_data.loc[events_data["Category"]=="stop"]
            #     events_data=events_data[["Start","End","Message"]]
            #     events_data.index=range(len(events_data.index))        
    
            #     if events_data.shape[0]<100:

            #         for event in range(len(events_data.index)):
                        
            #             min_y=min(actual_data["Actual"].min(),actual_data["Predicted"].min(),actual_data["Residuals"].min(),actual_data["p90"].min())
            #             max_y=max(actual_data["Actual"].max(),actual_data["Predicted"].max(),actual_data["Residuals"].max(),actual_data["p10"].max())
                        
            #             kpis_figure_2.add_shape(
            #                 type="rect",
            #                 x0=events_data["Start"].iloc[event],
            #                 y0=min_y,
            #                 x1=events_data["End"].iloc[event],
            #                 y1=max_y,
            #                 fillcolor='orange',
            #                 line_color='orange',
            #                 opacity=0.5
            #                 )
                    
            #             kpis_figure_2.add_trace(
            #                 go.Scatter(
            #                     x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
            #                     y=[min_y,max_y,max_y,min_y,min_y], 
            #                     legendgroup = 'f',
            #                     fill="toself",
            #                     mode='lines',
            #                     name='',
            #                     text=str(events_data["Message"].iloc[event]),
            #                     opacity=0,
            #                     showlegend=False,
            #                     hoverlabel=dict(bgcolor="white")
            #                     )
            #                 )
            # except:
            #     pass




    
            kpis_figure_2.update_yaxes(title="Temperature (°C)",showgrid=False)
            kpis_figure_2.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Generator bearing rear temperature Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure







    elif active_cell["column_id"]=="Main bearing temperature KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Main bearing temperature data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data.index=actual_data["Timestamp"]
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data.index,
                    y=actual_data["Predicted"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data.index,
                    y=actual_data["Actual"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'black'
                            },
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data.index,
                    y=actual_data["Residuals"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'grey'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p25"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p10"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p75"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p90"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

            # try:
            #     events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
            #     events_data=pd.DataFrame(events_data)
            #     events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            #     events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            #     events_data=events_data.loc[events_data["Category"]=="stop"]
            #     events_data=events_data[["Start","End","Message"]]
            #     events_data.index=range(len(events_data.index))        
    
            #     if events_data.shape[0]<100:

            #         for event in range(len(events_data.index)):
                        
            #             min_y=min(actual_data["Actual"].min(),actual_data["Predicted"].min(),actual_data["Residuals"].min(),actual_data["p90"].min())
            #             max_y=max(actual_data["Actual"].max(),actual_data["Predicted"].max(),actual_data["Residuals"].max(),actual_data["p10"].max())
                        
            #             kpis_figure_2.add_shape(
            #                 type="rect",
            #                 x0=events_data["Start"].iloc[event],
            #                 y0=min_y,
            #                 x1=events_data["End"].iloc[event],
            #                 y1=max_y,
            #                 fillcolor='orange',
            #                 line_color='orange',
            #                 opacity=0.5
            #                 )
                    
            #             kpis_figure_2.add_trace(
            #                 go.Scatter(
            #                     x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
            #                     y=[min_y,max_y,max_y,min_y,min_y], 
            #                     legendgroup = 'f',
            #                     fill="toself",
            #                     mode='lines',
            #                     name='',
            #                     text=str(events_data["Message"].iloc[event]),
            #                     opacity=0,
            #                     showlegend=False,
            #                     hoverlabel=dict(bgcolor="white")
            #                     )
            #                 )
            # except:
            #     pass




    
            kpis_figure_2.update_yaxes(title="Temperature (°C)",showgrid=False)
            kpis_figure_2.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Main bearing temperature Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure






    elif active_cell["column_id"]=="Metal particle count KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Metal particle count data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data.index=actual_data["Timestamp"]
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    x=actual_data.index,
                    y=actual_data["Predicted"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    x=actual_data.index,
                    y=actual_data["Actual"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'black'
                            },
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    x=actual_data.index,
                    y=actual_data["Residuals"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'grey'
                            },
                    showlegend=True
                ))
    


            # try:
            #     events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
            #     events_data=pd.DataFrame(events_data)
            #     events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            #     events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            #     events_data=events_data.loc[events_data["Category"]=="stop"]
            #     events_data=events_data[["Start","End","Message"]]
            #     events_data.index=range(len(events_data.index))        
    
            #     if events_data.shape[0]<100:

            #         for event in range(len(events_data.index)):
                        
            #             min_y=min(actual_data["Actual"].min(),actual_data["Predicted"].min(),actual_data["Residuals"].min().min())
            #             max_y=max(actual_data["Actual"].max(),actual_data["Predicted"].max(),actual_data["Residuals"].max(),actual_data["p10"].max())
                        
            #             kpis_figure_2.add_shape(
            #                 type="rect",
            #                 x0=events_data["Start"].iloc[event],
            #                 y0=min_y,
            #                 x1=events_data["End"].iloc[event],
            #                 y1=max_y,
            #                 fillcolor='orange',
            #                 line_color='orange',
            #                 opacity=0.5
            #                 )
                    
            #             kpis_figure_2.add_trace(
            #                 go.Scatter(
            #                     x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
            #                     y=[min_y,max_y,max_y,min_y,min_y], 
            #                     fill="toself",
            #                     mode='lines',
            #                     name='',
            #                     text=str(events_data["Message"].iloc[event]),
            #                     opacity=0,
            #                     showlegend=False,
            #                     hoverlabel=dict(bgcolor="white")
            #                     )
            #                 )
            # except:
            #     pass




    
            kpis_figure_2.update_yaxes(title="Count (-)",showgrid=False)
            kpis_figure_2.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Metal particle count Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure







    elif active_cell["column_id"]=="Gearbox oil temperature KPI":
        


        try:
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Gearbox oil temperature data"].iloc[0]
            actual_data=actual_data.replace("nan",'np.nan')
            actual_data=eval(actual_data)
            actual_data=pd.DataFrame(actual_data)
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data.index=actual_data["Timestamp"]
    
            kpis_figure_2 = go.Figure()
            
            kpis_figure_2.add_trace(go.Scatter(
                    name='Predicted',
                    legendgroup = 'a',
                    x=actual_data.index,
                    y=actual_data["Predicted"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='Actual',
                    legendgroup = 'b',
                    x=actual_data.index,
                    y=actual_data["Actual"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'black'
                            },
                    showlegend=True
                ))
    

            kpis_figure_2.add_trace(go.Scatter(
                    name='Residuals',
                    legendgroup = 'c',
                    x=actual_data.index,
                    y=actual_data["Residuals"],
                    mode='lines',
                    connectgaps=False,
                    line={
                            'color':'grey'
                            },
                    showlegend=True
                ))
    
            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p25"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p10"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=True
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P25/P75',
                    legendgroup = 'd',
                    x=actual_data.index,
                    y=actual_data["p75"],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))

            kpis_figure_2.add_trace(go.Scatter(
                    name='P10/P90',
                    legendgroup = 'e',
                    x=actual_data.index,
                    y=actual_data["p90"],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))

            # try:
            #     events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
            #     events_data=pd.DataFrame(events_data)
            #     events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            #     events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            #     events_data=events_data.loc[events_data["Category"]=="stop"]
            #     events_data=events_data[["Start","End","Message"]]
            #     events_data.index=range(len(events_data.index))        
    
            #     if events_data.shape[0]<100:

            #         for event in range(len(events_data.index)):
                        
            #             min_y=min(actual_data["Actual"].min(),actual_data["Predicted"].min(),actual_data["Residuals"].min(),actual_data["p90"].min())
            #             max_y=max(actual_data["Actual"].max(),actual_data["Predicted"].max(),actual_data["Residuals"].max(),actual_data["p10"].max())
                        
            #             kpis_figure_2.add_shape(
            #                 type="rect",
            #                 x0=events_data["Start"].iloc[event],
            #                 y0=min_y,
            #                 x1=events_data["End"].iloc[event],
            #                 y1=max_y,
            #                 fillcolor='orange',
            #                 line_color='orange',
            #                 opacity=0.5
            #                 )
                    
            #             kpis_figure_2.add_trace(
            #                 go.Scatter(
            #                     x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
            #                     y=[min_y,max_y,max_y,min_y,min_y], 
            #                     legendgroup = 'f',
            #                     fill="toself",
            #                     mode='lines',
            #                     name='',
            #                     text=str(events_data["Message"].iloc[event]),
            #                     opacity=0,
            #                     showlegend=False,
            #                     hoverlabel=dict(bgcolor="white")
            #                     )
            #                 )
            # except:
            #     pass




    
            kpis_figure_2.update_yaxes(title="Temperature (°C)",showgrid=False)
            kpis_figure_2.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_2.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Gearbox oil temperature Analysis',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_2=default_figure













        
    else:
        kpis_figure_2=default_figure

    
    
    return kpis_figure_2,figure_appearance












@app.callback(
        [Output('kpis-graph-3', 'figure'),
         Output("kpis-graph-3-container","style")
         ],[Input('kpis-datatable', 'active_cell'),
         Input('kpis-graph','clickData')])
def kpis_graph_3(active_cell,clickData):
    
    selected_wt=current_kpis["Wind turbine"].iloc[active_cell["row_id"]]

    if clickData is not None:
        selected_time=clickData['points'][0]['x']
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)
        selected_time=selected_time.strftime("%B %Y")

    graph_month=dt.strptime(selected_time,'%B %Y').strftime("%B")
    graph_year=dt.strptime(selected_time,'%B %Y').year

    figure_appearance={'display': 'none'}


    if active_cell["column_id"]=="Static yaw misalignment KPI":

        try:
            
            months=list(set(kpis["Month"].tolist()))
            months=[dt.strptime(i,'%B %Y') for i in months]
            months=sorted(months)
            months=[i.strftime('%B %Y') for i in months]
            yaw_mis_distribution_over_time=pd.DataFrame(index=months,columns=["Uncertain","<2.5°","2.5° to 5°","5° to 7.5°",">7.5°"])
            for month in months:
                monthly_yaw_mis_data=kpis.loc[kpis["Month"]==month]
                monthly_yaw_mis_data=monthly_yaw_mis_data[["Static yaw misalignment best guess"]]
                value=monthly_yaw_mis_data["Static yaw misalignment best guess"].isnull().sum()
                yaw_mis_distribution_over_time.at[month,"Uncertain"]=value
                monthly_yaw_mis_data=monthly_yaw_mis_data.dropna()
                value=monthly_yaw_mis_data.loc[(monthly_yaw_mis_data["Static yaw misalignment best guess"]>=-2.5)&(monthly_yaw_mis_data["Static yaw misalignment best guess"]<=2.5)].count()
                yaw_mis_distribution_over_time.at[month,"<2.5°"]=value.iloc[0]
                value=monthly_yaw_mis_data.loc[((monthly_yaw_mis_data["Static yaw misalignment best guess"]>=-5)&(monthly_yaw_mis_data["Static yaw misalignment best guess"]<-2.5))|((monthly_yaw_mis_data["Static yaw misalignment best guess"]<=5)&(monthly_yaw_mis_data["Static yaw misalignment best guess"]>2.5))].count()
                yaw_mis_distribution_over_time.at[month,"2.5° to 5°"]=value.iloc[0]
                value=monthly_yaw_mis_data.loc[((monthly_yaw_mis_data["Static yaw misalignment best guess"]>=-7.5)&(monthly_yaw_mis_data["Static yaw misalignment best guess"]<-5))|((monthly_yaw_mis_data["Static yaw misalignment best guess"]<=7.5)&(monthly_yaw_mis_data["Static yaw misalignment best guess"]>5))].count()
                yaw_mis_distribution_over_time.at[month,"5° to 7.5°"]=value.iloc[0]
                value=monthly_yaw_mis_data.loc[((monthly_yaw_mis_data["Static yaw misalignment best guess"]<-7.5))|((monthly_yaw_mis_data["Static yaw misalignment best guess"]>7.5))].count()
                yaw_mis_distribution_over_time.at[month,">7.5°"]=value.iloc[0]

            
            kpis_figure_3 = go.Figure(data=[
                go.Bar(name="Uncertain", x=yaw_mis_distribution_over_time.index, y=yaw_mis_distribution_over_time["Uncertain"],marker=go.bar.Marker(color='black')),
                go.Bar(name="<2.5°", x=yaw_mis_distribution_over_time.index, y=yaw_mis_distribution_over_time["<2.5°"],marker=go.bar.Marker(color='green')),
                go.Bar(name="2.5° to 5°", x=yaw_mis_distribution_over_time.index, y=yaw_mis_distribution_over_time["2.5° to 5°"],marker=go.bar.Marker(color='orange')),
                go.Bar(name="5° to 7.5°", x=yaw_mis_distribution_over_time.index, y=yaw_mis_distribution_over_time["5° to 7.5°"],marker=go.bar.Marker(color='red')),
                go.Bar(name=">7.5°", x=yaw_mis_distribution_over_time.index, y=yaw_mis_distribution_over_time[">7.5°"],marker=go.bar.Marker(color='purple')),
            ])

            kpis_figure_3.update_layout(barmode='stack')

            kpis_figure_3.update_yaxes(title="Static yaw misalignment distribution (-)",showgrid=False)
            kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_3.update_layout(title=dict(text='Static yaw misalignment distribution over time',x=0.5))
            figure_appearance={'display': 'block'}

        except:
            kpis_figure_3=default_figure












    elif active_cell["column_id"]=="Dynamic yaw misalignment KPI":

        try:




            wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_dymb.at[wt_type,"Dynamic yaw misalignment"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            
            bin_value=0.5
            binnedtable=pd.DataFrame()
            tempnewdf=actual_data[["Wind speed","Yaw misalignment"]].dropna()
            tempnewdf["bin"]=(tempnewdf["Wind speed"]-(bin_value/2))/bin_value
            tempnewdf["bin"]=tempnewdf["bin"].astype("int64")
            ultratempone=tempnewdf[["bin","Wind speed"]]
            ultratemptwo=tempnewdf[["bin","Yaw misalignment"]]
            tempbinnedtable1=ultratempone.groupby(["bin"]).mean()
            tempbinnedtable2=ultratemptwo.groupby(["bin"]).apply(median_angle)
            tempnewdf2=pd.concat([tempbinnedtable1, tempbinnedtable2[["Yaw misalignment"]]], axis=1)
            tempnewdf2=tempnewdf2.rename(columns={"Yaw misalignment":'Avg'})
            tempbinnedtable3=ultratempone.groupby(["bin"]).mean()
            tempbinnedtable4=ultratemptwo.groupby(["bin"]).std()
            tempnewdf3 = pd.concat([tempbinnedtable3,tempbinnedtable4], axis=1)
            tempnewdf3=tempnewdf3.rename(columns={"Yaw misalignment":'Stdev'})
            tempnewdf4 = pd.concat([tempnewdf2,tempnewdf3], axis=1)
            tempbinnedtable5=ultratempone.groupby(["bin"]).mean()
            tempbinnedtable6=ultratemptwo.groupby(["bin"]).count()
            tempnewdf5 = pd.concat([tempbinnedtable5,tempbinnedtable6], axis=1)
            tempnewdf5=tempnewdf5.rename(columns={"Yaw misalignment":'Count'})
            tempnewdf6 = pd.concat([tempnewdf4,tempnewdf5], axis=1)
            tempnewdf6=tempnewdf6.loc[tempnewdf6["Count"]>25]
            tempnewdf4=tempnewdf6.drop(columns=["Count"])
            tempnewdf4 = tempnewdf4.loc[:,~tempnewdf4.columns.duplicated()]
            tempnewdf4.index=tempnewdf4["Wind speed"]
            if tempnewdf4.empty==False:
                steps=np.around(np.arange(0,tempnewdf["Wind speed"].max(),0.1),1).tolist()
                steps_tmp=pd.DataFrame(index=steps,columns=tempnewdf4.columns)
                tempnewdf4=tempnewdf4._append(steps_tmp)
                tempnewdf4.sort_index(inplace=True)
                tempnewdf4=tempnewdf4.interpolate(method="index")
                tempnewdf4=tempnewdf4.loc[steps]
                tempnewdf4=tempnewdf4.dropna()
                tempnewdf4=tempnewdf4.loc[~tempnewdf4.index.duplicated(keep='first')]
            binnedtable = tempnewdf4[["Avg","Stdev"]]
            binnedtable.sort_index(inplace=True)
            actual_data["HiLim"]=actual_data.apply(lambda row: binnedtable.at[round(row["Wind speed"],1),"Avg"]+1.96*binnedtable.at[round(row["Wind speed"],1),"Stdev"] if (round(row["Wind speed"],1) in binnedtable.index and pd.isnull(row["Wind speed"])==False) else np.nan, axis=1)
            actual_data["LoLim"]=actual_data.apply(lambda row: binnedtable.at[round(row["Wind speed"],1),"Avg"]-1.96*binnedtable.at[round(row["Wind speed"],1),"Stdev"] if (round(row["Wind speed"],1) in binnedtable.index and pd.isnull(row["Wind speed"])==False) else np.nan, axis=1)
            actual_data["Filtered yaw misalignment"]=actual_data.apply(lambda row: row["Yaw misalignment"] if (row["Yaw misalignment"]>row["LoLim"] and row["Yaw misalignment"]<row["HiLim"]) else np.nan, axis=1)
            actual_data=actual_data[["Wind speed","Filtered yaw misalignment","Timestamp"]].dropna()
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data["Wind speed"]=actual_data["Wind speed"].apply(lambda x: round(x,1))  
            actual_data=actual_data.loc[actual_data["Wind speed"].isin(expected_relationship.index)]
            actual_data["Expected dynamic yaw misalignment"]=0
            actual_data["Expected dynamic yaw misalignment high bound"]=actual_data["Wind speed"].apply(lambda x: expected_relationship.at[round(x,1),"Max"])
            actual_data["Expected dynamic yaw misalignment low bound"]=actual_data["Wind speed"].apply(lambda x: expected_relationship.at[round(x,1),"Min"])
            actual_data["flag"]=actual_data.apply(lambda row: "red" if (round(row["Wind speed"],1) in expected_relationship.index and (row["Filtered yaw misalignment"]>expected_relationship.at[round(row["Wind speed"],1),"Max"] or row["Filtered yaw misalignment"]<expected_relationship.at[round(row["Wind speed"],1),"Min"])) else "black",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
            kpis_figure_3 = go.Figure()
            
            
            kpis_figure_3.add_trace(go.Scatter(
                    name='Avg',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected dynamic yaw misalignment"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=False
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='Max',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected dynamic yaw misalignment high bound"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='Expected bounds',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected dynamic yaw misalignment low bound"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Timestamp"],
                    y=actual_data["Filtered yaw misalignment"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
            
            try:
                events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
                events_data=pd.DataFrame(events_data)
                events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
                events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
                events_data=events_data.loc[events_data["Category"]=="stop"]
                events_data=events_data[["Start","End","Message"]]
                events_data.index=range(len(events_data.index))        
    
                for event in range(len(events_data.index)):
                    
                    min_y=min(actual_data["Blade angle"].min(),actual_data["Expected dynamic yaw misalignment low bound"].min())
                    max_y=max(actual_data["Blade angle"].max(),actual_data["Expected dynamic yaw misalignment high bound"].max())
                    
                    kpis_figure_3.add_shape(
                        type="rect",
                        x0=events_data["Start"].iloc[event],
                        y0=min_y,
                        x1=events_data["End"].iloc[event],
                        y1=max_y,
                        fillcolor='orange',
                        line_color='orange',
                        opacity=0.5
                    )
                
                    kpis_figure_3.add_trace(
                        go.Scatter(
                            x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
                            y=[min_y,max_y,max_y,min_y,min_y], 
                            fill="toself",
                            mode='lines',
                            name='',
                            text=str(events_data["Message"].iloc[event]),
                            opacity=0,
                            showlegend=False,
                            hoverlabel=dict(bgcolor="white")
                            )
                        )


            except:
                pass
    
    
    
            kpis_figure_3.update_yaxes(title="Dynamic yaw misalignment (°)",showgrid=False)
            kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_3.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Dynamic yaw misalignment based on wind speed over time',x=0.5))
            figure_appearance={'display': 'block'}

        except:
            kpis_figure_3=default_figure



















    elif active_cell["column_id"]=="Pitch vs Power KPI":

        try:

            
            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"Pitch vs Power"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Power","Blade angle","Timestamp"]].dropna()
            actual_data["Blade angle"]=actual_data["Blade angle"].apply(lambda x: ((x+180)%360)-180)
#            actual_data=actual_data.loc[actual_data["Blade angle"]<20]
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data["Power"]=actual_data["Power"].apply(lambda x: round(x,0))  
            actual_data=actual_data.loc[actual_data["Power"].isin(expected_relationship.index)]
            actual_data["Expected blade angle"]=actual_data["Power"].apply(lambda x: expected_relationship.at[x,"Avg"])
            actual_data["Expected blade angle high bound"]=actual_data["Power"].apply(lambda x: expected_relationship.at[x,"Avg"]+1.96*expected_relationship.at[x,"Stdev"])
            actual_data["Expected blade angle low bound"]=actual_data["Power"].apply(lambda x: expected_relationship.at[x,"Avg"]-1.96*expected_relationship.at[x,"Stdev"])
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Blade angle"]>=expected_relationship.at[row["Power"],"Avg"]-1.96*expected_relationship.at[row["Power"],"Stdev"])&(row["Blade angle"]<=expected_relationship.at[row["Power"],"Avg"]+1.96*expected_relationship.at[row["Power"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
    
            kpis_figure_3 = go.Figure()
            
            
            kpis_figure_3.add_trace(go.Scatter(
                    name='P50',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected blade angle"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected blade angle high bound"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P95/P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected blade angle low bound"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Timestamp"],
                    y=actual_data["Blade angle"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
            
            try:
                events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
                events_data=pd.DataFrame(events_data)
                events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
                events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
                events_data=events_data.loc[events_data["Category"]=="stop"]
                events_data=events_data[["Start","End","Message"]]
                events_data.index=range(len(events_data.index))        
    
                for event in range(len(events_data.index)):
                    
                    min_y=min(actual_data["Blade angle"].min(),actual_data["Expected blade angle low bound"].min())
                    max_y=max(actual_data["Blade angle"].max(),actual_data["Expected blade angle high bound"].max())
                    
                    kpis_figure_3.add_shape(
                        type="rect",
                        x0=events_data["Start"].iloc[event],
                        y0=min_y,
                        x1=events_data["End"].iloc[event],
                        y1=max_y,
                        fillcolor='orange',
                        line_color='orange',
                        opacity=0.5
                    )
                
                    kpis_figure_3.add_trace(
                        go.Scatter(
                            x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
                            y=[min_y,max_y,max_y,min_y,min_y], 
                            fill="toself",
                            mode='lines',
                            name='',
                            text=str(events_data["Message"].iloc[event]),
                            opacity=0,
                            showlegend=False,
                            hoverlabel=dict(bgcolor="white")
                            )
                        )


            except:
                pass
    
    
    
            kpis_figure_3.update_yaxes(title="Blade angle (°)",showgrid=False)
            kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_3.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Blade angle based on power over time',x=0.5))
            figure_appearance={'display': 'block'}

        except:
            kpis_figure_3=default_figure











    elif active_cell["column_id"]=="Pitch vs RPM KPI":

        try:


            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"Pitch vs RPM"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Rotor speed","Blade angle","Timestamp"]].dropna()
            actual_data["Blade angle"]=actual_data["Blade angle"].apply(lambda x: ((x+180)%360)-180)
#            actual_data=actual_data.loc[actual_data["Blade angle"]<20]
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data["Rotor speed"]=actual_data["Rotor speed"].apply(lambda x: round(x,1))  
            actual_data=actual_data.loc[actual_data["Rotor speed"].isin(expected_relationship.index)]
            actual_data["Expected blade angle"]=actual_data["Rotor speed"].apply(lambda x: expected_relationship.at[x,"Avg"])
            actual_data["Expected blade angle high bound"]=actual_data["Rotor speed"].apply(lambda x: expected_relationship.at[x,"Avg"]+1.96*expected_relationship.at[x,"Stdev"])
            actual_data["Expected blade angle low bound"]=actual_data["Rotor speed"].apply(lambda x: expected_relationship.at[x,"Avg"]-1.96*expected_relationship.at[x,"Stdev"])
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Blade angle"]>=expected_relationship.at[row["Rotor speed"],"Avg"]-1.96*expected_relationship.at[row["Rotor speed"],"Stdev"])&(row["Blade angle"]<=expected_relationship.at[row["Rotor speed"],"Avg"]+1.96*expected_relationship.at[row["Rotor speed"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
    
            kpis_figure_3 = go.Figure()
            
            
            kpis_figure_3.add_trace(go.Scatter(
                    name='P50',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected blade angle"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected blade angle high bound"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P95/P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected blade angle low bound"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Timestamp"],
                    y=actual_data["Blade angle"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
    
            try:
                events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
                events_data=pd.DataFrame(events_data)
                events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
                events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
                events_data=events_data.loc[events_data["Category"]=="stop"]
                events_data=events_data[["Start","End","Message"]]
                events_data.index=range(len(events_data.index))        
    
                for event in range(len(events_data.index)):
                    
                    min_y=min(actual_data["Blade angle"].min(),actual_data["Expected blade angle low bound"].min())
                    max_y=max(actual_data["Blade angle"].max(),actual_data["Expected blade angle high bound"].max())
                    
                    kpis_figure_3.add_shape(
                        type="rect",
                        x0=events_data["Start"].iloc[event],
                        y0=min_y,
                        x1=events_data["End"].iloc[event],
                        y1=max_y,
                        fillcolor='orange',
                        line_color='orange',
                        opacity=0.5
                    )
                
                    kpis_figure_3.add_trace(
                        go.Scatter(
                            x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
                            y=[min_y,max_y,max_y,min_y,min_y], 
                            fill="toself",
                            mode='lines',
                            name='',
                            text=str(events_data["Message"].iloc[event]),
                            opacity=0,
                            showlegend=False,
                            hoverlabel=dict(bgcolor="white")
                            )
                        )
            except:
                pass
    
            kpis_figure_3.update_yaxes(title="Blade angle (°)",showgrid=False)
            kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_3.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Blade angle based on rotor speed over time',x=0.5))
            figure_appearance={'display': 'block'}

        except:
            kpis_figure_3=default_figure






    elif active_cell["column_id"]=="RPM vs Power KPI":

        try:



            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"RPM vs Power"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Power","Rotor speed","Timestamp"]].dropna()
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data["Power"]=actual_data["Power"].apply(lambda x: round(x,0))  
            actual_data=actual_data.loc[actual_data["Power"].isin(expected_relationship.index)]
            actual_data["Expected rotor speed"]=actual_data["Power"].apply(lambda x: expected_relationship.at[x,"Avg"])
            actual_data["Expected rotor speed high bound"]=actual_data["Power"].apply(lambda x: expected_relationship.at[x,"Avg"]+1.96*expected_relationship.at[x,"Stdev"])
            actual_data["Expected rotor speed low bound"]=actual_data["Power"].apply(lambda x: expected_relationship.at[x,"Avg"]-1.96*expected_relationship.at[x,"Stdev"])
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Rotor speed"]>=expected_relationship.at[row["Power"],"Avg"]-1.96*expected_relationship.at[row["Power"],"Stdev"])&(row["Rotor speed"]<=expected_relationship.at[row["Power"],"Avg"]+1.96*expected_relationship.at[row["Power"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
    
            kpis_figure_3 = go.Figure()
            
            
            kpis_figure_3.add_trace(go.Scatter(
                    name='P50',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected rotor speed"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected rotor speed high bound"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P95/P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected rotor speed low bound"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Timestamp"],
                    y=actual_data["Rotor speed"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
            try:
                events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
                events_data=pd.DataFrame(events_data)
                events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
                events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
                events_data=events_data.loc[events_data["Category"]=="stop"]
                events_data=events_data[["Start","End","Message"]]
                events_data.index=range(len(events_data.index))        
    
                for event in range(len(events_data.index)):
                    
                    min_y=min(actual_data["Rotor speed"].min(),actual_data["Expected rotor speed low bound"].min())
                    max_y=max(actual_data["Rotor speed"].max(),actual_data["Expected rotor speed high bound"].max())
                    
                    kpis_figure_3.add_shape(
                        type="rect",
                        x0=events_data["Start"].iloc[event],
                        y0=min_y,
                        x1=events_data["End"].iloc[event],
                        y1=max_y,
                        fillcolor='orange',
                        line_color='orange',
                        opacity=0.5
                    )
                
                    kpis_figure_3.add_trace(
                        go.Scatter(
                            x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
                            y=[min_y,max_y,max_y,min_y,min_y], 
                            fill="toself",
                            mode='lines',
                            name='',
                            text=str(events_data["Message"].iloc[event]),
                            opacity=0,
                            showlegend=False,
                            hoverlabel=dict(bgcolor="white")
                            )
                        )
            except:
                pass
    
    
            kpis_figure_3.update_yaxes(title="Rotor speed (RPM)",showgrid=False)
            kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_3.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Rotor speed based on power over time',x=0.5))
            figure_appearance={'display': 'block'}

        except:
            kpis_figure_3=default_figure




    elif active_cell["column_id"]=="Power vs RPM KPI":

        try:

            
            
            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"Power vs RPM"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Rotor speed","Power","Timestamp"]].dropna()
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data["Rotor speed"]=actual_data["Rotor speed"].apply(lambda x: round(x,1))  
            actual_data=actual_data.loc[actual_data["Rotor speed"].isin(expected_relationship.index)]
            actual_data["Expected power"]=actual_data["Rotor speed"].apply(lambda x: expected_relationship.at[x,"Avg"])
            actual_data["Expected power high bound"]=actual_data["Rotor speed"].apply(lambda x: expected_relationship.at[x,"Avg"]+1.96*expected_relationship.at[x,"Stdev"])
            actual_data["Expected power low bound"]=actual_data["Rotor speed"].apply(lambda x: expected_relationship.at[x,"Avg"]-1.96*expected_relationship.at[x,"Stdev"])
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Power"]>=expected_relationship.at[row["Rotor speed"],"Avg"]-1.96*expected_relationship.at[row["Rotor speed"],"Stdev"])&(row["Power"]<=expected_relationship.at[row["Rotor speed"],"Avg"]+1.96*expected_relationship.at[row["Rotor speed"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}      
    
    
            kpis_figure_3 = go.Figure()
            
            
            kpis_figure_3.add_trace(go.Scatter(
                    name='P50',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected power"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected power high bound"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P95/P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected power low bound"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Timestamp"],
                    y=actual_data["Power"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
            try:
                events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
                events_data=pd.DataFrame(events_data)
                events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
                events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
                events_data=events_data.loc[events_data["Category"]=="stop"]
                events_data=events_data[["Start","End","Message"]]
                events_data.index=range(len(events_data.index))        
    
                for event in range(len(events_data.index)):
                    
                    min_y=min(actual_data["Power"].min(),actual_data["Expected power low bound"].min())
                    max_y=max(actual_data["Power"].max(),actual_data["Expected power high bound"].max())
                    
                    kpis_figure_3.add_shape(
                        type="rect",
                        x0=events_data["Start"].iloc[event],
                        y0=min_y,
                        x1=events_data["End"].iloc[event],
                        y1=max_y,
                        fillcolor='orange',
                        line_color='orange',
                        opacity=0.5
                    )
                
                    kpis_figure_3.add_trace(
                        go.Scatter(
                            x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
                            y=[min_y,max_y,max_y,min_y,min_y], 
                            fill="toself",
                            mode='lines',
                            name='',
                            text=str(events_data["Message"].iloc[event]),
                            opacity=0,
                            showlegend=False,
                            hoverlabel=dict(bgcolor="white")
                            )
                        )
            except:
                pass
    
    
            kpis_figure_3.update_yaxes(title="Power (kW)",showgrid=False)
            kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_3.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Power based on rotor speed over time',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_3=default_figure















    elif active_cell["column_id"]=="Power vs Speed KPI":

        try:

            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"Power vs Speed"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Corrected wind speed","Power","Timestamp"]].dropna()
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data["Corrected wind speed"]=actual_data["Corrected wind speed"].apply(lambda x: round(x,1))  
            actual_data=actual_data.loc[actual_data["Corrected wind speed"].isin(expected_relationship.index)]
            actual_data["Expected power"]=actual_data["Corrected wind speed"].apply(lambda x: expected_relationship.at[x,"Avg"])
            actual_data["Expected power high bound"]=actual_data["Corrected wind speed"].apply(lambda x: expected_relationship.at[x,"Avg"]+1.96*expected_relationship.at[x,"Stdev"])
            actual_data["Expected power low bound"]=actual_data["Corrected wind speed"].apply(lambda x: expected_relationship.at[x,"Avg"]-1.96*expected_relationship.at[x,"Stdev"])
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Power"]>=expected_relationship.at[row["Corrected wind speed"],"Avg"]-1.96*expected_relationship.at[row["Corrected wind speed"],"Stdev"])&(row["Power"]<=expected_relationship.at[row["Corrected wind speed"],"Avg"]+1.96*expected_relationship.at[row["Corrected wind speed"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
    
            kpis_figure_3 = go.Figure()
            
            
            kpis_figure_3.add_trace(go.Scatter(
                    name='P50',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected power"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected power high bound"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P95/P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected power low bound"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Timestamp"],
                    y=actual_data["Power"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
            try:
                events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
                events_data=pd.DataFrame(events_data)
                events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
                events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
                events_data=events_data.loc[events_data["Category"]=="stop"]
                events_data=events_data[["Start","End","Message"]]
                events_data.index=range(len(events_data.index))        
    
                for event in range(len(events_data.index)):
                    
                    min_y=min(actual_data["Power"].min(),actual_data["Expected power low bound"].min())
                    max_y=max(actual_data["Power"].max(),actual_data["Expected power high bound"].max())
                    
                    kpis_figure_3.add_shape(
                        type="rect",
                        x0=events_data["Start"].iloc[event],
                        y0=min_y,
                        x1=events_data["End"].iloc[event],
                        y1=max_y,
                        fillcolor='orange',
                        line_color='orange',
                        opacity=0.5
                    )
                
                    kpis_figure_3.add_trace(
                        go.Scatter(
                            x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
                            y=[min_y,max_y,max_y,min_y,min_y], 
                            fill="toself",
                            mode='lines',
                            name='',
                            text=str(events_data["Message"].iloc[event]),
                            opacity=0,
                            showlegend=False,
                            hoverlabel=dict(bgcolor="white")
                            )
                        )
            except:
                pass
    
    
            kpis_figure_3.update_yaxes(title="Power (kW)",showgrid=False)
            kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_3.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Power based on wind speed over time',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_3=default_figure





    elif active_cell["column_id"]=="Pitch vs Speed KPI":

        try:

            
            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"Pitch vs Speed"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Wind speed","Blade angle","Timestamp"]].dropna()
            actual_data["Blade angle"]=actual_data["Blade angle"].apply(lambda x: ((x+180)%360)-180)
#            actual_data=actual_data.loc[actual_data["Blade angle"]<20]
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data["Wind speed"]=actual_data["Wind speed"].apply(lambda x: round(x,1))  
            actual_data=actual_data.loc[actual_data["Wind speed"].isin(expected_relationship.index)]
            actual_data["Expected blade angle"]=actual_data["Wind speed"].apply(lambda x: expected_relationship.at[x,"Avg"])
            actual_data["Expected blade angle high bound"]=actual_data["Wind speed"].apply(lambda x: expected_relationship.at[x,"Avg"]+1.96*expected_relationship.at[x,"Stdev"])
            actual_data["Expected blade angle low bound"]=actual_data["Wind speed"].apply(lambda x: expected_relationship.at[x,"Avg"]-1.96*expected_relationship.at[x,"Stdev"])
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Blade angle"]>=expected_relationship.at[row["Wind speed"],"Avg"]-1.96*expected_relationship.at[row["Wind speed"],"Stdev"])&(row["Blade angle"]<=expected_relationship.at[row["Wind speed"],"Avg"]+1.96*expected_relationship.at[row["Wind speed"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
    
            kpis_figure_3 = go.Figure()
            
            
            kpis_figure_3.add_trace(go.Scatter(
                    name='P50',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected blade angle"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected blade angle high bound"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P95/P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected blade angle low bound"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Timestamp"],
                    y=actual_data["Blade angle"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
            try:
                events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
                events_data=pd.DataFrame(events_data)
                events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
                events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
                events_data=events_data.loc[events_data["Category"]=="stop"]
                events_data=events_data[["Start","End","Message"]]
                events_data.index=range(len(events_data.index))        
    
                for event in range(len(events_data.index)):
                    
                    min_y=min(actual_data["Blade angle"].min(),actual_data["Expected blade angle low bound"].min())
                    max_y=max(actual_data["Blade angle"].max(),actual_data["Expected blade angle high bound"].max())
                    
                    kpis_figure_3.add_shape(
                        type="rect",
                        x0=events_data["Start"].iloc[event],
                        y0=min_y,
                        x1=events_data["End"].iloc[event],
                        y1=max_y,
                        fillcolor='orange',
                        line_color='orange',
                        opacity=0.5
                    )
                
                    kpis_figure_3.add_trace(
                        go.Scatter(
                            x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
                            y=[min_y,max_y,max_y,min_y,min_y], 
                            fill="toself",
                            mode='lines',
                            name='',
                            text=str(events_data["Message"].iloc[event]),
                            opacity=0,
                            showlegend=False,
                            hoverlabel=dict(bgcolor="white")
                            )
                        )
            except:
                pass
    
    
            kpis_figure_3.update_yaxes(title="Blade angle (°)",showgrid=False)
            kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_3.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Blade angle based on wind speed over time',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_3=default_figure





    elif active_cell["column_id"]=="RPM vs Speed KPI":

        try:

            
            current_wt_type=kpis.loc[kpis["Wind turbine"]==selected_wt]["Wind turbine type"].iloc[0]
            expected_relationship=wt_types_relationships.at[current_wt_type,"RPM vs Speed"]
            expected_relationship=pd.DataFrame(expected_relationship)
            actual_data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Filtered data"].iloc[0]
            actual_data=pd.DataFrame(actual_data)
            check_if_yaw_misalignment_data=actual_data["Yaw misalignment"].count()
            if check_if_yaw_misalignment_data>0:
                actual_data=actual_data.loc[(actual_data["Yaw misalignment"]>=-5) & (actual_data["Yaw misalignment"]<=5)]
            actual_data=actual_data[["Wind speed","Rotor speed","Timestamp"]].dropna()
            actual_data["Timestamp"]=actual_data["Timestamp"].apply(lambda x: dt.strptime(x,'%d/%m/%Y %H:%M'))
            actual_data["Wind speed"]=actual_data["Wind speed"].apply(lambda x: round(x,1))  
            actual_data=actual_data.loc[actual_data["Wind speed"].isin(expected_relationship.index)]
            actual_data["Expected rotor speed"]=actual_data["Wind speed"].apply(lambda x: expected_relationship.at[x,"Avg"])
            actual_data["Expected rotor speed high bound"]=actual_data["Wind speed"].apply(lambda x: expected_relationship.at[x,"Avg"]+1.96*expected_relationship.at[x,"Stdev"])
            actual_data["Expected rotor speed low bound"]=actual_data["Wind speed"].apply(lambda x: expected_relationship.at[x,"Avg"]-1.96*expected_relationship.at[x,"Stdev"])
            actual_data["flag"]=actual_data.apply(lambda row: "black" if (row["Rotor speed"]>=expected_relationship.at[row["Wind speed"],"Avg"]-1.96*expected_relationship.at[row["Wind speed"],"Stdev"])&(row["Rotor speed"]<=expected_relationship.at[row["Wind speed"],"Avg"]+1.96*expected_relationship.at[row["Wind speed"],"Stdev"]) else "red",axis=1)
            colors=actual_data["flag"].tolist()
            marker={'color': colors,'size': 3}            
    
    
            kpis_figure_3 = go.Figure()
            
            
            kpis_figure_3.add_trace(go.Scatter(
                    name='P50',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected rotor speed"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected rotor speed high bound"],
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    marker=dict(color="#444"),
                    showlegend=False
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='P95/P5',
                    x=actual_data["Timestamp"],
                    y=actual_data["Expected rotor speed low bound"],
                    marker=dict(color="#444"),
                    mode='lines',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'width':0
                            },
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=True
                ))
    
            kpis_figure_3.add_trace(go.Scatter(
                    name='Actual data',
                    x=actual_data["Timestamp"],
                    y=actual_data["Rotor speed"],
                    mode='markers',
                    opacity=0.7,
                    marker=marker,
                    showlegend=True
                ))
    
            try:
                events_data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["details"].iloc[0]
                events_data=pd.DataFrame(events_data)
                events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
                events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
                events_data=events_data.loc[events_data["Category"]=="stop"]
                events_data=events_data[["Start","End","Message"]]
                events_data.index=range(len(events_data.index))        
    
                for event in range(len(events_data.index)):
                    
                    min_y=min(actual_data["Rotor speed"].min(),actual_data["Expected rotor speed low bound"].min())
                    max_y=max(actual_data["Rotor speed"].max(),actual_data["Expected rotor speed high bound"].max())
                    
                    kpis_figure_3.add_shape(
                        type="rect",
                        x0=events_data["Start"].iloc[event],
                        y0=min_y,
                        x1=events_data["End"].iloc[event],
                        y1=max_y,
                        fillcolor='orange',
                        line_color='orange',
                        opacity=0.5
                    )
                
                    kpis_figure_3.add_trace(
                        go.Scatter(
                            x=[events_data["Start"].iloc[event],events_data["Start"].iloc[event],events_data["End"].iloc[event],events_data["End"].iloc[event],events_data["Start"].iloc[event]], 
                            y=[min_y,max_y,max_y,min_y,min_y], 
                            fill="toself",
                            mode='lines',
                            name='',
                            text=str(events_data["Message"].iloc[event]),
                            opacity=0,
                            showlegend=False,
                            hoverlabel=dict(bgcolor="white")
                            )
                        )
            except:
                pass
    
    
            kpis_figure_3.update_yaxes(title="Rotor speed (RPM)",showgrid=False)
            kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_3.update_layout(title=dict(text=str(selected_wt)+" - "+str(graph_month)+" "+str(graph_year)+' - Rotor speed based on wind speed over time',x=0.5))
            figure_appearance={'display': 'block'}
        
        except:
            kpis_figure_3=default_figure

    else:
        kpis_figure_3=default_figure

    return kpis_figure_3,figure_appearance











@app.callback(
        [Output('kpis-graph-4', 'figure'),
         Output("kpis-graph-4-container","style")
         ],[Input('kpis-datatable', 'active_cell'),
         Input('kpis-graph','clickData')])
def kpis_graph_4(active_cell,clickData):
    

    if clickData is not None:
        selected_time=clickData['points'][0]['x']
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)
        selected_time=selected_time.strftime("%B %Y")

    graph_month=dt.strptime(selected_time,'%B %Y').strftime("%B")
    graph_year=dt.strptime(selected_time,'%B %Y').year

    figure_appearance={'display': 'none'}



    if active_cell["column_id"]=="Static yaw misalignment KPI":

        try:
            
            yaw_mis_distribution=kpis.loc[(kpis["Month"]==selected_time)]
            yaw_mis_distribution=yaw_mis_distribution[["Manufacturer","Static yaw misalignment best guess"]]
            
            manufacturers=sorted(list(set(yaw_mis_distribution["Manufacturer"].tolist())))
            yaw_mis_distributions_by_type=pd.DataFrame(index=manufacturers,columns=["Uncertain","<2.5°","2.5° to 5°","5° to 7.5°",">7.5°"])
            for manufacturer in manufacturers:
                type_yaw_mis_distribution=yaw_mis_distribution.loc[yaw_mis_distribution["Manufacturer"]==manufacturer]
                type_yaw_mis_distribution=type_yaw_mis_distribution[["Static yaw misalignment best guess"]]
                try:
                    value=type_yaw_mis_distribution["Static yaw misalignment best guess"].isnull().sum()
                    yaw_mis_distributions_by_type.at[manufacturer,"Uncertain"]=value
                except:
                    yaw_mis_distributions_by_type.at[manufacturer,"Uncertain"]=np.nan    
                type_yaw_mis_distribution=type_yaw_mis_distribution.dropna()
                try:
                    value=type_yaw_mis_distribution.loc[(type_yaw_mis_distribution["Static yaw misalignment best guess"]>=-2.5)&(type_yaw_mis_distribution["Static yaw misalignment best guess"]<=2.5)].count()
                    yaw_mis_distributions_by_type.at[manufacturer,"<2.5°"]=value.iloc[0]
                except:
                    yaw_mis_distributions_by_type.at[manufacturer,"<2.5°"]=np.nan    
                try:
                    value=type_yaw_mis_distribution.loc[((type_yaw_mis_distribution["Static yaw misalignment best guess"]>=-5)&(type_yaw_mis_distribution["Static yaw misalignment best guess"]<-2.5))|((type_yaw_mis_distribution["Static yaw misalignment best guess"]<=5)&(type_yaw_mis_distribution["Static yaw misalignment best guess"]>2.5))].count()
                    yaw_mis_distributions_by_type.at[manufacturer,"2.5° to 5°"]=value.iloc[0]
                except:
                    yaw_mis_distributions_by_type.at[manufacturer,"2.5° to 5°"]=np.nan    
                try:
                    value=type_yaw_mis_distribution.loc[((type_yaw_mis_distribution["Static yaw misalignment best guess"]>=-7.5)&(type_yaw_mis_distribution["Static yaw misalignment best guess"]<-5))|((type_yaw_mis_distribution["Static yaw misalignment best guess"]<=7.5)&(type_yaw_mis_distribution["Static yaw misalignment best guess"]>5))].count()
                    yaw_mis_distributions_by_type.at[manufacturer,"5° to 7.5°"]=value.iloc[0]
                except:
                    yaw_mis_distributions_by_type.at[manufacturer,"5° to 7.5°"]=np.nan    
                try:
                    value=type_yaw_mis_distribution.loc[((type_yaw_mis_distribution["Static yaw misalignment best guess"]<-7.5))|((type_yaw_mis_distribution["Static yaw misalignment best guess"]>7.5))].count()
                    yaw_mis_distributions_by_type.at[manufacturer,">7.5°"]=value.iloc[0]
                except:
                    yaw_mis_distributions_by_type.at[manufacturer,">7.5°"]=np.nan    
            
            kpis_figure_4 = go.Figure(data=[
                go.Bar(name="Uncertain", x=yaw_mis_distributions_by_type.index, y=yaw_mis_distributions_by_type["Uncertain"],marker=go.bar.Marker(color='black')),
                go.Bar(name="<2.5°", x=yaw_mis_distributions_by_type.index, y=yaw_mis_distributions_by_type["<2.5°"],marker=go.bar.Marker(color='green')),
                go.Bar(name="2.5° to 5°", x=yaw_mis_distributions_by_type.index, y=yaw_mis_distributions_by_type["2.5° to 5°"],marker=go.bar.Marker(color='orange')),
                go.Bar(name="5° to 7.5°", x=yaw_mis_distributions_by_type.index, y=yaw_mis_distributions_by_type["5° to 7.5°"],marker=go.bar.Marker(color='red')),
                go.Bar(name=">7.5°", x=yaw_mis_distributions_by_type.index, y=yaw_mis_distributions_by_type[">7.5°"],marker=go.bar.Marker(color='purple')),
            ])

            kpis_figure_4.update_layout(barmode='stack')

            kpis_figure_4.update_yaxes(title="Static yaw misalignment distribution (-)",showgrid=False)
            kpis_figure_4.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_4.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - Static yaw misalignment distribution by manufacturer',x=0.5))
            figure_appearance={'display': 'block'}


        except:
            kpis_figure_4=default_figure




        
    else:
        kpis_figure_4=default_figure

    
    
    return kpis_figure_4,figure_appearance











@app.callback(
        [Output('kpis-graph-5', 'figure'),
         Output("kpis-graph-5-container","style")
         ],[Input('kpis-datatable', 'active_cell'),
         Input('kpis-graph','clickData')])
def kpis_graph_5(active_cell,clickData):
    

    if clickData is not None:
        selected_time=clickData['points'][0]['x']
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)
        selected_time=selected_time.strftime("%B %Y")

    graph_month=dt.strptime(selected_time,'%B %Y').strftime("%B")
    graph_year=dt.strptime(selected_time,'%B %Y').year

    figure_appearance={'display': 'none'}


    if active_cell["column_id"]=="Static yaw misalignment KPI":



        try:
            
            yaw_mis_distribution=kpis.loc[(kpis["Month"]==selected_time)]
            yaw_mis_distribution=yaw_mis_distribution[["Manufacturer","Static yaw misalignment best guess","SCADA Prod"]]
            yaw_mis_distribution["Loss to static yaw misalignment (%)"]=yaw_mis_distribution["Static yaw misalignment best guess"].apply(lambda x: 1-((math.cos(x*math.pi/180))**3))
            yaw_mis_distribution["Loss to static yaw misalignment (MWh)"]=yaw_mis_distribution["Loss to static yaw misalignment (%)"]*yaw_mis_distribution["SCADA Prod"]
            yaw_mis_distribution=yaw_mis_distribution[["Manufacturer","SCADA Prod","Loss to static yaw misalignment (MWh)"]]
            yaw_mis_distribution=yaw_mis_distribution.groupby(["Manufacturer"]).sum()
            yaw_mis_distribution["Loss to static yaw misalignment (%)"]=100*yaw_mis_distribution["Loss to static yaw misalignment (MWh)"]/yaw_mis_distribution["SCADA Prod"]
            yaw_mis_distribution.sort_index(inplace=True)


            

            
            
            kpis_figure_5 = go.Figure(data=[
                go.Bar(name="<2.5°", x=yaw_mis_distribution.index, y=yaw_mis_distribution["Loss to static yaw misalignment (%)"]),
            ])


            kpis_figure_5.update_yaxes(title="Loss to static yaw misalignment (%)",showgrid=False)
            kpis_figure_5.update_xaxes(showgrid=False)
    
    
    
            kpis_figure_5.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - Loss to static yaw misalignment by manufacturer',x=0.5))


            figure_appearance={'display': 'block'}
        except:
            kpis_figure_5=default_figure
        
        
    

        
    else:
        kpis_figure_5=default_figure

    
    
    return kpis_figure_5,figure_appearance








    
    
    
    
    
    
    
    
@app.callback(
        [Output('resource-kpis-graph', 'figure'),
         Output("resource-kpis-graph-container","style")
         ],[Input('resource-kpis-datatable', 'active_cell')])
def resource_kpis_graph(active_cell):
    selected_wf=current_resource_kpis["Wind farm"].iloc[active_cell["row_id"]]
    figure_appearance={'display': 'none'}

    if active_cell["column_id"]=="Monthly Wind Resource KPI":
        wt_data=resource_kpis.loc[resource_kpis["Wind farm"]==selected_wf]
        WIs=wt_data["Wind Indexes"].iloc[0]
        P50s=wt_data["Monthly P50s"].iloc[0]
        P75s=wt_data["Monthly P75s"].iloc[0]
        P25s=wt_data["Monthly P25s"].iloc[0]
        P90s=wt_data["Monthly P90s"].iloc[0]
        P10s=wt_data["Monthly P10s"].iloc[0]
        WIs=pd.DataFrame.from_dict(WIs, orient='index',columns=["WIs"])
        P50s=pd.DataFrame.from_dict(P50s, orient='index',columns=["P50s"])
        P75s=pd.DataFrame.from_dict(P75s, orient='index',columns=["P75s"])
        P25s=pd.DataFrame.from_dict(P25s, orient='index',columns=["P25s"])
        P90s=pd.DataFrame.from_dict(P90s, orient='index',columns=["P90s"])
        P10s=pd.DataFrame.from_dict(P10s, orient='index',columns=["P10s"])        
        data = pd.concat([WIs,P50s,P75s,P25s,P90s,P10s], axis=1)
        
        resource_kpis_figure = go.Figure()
        
        
        resource_kpis_figure.add_trace(go.Scatter(
                name='Wind resource',
                legendgroup = 'a',
                x=data.index,
                y=data["WIs"],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P50',
                legendgroup = 'b',
                x=data.index,
                y=data["P50s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Green',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'c',
                x=data.index,
                y=data["P90s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'd',
                x=data.index,
                y=data["P75s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'd',
                x=data.index,
                y=data["P25s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'c',
                x=data.index,
                y=data["P10s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))

        resource_kpis_figure.update_yaxes(title="Wind index (-)",showgrid=False)
        resource_kpis_figure.update_xaxes(showgrid=False,range=[dt.today().replace(month=1,day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-12), dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)])
        resource_kpis_figure.update_layout(title=dict(text=selected_wf+' - Historical wind resource',x=0.5))
        figure_appearance={'display': 'block'}
        
        
    elif active_cell["column_id"]=="Monthly Year-To-Date Wind Resource KPI":
        wt_data=resource_kpis.loc[resource_kpis["Wind farm"]==selected_wf]
        WIs=wt_data["Monthly YTD Wind Indexes"].iloc[0]
        P50s=wt_data["Monthly YTD P50s"].iloc[0]
        P75s=wt_data["Monthly YTD P75s"].iloc[0]
        P25s=wt_data["Monthly YTD P25s"].iloc[0]
        P90s=wt_data["Monthly YTD P90s"].iloc[0]
        P10s=wt_data["Monthly YTD P10s"].iloc[0]
        WIs=pd.DataFrame.from_dict(WIs, orient='index',columns=["WIs"])
        P50s=pd.DataFrame.from_dict(P50s, orient='index',columns=["P50s"])
        P75s=pd.DataFrame.from_dict(P75s, orient='index',columns=["P75s"])
        P25s=pd.DataFrame.from_dict(P25s, orient='index',columns=["P25s"])
        P90s=pd.DataFrame.from_dict(P90s, orient='index',columns=["P90s"])
        P10s=pd.DataFrame.from_dict(P10s, orient='index',columns=["P10s"])        
        data = pd.concat([WIs,P50s,P75s,P25s,P90s,P10s], axis=1)
        
        resource_kpis_figure = go.Figure()
        
        
        resource_kpis_figure.add_trace(go.Scatter(
                name='Wind resource',
                legendgroup = 'a',
                x=data.index,
                y=data["WIs"],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P50',
                legendgroup = 'b',
                x=data.index,
                y=data["P50s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Green',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'c',
                x=data.index,
                y=data["P90s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'd',
                x=data.index,
                y=data["P75s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'd',
                x=data.index,
                y=data["P25s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'c',
                x=data.index,
                y=data["P10s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))

        resource_kpis_figure.update_yaxes(title="Wind index (-)",showgrid=False)
        resource_kpis_figure.update_xaxes(showgrid=False,range=[dt.today().replace(month=1,day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-12), dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)])

        resource_kpis_figure.update_layout(title=dict(text=selected_wf+' - Historical YTD wind resource',x=0.5))
        figure_appearance={'display': 'block'}

        
    else:
        resource_kpis_figure=default_figure

    
    
    return resource_kpis_figure,figure_appearance
    
    
    
@app.callback(
        [Output('resource-kpis-graph-2', 'figure'),
         Output("resource-kpis-graph-2-container","style")
         ],[Input('resource-kpis-datatable', 'active_cell'),
         Input('resource-kpis-graph','clickData')])
def resource_kpis_graph_2(active_cell,clickData):    
    
    if clickData is not None:
        selected_time=clickData['points'][0]['x']
        selected_time=dt.strptime(selected_time,'%Y-%m-%d')
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)

    graph_month=selected_time.strftime("%B")
    graph_year=selected_time.year

    figure_appearance={'display': 'none'}

    if active_cell["column_id"]=="Monthly Wind Resource KPI":
        
        data=resource_kpis[["WTLongitude","WTLatitude","Wind Indexes","Monthly WI Graph Colors"]]
        data["Wind Indexes"]=data["Wind Indexes"].apply(lambda x: x[selected_time])        
        data["Monthly WI Graph Colors"]=data["Monthly WI Graph Colors"].apply(lambda x: x[selected_time])
        colors=data["Monthly WI Graph Colors"].tolist()
        color_names=["green","orange","red"]
        color_vals = list(range(len(color_names)))
        num_colors=len(color_vals)
        colorscale = [[int((i+1)/2)/num_colors,color_names[int(i/2)]] for i in range(2*num_colors)]
        color_names=[">P50","P75-P50","<P75"]
        cmin = -0.5
        cmax = num_colors - 0.5
        marker={'color': colors,'colorbar': {'title': "Legend",'tickvals': color_vals,'ticktext': color_names,'y': 1,'len': 0.4,'yanchor': 'top'},'colorscale': colorscale,'cmin': cmin,'cmax': cmax,'size': 8}            

        resource_kpis_figure_2=go.Figure(data=go.Scattergeo(
        lon = data["WTLongitude"],
        lat = data["WTLatitude"],
        text=data["Wind Indexes"],
        mode='markers',
        marker=marker
        ))

        resource_kpis_figure_2.update_geos(
                lataxis_range=[data["WTLatitude"].min()-5, data["WTLatitude"].max()+5],
                lonaxis_range=[data["WTLongitude"].min()-5, data["WTLongitude"].max()+5],
                projection_type='mercator',
                showcountries=True,
                showframe=False
                )

        resource_kpis_figure_2.update_layout(geo=dict(resolution=50),title=dict(text=str(graph_month)+" "+str(graph_year)+" monthly wind resource against historical average",x=0.5),height=900)
        figure_appearance={'display': 'block'}


    elif active_cell["column_id"]=="Monthly Year-To-Date Wind Resource KPI":
        
        data=resource_kpis[["WTLongitude","WTLatitude","Monthly YTD Wind Indexes","Monthly YTD WI Graph Colors"]]
        data["Monthly YTD Wind Indexes"]=data["Monthly YTD Wind Indexes"].apply(lambda x: x[selected_time])        
        data["Monthly YTD WI Graph Colors"]=data["Monthly YTD WI Graph Colors"].apply(lambda x: x[selected_time])
        colors=data["Monthly YTD WI Graph Colors"].tolist()
        color_names=["green","orange","red"]
        color_vals = list(range(len(color_names)))
        num_colors=len(color_vals)
        colorscale = [[int((i+1)/2)/num_colors,color_names[int(i/2)]] for i in range(2*num_colors)]
        color_names=[">P50","P75-P50","<P75"]
        cmin = -0.5
        cmax = num_colors - 0.5
        marker={'color': colors,'colorbar': {'title': "Legend",'tickvals': color_vals,'ticktext': color_names,'y': 1,'len': 0.4,'yanchor': 'top'},'colorscale': colorscale,'cmin': cmin,'cmax': cmax,'size': 8}            

        resource_kpis_figure_2=go.Figure(data=go.Scattergeo(
        lon = data["WTLongitude"],
        lat = data["WTLatitude"],
        text=data["Monthly YTD Wind Indexes"],
        mode='markers',
        marker=marker
        ))

        resource_kpis_figure_2.update_geos(
                lataxis_range=[data["WTLatitude"].min()-5, data["WTLatitude"].max()+5],
                lonaxis_range=[data["WTLongitude"].min()-5, data["WTLongitude"].max()+5],
                projection_type='mercator',
                showcountries=True,
                showframe=False
                )

        resource_kpis_figure_2.update_layout(geo=dict(resolution=50),title=dict(text=str(graph_month)+" "+str(graph_year)+" YTD wind resource against historical average",x=0.5),height=900)
        figure_appearance={'display': 'block'}


    else:
        resource_kpis_figure_2=default_figure_2

    
    
    return resource_kpis_figure_2,figure_appearance
    
    
    















print('11808')









@app.callback(
        [Output('resource-kpis-graph-3', 'figure'),
         Output("resource-kpis-graph-3-container","style")
         ],[Input('resource-kpis-datatable', 'active_cell')])
def resource_kpis_graph_3(active_cell):
    selected_wf=current_resource_kpis["Wind farm"].iloc[active_cell["row_id"]]
    selected_country=resource_kpis.loc[resource_kpis["Wind farm"]==selected_wf]["Country"].iloc[0]
    selected_cluster=[k for k in clusters.keys() if selected_country in clusters[k]]
    selected_cluster=selected_cluster[0]
    
    figure_appearance={'display': 'none'}
    
    
    if active_cell["column_id"]=="Monthly Wind Resource KPI":
        wt_data=clustered_resource_kpis.loc[clustered_resource_kpis["Cluster"]==selected_cluster]
        WIs=wt_data["Wind Indexes"].iloc[0]
        P50s=wt_data["Monthly P50s"].iloc[0]
        P75s=wt_data["Monthly P75s"].iloc[0]
        P25s=wt_data["Monthly P25s"].iloc[0]
        P90s=wt_data["Monthly P90s"].iloc[0]
        P10s=wt_data["Monthly P10s"].iloc[0]
        WIs=pd.DataFrame.from_dict(WIs, orient='index',columns=["WIs"])
        P50s=pd.DataFrame.from_dict(P50s, orient='index',columns=["P50s"])
        P75s=pd.DataFrame.from_dict(P75s, orient='index',columns=["P75s"])
        P25s=pd.DataFrame.from_dict(P25s, orient='index',columns=["P25s"])
        P90s=pd.DataFrame.from_dict(P90s, orient='index',columns=["P90s"])
        P10s=pd.DataFrame.from_dict(P10s, orient='index',columns=["P10s"])        
        data = pd.concat([WIs,P50s,P75s,P25s,P90s,P10s], axis=1)
        
        resource_kpis_figure = go.Figure()
        
        
        resource_kpis_figure.add_trace(go.Scatter(
                name='Wind resource',
                legendgroup = 'a',
                x=data.index,
                y=data["WIs"],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P50',
                legendgroup = 'b',
                x=data.index,
                y=data["P50s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Green',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'c',
                x=data.index,
                y=data["P90s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'd',
                x=data.index,
                y=data["P75s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'd',
                x=data.index,
                y=data["P25s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'c',
                x=data.index,
                y=data["P10s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))

        resource_kpis_figure.update_yaxes(title="Wind index (-)",showgrid=False)
        resource_kpis_figure.update_xaxes(showgrid=False,range=[dt.today().replace(month=1,day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-12), dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)])
        resource_kpis_figure.update_layout(title=dict(text=selected_cluster+' - Historical wind resource',x=0.5))
        figure_appearance={'display': 'block'}
        
        
    elif active_cell["column_id"]=="Monthly Year-To-Date Wind Resource KPI":
        wt_data=clustered_resource_kpis.loc[clustered_resource_kpis["Cluster"]==selected_cluster]
        WIs=wt_data["Monthly YTD Wind Indexes"].iloc[0]
        P50s=wt_data["Monthly YTD P50s"].iloc[0]
        P75s=wt_data["Monthly YTD P75s"].iloc[0]
        P25s=wt_data["Monthly YTD P25s"].iloc[0]
        P90s=wt_data["Monthly YTD P90s"].iloc[0]
        P10s=wt_data["Monthly YTD P10s"].iloc[0]
        WIs=pd.DataFrame.from_dict(WIs, orient='index',columns=["WIs"])
        P50s=pd.DataFrame.from_dict(P50s, orient='index',columns=["P50s"])
        P75s=pd.DataFrame.from_dict(P75s, orient='index',columns=["P75s"])
        P25s=pd.DataFrame.from_dict(P25s, orient='index',columns=["P25s"])
        P90s=pd.DataFrame.from_dict(P90s, orient='index',columns=["P90s"])
        P10s=pd.DataFrame.from_dict(P10s, orient='index',columns=["P10s"])        
        data = pd.concat([WIs,P50s,P75s,P25s,P90s,P10s], axis=1)
        
        resource_kpis_figure = go.Figure()
        
        
        resource_kpis_figure.add_trace(go.Scatter(
                name='Wind resource',
                legendgroup = 'a',
                x=data.index,
                y=data["WIs"],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P50',
                legendgroup = 'b',
                x=data.index,
                y=data["P50s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Green',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'c',
                x=data.index,
                y=data["P90s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'd',
                x=data.index,
                y=data["P75s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'd',
                x=data.index,
                y=data["P25s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))

        resource_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'c',
                x=data.index,
                y=data["P10s"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))

        resource_kpis_figure.update_yaxes(title="Wind index (-)",showgrid=False)
        resource_kpis_figure.update_xaxes(showgrid=False,range=[dt.today().replace(month=1,day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-12), dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)])

        resource_kpis_figure.update_layout(title=dict(text=selected_cluster+' - Historical YTD wind resource',x=0.5))
        figure_appearance={'display': 'block'}

        
    else:
        resource_kpis_figure=default_figure

    
    
    return resource_kpis_figure,figure_appearance
    
    
    





    
    
    
    
@app.callback(
        [Output('availability-kpis-graph', 'figure'),
         Output("availability-kpis-graph-container","style")
         ],[Input('availability-kpis-datatable', 'active_cell')])
def availability_kpis_graph(active_cell):
    
    selected_wt=current_availability_kpis["Wind turbine"].iloc[active_cell["row_id"]]
    figure_appearance={'display': 'none'}



    if active_cell["column_id"]=="Monthly time-based System Avail. KPI":
        data_over_time=availability_kpis.loc[availability_kpis["Wind turbine"]==selected_wt]

        availability_kpis_figure = go.Figure()
        
        
        availability_kpis_figure.add_trace(go.Scatter(
                name='Availability',
                x=data_over_time["Date"],
                y=data_over_time["Time-based System Avail."],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))

        availability_kpis_figure.add_trace(go.Scatter(
                name="P50",
                x=data_over_time["Date"],
                y=data_over_time["Time-based System Avail. P50"],
                mode='lines',
                line=dict(color="Green",dash="dashdot"),
                showlegend=True
            ))


        availability_kpis_figure.add_trace(go.Scatter(
                name="P95",
                x=data_over_time["Date"],
                y=data_over_time["Time-based System Avail. P95"],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=True
            ))

        availability_kpis_figure.add_trace(go.Scatter(
                name="P99",
                x=data_over_time["Date"],
                y=data_over_time["Time-based System Avail. P99"],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=True
            ))


        availability_kpis_figure.update_yaxes(title="Time-based System Avail. (%)",showgrid=False)
        availability_kpis_figure.update_xaxes(showgrid=False)



        availability_kpis_figure.update_layout(title=dict(text=selected_wt+' - Time-based System Availability',x=0.5))
        
        figure_appearance={'display': 'block'}



    elif active_cell["column_id"]=="Monthly production-based System Avail. KPI":
        data_over_time=availability_kpis.loc[availability_kpis["Wind turbine"]==selected_wt]
#        data_over_Production=data_over_Production.sort_values("Date")


        availability_kpis_figure = go.Figure()
        
        
        availability_kpis_figure.add_trace(go.Scatter(
                name='Availability',
                x=data_over_time["Date"],
                y=data_over_time["Production-based System Avail."],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))

        availability_kpis_figure.add_trace(go.Scatter(
                name="P50",
                x=data_over_time["Date"],
                y=data_over_time["Production-based System Avail. P50"],
                mode='lines',
                line=dict(color="Green",dash="dashdot"),
                showlegend=True
            ))


        availability_kpis_figure.add_trace(go.Scatter(
                name="P95",
                x=data_over_time["Date"],
                y=data_over_time["Production-based System Avail. P95"],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=True
            ))

        availability_kpis_figure.add_trace(go.Scatter(
                name="P99",
                x=data_over_time["Date"],
                y=data_over_time["Production-based System Avail. P99"],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=True
            ))


        availability_kpis_figure.update_yaxes(title="Production-based System Avail. (%)",showgrid=False)
        availability_kpis_figure.update_xaxes(showgrid=False)



        availability_kpis_figure.update_layout(title=dict(text=selected_wt+' - Production-based System Availability',x=0.5))

        figure_appearance={'display': 'block'}




    elif active_cell["column_id"]=="Monthly Year-To-Date time-based System Avail. KPI":
        data_over_time=restructured_ytd_availability_kpis.loc[restructured_ytd_availability_kpis["Wind turbine"]==selected_wt]

        availability_kpis_figure = go.Figure()
        
        
        availability_kpis_figure.add_trace(go.Scatter(
                name='Availability',
                x=data_over_time["Date"],
                y=data_over_time["Time-based System Avail."],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))

        availability_kpis_figure.add_trace(go.Scatter(
                name="P50",
                x=data_over_time["Date"],
                y=data_over_time["Time-based System Avail. P50"],
                mode='lines',
                line=dict(color="Green",dash="dashdot"),
                showlegend=True
            ))


        availability_kpis_figure.add_trace(go.Scatter(
                name="P95",
                x=data_over_time["Date"],
                y=data_over_time["Time-based System Avail. P95"],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=True
            ))

        availability_kpis_figure.add_trace(go.Scatter(
                name="P99",
                x=data_over_time["Date"],
                y=data_over_time["Time-based System Avail. P99"],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=True
            ))


        availability_kpis_figure.update_yaxes(title="Time-based System Avail. (%)",showgrid=False)
        availability_kpis_figure.update_xaxes(showgrid=False)



        availability_kpis_figure.update_layout(title=dict(text=selected_wt+' - YTD time-based System Availability',x=0.5))
        
        figure_appearance={'display': 'block'}



    elif active_cell["column_id"]=="Monthly Year-To-Date production-based System Avail. KPI":
        data_over_time=restructured_ytd_availability_kpis.loc[restructured_ytd_availability_kpis["Wind turbine"]==selected_wt]


        availability_kpis_figure = go.Figure()
        
        
        availability_kpis_figure.add_trace(go.Scatter(
                name='Availability',
                x=data_over_time["Date"],
                y=data_over_time["Production-based System Avail."],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))

        availability_kpis_figure.add_trace(go.Scatter(
                name="P50",
                x=data_over_time["Date"],
                y=data_over_time["Production-based System Avail. P50"],
                mode='lines',
                line=dict(color="Green",dash="dashdot"),
                showlegend=True
            ))


        availability_kpis_figure.add_trace(go.Scatter(
                name="P95",
                x=data_over_time["Date"],
                y=data_over_time["Production-based System Avail. P95"],
                mode='lines',
                line=dict(color="Orange",dash="dashdot"),
                showlegend=True
            ))

        availability_kpis_figure.add_trace(go.Scatter(
                name="P99",
                x=data_over_time["Date"],
                y=data_over_time["Production-based System Avail. P99"],
                mode='lines',
                line=dict(color="Red",dash="dashdot"),
                showlegend=True
            ))


        availability_kpis_figure.update_yaxes(title="Production-based System Avail. (%)",showgrid=False)
        availability_kpis_figure.update_xaxes(showgrid=False)



        availability_kpis_figure.update_layout(title=dict(text=selected_wt+' - YTD production-based System Availability',x=0.5))

        figure_appearance={'display': 'block'}


    elif active_cell["column_id"]=="Monthly own stops KPI":
        data_over_time=availability_kpis.loc[availability_kpis["Wind turbine"]==selected_wt]

        availability_kpis_figure = go.Figure()
        
        
        availability_kpis_figure.add_trace(go.Scatter(
                name='Assessment of own stops',
                x=data_over_time["Date"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Monthly own stops Graph Colors"].tolist(),'size': 12},
                showlegend=False
            ))



        availability_kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        availability_kpis_figure.update_xaxes(showgrid=False)



        availability_kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of own stops',x=0.5))
        
        figure_appearance={'display': 'block'}

    elif active_cell["column_id"]=="Monthly scheduled maintenance stops KPI":
        data_over_time=availability_kpis.loc[availability_kpis["Wind turbine"]==selected_wt]

        availability_kpis_figure = go.Figure()
        
        
        availability_kpis_figure.add_trace(go.Scatter(
                name='Assessment of scheduled maintenance stops',
                x=data_over_time["Date"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Monthly scheduled maintenance stops Graph Colors"].tolist(),'size': 12},
                showlegend=False
            ))



        availability_kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        availability_kpis_figure.update_xaxes(showgrid=False)



        availability_kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of scheduled maintenance stops',x=0.5))

        figure_appearance={'display': 'block'}


    elif active_cell["column_id"]=="Monthly Year-To-Date own stops KPI":
        data_over_time=restructured_ytd_availability_kpis.loc[restructured_ytd_availability_kpis["WT"]==selected_wt]

        availability_kpis_figure = go.Figure()
        
        
        availability_kpis_figure.add_trace(go.Scatter(
                name='Assessment of own stops',
                x=data_over_time["Date"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Monthly own stops Graph Colors"].tolist(),'size': 12},
                showlegend=False
            ))



        availability_kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        availability_kpis_figure.update_xaxes(showgrid=False)



        availability_kpis_figure.update_layout(title=dict(text=selected_wt+' - YTD assessment of own stops',x=0.5))
        
        figure_appearance={'display': 'block'}


    elif active_cell["column_id"]=="Monthly Year-To-Date scheduled maintenance stops KPI":
        data_over_time=restructured_ytd_availability_kpis.loc[restructured_ytd_availability_kpis["WT"]==selected_wt]

        availability_kpis_figure = go.Figure()
        
        
        availability_kpis_figure.add_trace(go.Scatter(
                name='Assessment of scheduled maintenance stops',
                x=data_over_time["Date"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Monthly scheduled maintenance stops Graph Colors"].tolist(),'size': 12},
                showlegend=False
            ))

        availability_kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        availability_kpis_figure.update_xaxes(showgrid=False)
        availability_kpis_figure.update_layout(title=dict(text=selected_wt+' - YTD assessment of scheduled maintenance stops',x=0.5))

        figure_appearance={'display': 'block'}

    elif active_cell["column_id"]=="Status Codes KPI":
        
        data_over_time=availability_kpis.loc[availability_kpis["Wind turbine"]==selected_wt]

        availability_kpis_figure = go.Figure()
        
        
        availability_kpis_figure.add_trace(go.Scatter(
                name='Assessment of status codes occurence',
                x=data_over_time["Date"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Status Codes KPI Graph Colors"].tolist(),'size': 12},
                showlegend=False
            ))



        availability_kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        availability_kpis_figure.update_xaxes(showgrid=False)
        availability_kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of status codes occurence',x=0.5))

        figure_appearance={'display': 'block'}
        
    elif active_cell["column_id"]=="Warnings KPI":
        
        data_over_time=availability_kpis.loc[availability_kpis["Wind turbine"]==selected_wt]

        availability_kpis_figure = go.Figure()
        
        
        availability_kpis_figure.add_trace(go.Scatter(
                name='Assessment of warnings occurence',
                x=data_over_time["Date"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Warnings KPI Graph Colors"].tolist(),'size': 12},
                showlegend=False
            ))



        availability_kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        availability_kpis_figure.update_xaxes(showgrid=False)
        availability_kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of warnings occurence',x=0.5))

        figure_appearance={'display': 'block'}

    elif active_cell["column_id"]=="Unclassified curtailments KPI":
        data_over_time=kpis.loc[kpis["Wind turbine"]==selected_wt]

        availability_kpis_figure = go.Figure()
        
        
        availability_kpis_figure.add_trace(go.Scatter(
                name="Assessment of unclassified curtailments",
                x=data_over_time["Month"],
                y=[0 for i in range(len(data_over_time.index))],
                mode='markers',
                marker={'color': data_over_time["Unclassified curtailments KPI Colors"].tolist(),'size': 12},
                showlegend=False
            ))



        availability_kpis_figure.update_yaxes(showgrid=False,showticklabels=False)
        availability_kpis_figure.update_xaxes(showgrid=False)



        availability_kpis_figure.update_layout(title=dict(text=selected_wt+' - Assessment of unclassified curtailments',x=0.5))
        
        figure_appearance={'display': 'block'}

    else:
        availability_kpis_figure=default_figure

    
    
    return availability_kpis_figure,figure_appearance












@app.callback(
        [Output('availability-kpis-datatable-2', 'data'),
         Output('availability-kpis-datatable-2', 'columns'),
         Output("period-container","style"),
         Output("availability-kpis-datatable-2-container","style"),
         Output('period','children'),
         ], [Input('availability-kpis-datatable', 'active_cell'),
         Input('availability-kpis-graph','clickData')])
def availability_kpis_graph_2(active_cell,clickData):
    
    selected_wt=current_availability_kpis["Wind turbine"].iloc[active_cell["row_id"]]

    if clickData is not None:
        selected_time=clickData['points'][0]['x']
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        selected_time=selected_time+relativedelta(months=-1)
        selected_time=selected_time.strftime("%B %Y")

    table_appearance={'display': 'none'}


    if active_cell["column_id"]=="Monthly time-based System Avail. KPI":
        
        try:
            data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["AllStops3"].iloc[0]
            data=pd.DataFrame(data)
            data.index=range(len(data.index))
            data["Start"]=pd.to_datetime(data["Start"], format='%d/%m/%Y %H:%M')
            data["End"]=pd.to_datetime(data["End"], format='%d/%m/%Y %H:%M')
            data["Duration"]=data["End"]-data["Start"]
            data["Duration"]=data["Duration"].apply(lambda x: x.total_seconds())
            data["Duration"]=data["Duration"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Start"]=data["Start"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["End"]=data["End"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["% Time"]=data["% Time"].apply(lambda x: round(x,2))
            data["Duration KPI"]=data["% Time"].apply(lambda x: "🔴" if x>3 else("🟠" if x>1 else ("🟢" if x<=1 else "⚫")))
            data=data[["Duration KPI","Start","End","Duration","% Time","Code","Message","Category","Global contract category"]]
            data["id"]=range(len(data.index))
        
            columns=[{"name": i, "id": i} for i in ["Duration KPI","Start","End","Duration","% Time","Code","Message","Category","Global contract category"]]
            
            if data.shape[0]>0:
                table_appearance={'display': 'block'}
                text=selected_time+" events"
            else:
                table_appearance={'display': 'none'}
                text=selected_time+" selected - no events to display"
            
        except:
            data=pd.DataFrame(columns=["Duration KPI","Start","End","Duration","% Time","Code","Message","Category","Global contract category"])
            columns=[{"name": i, "id": i} for i in ["Duration KPI","Start","End","Duration","% Time","Code","Message","Category","Global contract category"]]
            text=selected_time+" selected - no events to display"

    elif active_cell["column_id"]=="Monthly production-based System Avail. KPI":

        try:
            data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["AllStops3"].iloc[0]
            data=pd.DataFrame(data)
            data.index=range(len(data.index))
            data["Start"]=pd.to_datetime(data["Start"], format='%d/%m/%Y %H:%M')
            data["End"]=pd.to_datetime(data["End"], format='%d/%m/%Y %H:%M')
            data["Duration"]=data["End"]-data["Start"]
            data["Duration"]=data["Duration"].apply(lambda x: x.total_seconds())
            data["Duration"]=data["Duration"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Start"]=data["Start"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["End"]=data["End"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["% Loss"]=data["% Loss"].apply(lambda x: round(x,2))
            data["Loss KPI"]=data["% Loss"].apply(lambda x: "🔴" if x>3 else("🟠" if x>1 else ("🟢" if x<=1 else "⚫")))
            data=data[["Loss KPI","Start","End","Duration","% Loss","Code","Message","Category","Global contract category"]]
            data["id"]=range(len(data.index))
        
            columns=[{"name": i, "id": i} for i in ["Loss KPI","Start","End","Duration","% Loss","Code","Message","Category","Global contract category"]]
            if data.shape[0]>0:
                table_appearance={'display': 'block'}
                text=selected_time+" events"
            else:
                table_appearance={'display': 'none'}
                text=selected_time+" selected - no events to display"

        except:
            data=pd.DataFrame(columns=["Loss KPI","Start","End","Duration","% Time","Code","Message","Category","Global contract category"])
            columns=[{"name": i, "id": i} for i in ["Loss KPI","Start","End","Duration","% Loss","Code","Message","Category","Global contract category"]]
            text=selected_time+" selected - no events to display"
        
    elif active_cell["column_id"]=="Monthly Year-To-Date time-based System Avail. KPI":

        try:
            data=restructured_ytd_availability_kpis.loc[(restructured_ytd_availability_kpis["Wind turbine"]==selected_wt)&(restructured_ytd_availability_kpis["Date"]==selected_time)]["AllStops3"].iloc[0]
            data=pd.DataFrame(data)
            data.index=range(len(data.index))
            data["Start"]=pd.to_datetime(data["Start"], format='%d/%m/%Y %H:%M')
            data["End"]=pd.to_datetime(data["End"], format='%d/%m/%Y %H:%M')
            data["Duration"]=data["End"]-data["Start"]
            data["Duration"]=data["Duration"].apply(lambda x: x.total_seconds())
            data["Duration"]=data["Duration"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Start"]=data["Start"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["End"]=data["End"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["% Time"]=data["% Time"].apply(lambda x: round(x,2))
            data["Duration KPI"]=data["% Time"].apply(lambda x: "🔴" if x>3 else("🟠" if x>1 else ("🟢" if x<=1 else "⚫")))
            data=data[["Duration KPI","Start","End","Duration","% Time","Code","Message","Category","Global contract category"]]
            data["id"]=range(len(data.index))
        
            columns=[{"name": i, "id": i} for i in ["Duration KPI","Start","End","Duration","% Time","Code","Message","Category","Global contract category"]]
            if data.shape[0]>0:
                table_appearance={'display': 'block'}
                text=selected_time+" events"
            else:
                table_appearance={'display': 'none'}
                text=selected_time+" selected - no events to display"

        except:
            data=pd.DataFrame(columns=["Duration KPI","Start","End","Duration","% Time","Code","Message","Category","Global contract category"])
            columns=[{"name": i, "id": i} for i in ["Duration KPI","Start","End","Duration","% Time","Code","Message","Category","Global contract category"]]
            text=selected_time+" selected - no events to display"

    elif active_cell["column_id"]=="Monthly Year-To-Date production-based System Avail. KPI":

        try:
            data=restructured_ytd_availability_kpis.loc[(restructured_ytd_availability_kpis["Wind turbine"]==selected_wt)&(restructured_ytd_availability_kpis["Date"]==selected_time)]["AllStops3"].iloc[0]
            data=pd.DataFrame(data)
            data.index=range(len(data.index))
            data["Start"]=pd.to_datetime(data["Start"], format='%d/%m/%Y %H:%M')
            data["End"]=pd.to_datetime(data["End"], format='%d/%m/%Y %H:%M')
            data["Duration"]=data["End"]-data["Start"]
            data["Duration"]=data["Duration"].apply(lambda x: x.total_seconds())
            data["Duration"]=data["Duration"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Start"]=data["Start"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["End"]=data["End"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["% Loss"]=data["% Loss"].apply(lambda x: round(x,2))
            data["Loss KPI"]=data["% Loss"].apply(lambda x: "🔴" if x>3 else("🟠" if x>1 else ("🟢" if x<=1 else "⚫")))
            data=data[["Loss KPI","Start","End","Duration","% Loss","Code","Message","Category","Global contract category"]]
            data["id"]=range(len(data.index))
        
            columns=[{"name": i, "id": i} for i in ["Loss KPI","Start","End","Duration","% Loss","Code","Message","Category","Global contract category"]]
            if data.shape[0]>0:
                table_appearance={'display': 'block'}
                text=selected_time+" events"
            else:
                table_appearance={'display': 'none'}
                text=selected_time+" selected - no events to display"

        except:
            data=pd.DataFrame(columns=["Loss KPI","Start","End","Duration","% Time","Code","Message","Category","Global contract category"])
            columns=[{"name": i, "id": i} for i in ["Loss KPI","Start","End","Duration","% Loss","Code","Message","Category","Global contract category"]]
            text=selected_time+" selected - no events to display"



    elif active_cell["column_id"]=="Monthly own stops KPI":
        
        try:

            data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["OwnStops3"].iloc[0]
        
            data=pd.DataFrame(data)
            data.index=range(len(data.index))
            data["Start"]=pd.to_datetime(data["Start"], format='%d/%m/%Y %H:%M')
            data["End"]=pd.to_datetime(data["End"], format='%d/%m/%Y %H:%M')
            data["Duration"]=data["End"]-data["Start"]
            data["Duration"]=data["Duration"].apply(lambda x: x.total_seconds())
            data["Duration"]=data["Duration"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Start"]=data["Start"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["End"]=data["End"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["Own stop KPI"]=data["Stop optimization factor"].apply(lambda x: "🔴" if x>2 else("🟠" if x>1 else ("🟢" if x<=1 else "⚫")))
            data["% Time"]=data["% Time"].apply(lambda x: round(x,2))
            data["% Loss"]=data["% Loss"].apply(lambda x: round(x,2))
            data["Stop optimization factor"]=data["Stop optimization factor"].apply(lambda x: round(x,2))
            data=data[["Own stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            data["id"]=range(len(data.index))
        
            columns=[{"name": i, "id": i} for i in ["Own stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            if data.shape[0]>0:
                table_appearance={'display': 'block'}
                text=selected_time+" events"
            else:
                table_appearance={'display': 'none'}
                text=selected_time+" selected - no events to display"

        except:

            data=pd.DataFrame(columns=["Own stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"])
            columns=[{"name": i, "id": i} for i in ["Own stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            text=selected_time+" selected - no events to display"


    elif active_cell["column_id"]=="Monthly scheduled maintenance stops KPI":
        
        try:

            data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["ScheduledMaintenanceStops3"].iloc[0]
        
            data=pd.DataFrame(data)
            data.index=range(len(data.index))
            data["Start"]=pd.to_datetime(data["Start"], format='%d/%m/%Y %H:%M')
            data["End"]=pd.to_datetime(data["End"], format='%d/%m/%Y %H:%M')
            data["Duration"]=data["End"]-data["Start"]
            data["Duration"]=data["Duration"].apply(lambda x: x.total_seconds())
            data["Duration"]=data["Duration"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Start"]=data["Start"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["End"]=data["End"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["Scheduled maintenance stop KPI"]=data["Stop optimization factor"].apply(lambda x: "🔴" if x>2 else("🟠" if x>1 else ("🟢" if x<=1 else "⚫")))
            data["% Time"]=data["% Time"].apply(lambda x: round(x,2))
            data["% Loss"]=data["% Loss"].apply(lambda x: round(x,2))
            data["Stop optimization factor"]=data["Stop optimization factor"].apply(lambda x: round(x,2))
            data=data[["Scheduled maintenance stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            data["id"]=range(len(data.index))
        
            columns=[{"name": i, "id": i} for i in ["Scheduled maintenance stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            if data.shape[0]>0:
                table_appearance={'display': 'block'}
                text=selected_time+" events"
            else:
                table_appearance={'display': 'none'}
                text=selected_time+" selected - no events to display"

        except:

            data=pd.DataFrame(columns=["Scheduled maintenance stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"])
            columns=[{"name": i, "id": i} for i in ["Scheduled maintenance stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            text=selected_time+" selected - no events to display"




    elif active_cell["column_id"]=="Monthly Year-To-Date own stops KPI":
        
        try:

            data=restructured_ytd_availability_kpis.loc[(restructured_ytd_availability_kpis["Wind turbine"]==selected_wt)&(restructured_ytd_availability_kpis["Date"]==selected_time)]["OwnStops3"].iloc[0]
        
            data=pd.DataFrame(data)
            data.index=range(len(data.index))
            data["Start"]=pd.to_datetime(data["Start"], format='%d/%m/%Y %H:%M')
            data["End"]=pd.to_datetime(data["End"], format='%d/%m/%Y %H:%M')
            data["Duration"]=data["End"]-data["Start"]
            data["Duration"]=data["Duration"].apply(lambda x: x.total_seconds())
            data["Duration"]=data["Duration"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Start"]=data["Start"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["End"]=data["End"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["Own stop KPI"]=data["Stop optimization factor"].apply(lambda x: "🔴" if x>2 else("🟠" if x>1 else ("🟢" if x<=1 else "⚫")))
            data["% Time"]=data["% Time"].apply(lambda x: round(x,2))
            data["% Loss"]=data["% Loss"].apply(lambda x: round(x,2))
            data["Stop optimization factor"]=data["Stop optimization factor"].apply(lambda x: round(x,2))
            data=data[["Own stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            data["id"]=range(len(data.index))
        
            columns=[{"name": i, "id": i} for i in ["Own stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            if data.shape[0]>0:
                table_appearance={'display': 'block'}
                text=selected_time+" events"
            else:
                table_appearance={'display': 'none'}
                text=selected_time+" selected - no events to display"

        except:

            data=pd.DataFrame(columns=["Own stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"])
            columns=[{"name": i, "id": i} for i in ["Own stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            text=selected_time+" selected - no events to display"


    elif active_cell["column_id"]=="Monthly Year-To-Date scheduled maintenance stops KPI":
        
        try:

            data=restructured_ytd_availability_kpis.loc[(restructured_ytd_availability_kpis["Wind turbine"]==selected_wt)&(restructured_ytd_availability_kpis["Date"]==selected_time)]["ScheduledMaintenanceStops3"].iloc[0]
        
            data=pd.DataFrame(data)
            data.index=range(len(data.index))
            data["Start"]=pd.to_datetime(data["Start"], format='%d/%m/%Y %H:%M')
            data["End"]=pd.to_datetime(data["End"], format='%d/%m/%Y %H:%M')
            data["Duration"]=data["End"]-data["Start"]
            data["Duration"]=data["Duration"].apply(lambda x: x.total_seconds())
            data["Duration"]=data["Duration"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Start"]=data["Start"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["End"]=data["End"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["Scheduled maintenance stop KPI"]=data["Stop optimization factor"].apply(lambda x: "🔴" if x>2 else("🟠" if x>1 else ("🟢" if x<=1 else "⚫")))
            data["% Time"]=data["% Time"].apply(lambda x: round(x,2))
            data["% Loss"]=data["% Loss"].apply(lambda x: round(x,2))
            data["Stop optimization factor"]=data["Stop optimization factor"].apply(lambda x: round(x,2))
            data=data[["Scheduled maintenance stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            data["id"]=range(len(data.index))
        
            columns=[{"name": i, "id": i} for i in ["Scheduled maintenance stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            if data.shape[0]>0:
                table_appearance={'display': 'block'}
                text=selected_time+" events"
            else:
                table_appearance={'display': 'none'}
                text=selected_time+" selected - no events to display"

        except:

            data=pd.DataFrame(columns=["Scheduled maintenance stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"])
            columns=[{"name": i, "id": i} for i in ["Scheduled maintenance stop KPI","Start","End","Duration","% Time","% Loss","Stop optimization factor","Code","Message","Category","Global contract category"]]
            text=selected_time+" selected - no events to display"


    elif active_cell["column_id"]=="Status Codes KPI":
        
        try:

            data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["Status Codes KPI Data"].iloc[0]
        
            data=pd.DataFrame.from_dict(data,orient='index')
            data=data.loc[((data["Count"]>data["Count P95"]) & (data["Count"]>2))|((data["Duration"]>data["Duration P95"]) & (data["Duration"]>60*60))]
            data["Code"]=data.index
            data.index=range(len(data.index))
            data["Status Code KPI"]=data["Color"]
            data=data[["Status Code KPI","Code","Message","Category","Global contract category","Duration","Duration P95","Duration P99","Count","Count P95","Count P99"]]
            data["Duration"]=data["Duration"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Duration P95"]=data["Duration P95"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Duration P99"]=data["Duration P99"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data=data.sort_values(by=["Code"],key=natsort_keygen())
            data["id"]=range(len(data.index))
        
            columns=[{"name": i, "id": i} for i in ["Status Code KPI","Code","Message","Category","Global contract category","Duration","Duration P95","Duration P99","Count","Count P95","Count P99"]]
            if data.shape[0]>0:
                table_appearance={'display': 'block'}
                text=selected_time+" events"
            else:
                table_appearance={'display': 'none'}
                text=selected_time+" selected - no events to display"

        except:

            data=pd.DataFrame(columns=["Status Code KPI","Code","Message","Category","Global contract category","Duration","Duration P95","Duration P99","Count","Count P95","Count P99"])
            columns=[{"name": i, "id": i} for i in ["Status Code KPI","Code","Message","Category","Global contract category","Duration","Duration P95","Duration P99","Count","Count P95","Count P99"]]
            text=selected_time+" selected - no events to display"

    elif active_cell["column_id"]=="Warnings KPI":
        
        try:

            data=availability_kpis.loc[(availability_kpis["Wind turbine"]==selected_wt)&(availability_kpis["Date"]==selected_time)]["Warnings KPI Data"].iloc[0]
        
            data=pd.DataFrame.from_dict(data,orient='index')
            data=data.loc[((data["Count"]>data["Count P95"]) & (data["Count"]>2))|((data["Duration"]>data["Duration P95"]) & (data["Duration"]>60*60))]
            data["Code"]=data.index
            data.index=range(len(data.index))
            data["Warning KPI"]=data["Color"]
            data=data[["Warning KPI","Code","Message","Category","Global contract category","Duration","Duration P95","Duration P99","Count","Count P95","Count P99"]]
            data["Duration"]=data["Duration"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Duration P95"]=data["Duration P95"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Duration P99"]=data["Duration P99"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data=data.sort_values(by=["Code"],key=natsort_keygen())
            data["id"]=range(len(data.index))
        
            columns=[{"name": i, "id": i} for i in ["Warning KPI","Code","Message","Category","Global contract category","Duration","Duration P95","Duration P99","Count","Count P95","Count P99"]]
            if data.shape[0]>0:
                table_appearance={'display': 'block'}
                text=selected_time+" events"
            else:
                table_appearance={'display': 'none'}
                text=selected_time+" selected - no events to display"

        except:

            data=pd.DataFrame(columns=["Warning KPI","Code","Message","Category","Global contract category","Duration","Duration P95","Duration P99","Count","Count P95","Count P99"])
            columns=[{"name": i, "id": i} for i in ["Warning KPI","Code","Message","Category","Global contract category","Duration","Duration P95","Duration P99","Count","Count P95","Count P99"]]
            text=selected_time+" selected - no events to display"


    elif active_cell["column_id"]=="Unclassified curtailments KPI":
        
        try:

            data=kpis.loc[(kpis["Wind turbine"]==selected_wt)&(kpis["Month"]==selected_time)]["Unclassified curtailment periods"].iloc[0]
            data=pd.DataFrame(data)
            data["Start"]=pd.to_datetime(data["Start"], format='%d/%m/%Y %H:%M')
            data["End"]=pd.to_datetime(data["End"], format='%d/%m/%Y %H:%M')
            data["Duration"]=data["End"]-data["Start"]
            data["Duration"]=data["Duration"].apply(lambda x: x.total_seconds())
            data["Duration"]=data["Duration"].apply(lambda x: str(humanfriendly.format_timespan(x)))
            data["Start"]=data["Start"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["End"]=data["End"].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
            data["id"]=range(len(data.index))
        
            columns=[{"name": i, "id": i} for i in ["Start","End","Duration"]]
            if data.shape[0]>0:
                table_appearance={'display': 'block'}
                text=selected_time+" events"
            else:
                table_appearance={'display': 'none'}
                text=selected_time+" selected - no events to display"

        except:

            data=pd.DataFrame(columns=["Start","End","Duration"])
            columns=[{"name": i, "id": i} for i in ["Start","End","Duration"]]
            text=selected_time+" selected - no events to display"

    return data.to_dict('records'),columns,{'display': 'block'},table_appearance,html.H6(text,style={'color': '#404756','text-align': 'center'})


    
    



























@app.callback(
        [Output('availability-kpis-graph-3', 'figure'),
         Output("availability-kpis-graph-3-container","style")
         ],[Input('availability-kpis-datatable', 'active_cell'),
         Input('availability-kpis-graph','clickData'),
         Input('availability-kpis-datatable-2', 'data'),
         Input('availability-kpis-datatable-2', 'active_cell')])
def availability_kpis_graph_3(active_cell,clickData,status_data,active_cell_2):
    
    if clickData is not None:
        selected_time=clickData['points'][0]['x']
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)
        selected_time=selected_time.strftime("%B %Y")

    graph_month=dt.strptime(selected_time,'%B %Y').strftime("%B")
    graph_year=dt.strptime(selected_time,'%B %Y').year

    figure_appearance={'display': 'none'}

    if active_cell["column_id"]=="Monthly time-based System Avail. KPI":

        try:
            
            availability_data=availability_kpis.loc[(availability_kpis["Date"]==selected_time)]
            availability_data=availability_data[["Manufacturer","Time-based System Avail."]]
            availability_data=availability_data.groupby(["Manufacturer"]).mean()
            availability_data.sort_index(inplace=True)

            availability_kpis_figure_3 = go.Figure(data=[
                go.Bar(name="Time-based System Avail.", x=availability_data.index, y=availability_data["Time-based System Avail."],showlegend=False),
            ])

            availability_kpis_figure_3.update_yaxes(range=[90,100],title="Time-based System Avail. (%)",showgrid=False)
            availability_kpis_figure_3.update_xaxes(showgrid=False)
    
            availability_kpis_figure_3.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - Time-based System Avail. by manufacturer',x=0.5))
            figure_appearance={'display': 'block'}


        except:
            availability_kpis_figure_3=default_figure
        
    elif active_cell["column_id"]=="Monthly Year-To-Date time-based System Avail. KPI":

        try:
            
            availability_data=restructured_ytd_availability_kpis.loc[(restructured_ytd_availability_kpis["Date"]==selected_time)]
            availability_data=availability_data[["Manufacturer","Time-based System Avail."]]
            availability_data=availability_data.groupby(["Manufacturer"]).mean()
            availability_data.sort_index(inplace=True)

            availability_kpis_figure_3 = go.Figure(data=[
                go.Bar(name="Time-based System Avail.", x=availability_data.index, y=availability_data["Time-based System Avail."],showlegend=False),
            ])

            availability_kpis_figure_3.update_yaxes(range=[90,100],title="Time-based System Avail. (%)",showgrid=False)
            availability_kpis_figure_3.update_xaxes(showgrid=False)
    
            availability_kpis_figure_3.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - YTD time-based System Avail. by manufacturer',x=0.5))
            figure_appearance={'display': 'block'}


        except:
            availability_kpis_figure_3=default_figure
        
    
    elif active_cell["column_id"]=="Monthly production-based System Avail. KPI":

        try:
            
            availability_data=availability_kpis.loc[(availability_kpis["Date"]==selected_time)]
            availability_data=availability_data[["Manufacturer","Production-based System Avail.","PotentialPower"]]
            availability_data["Weighted production-based System Avail."]=availability_data["Production-based System Avail."]*availability_data["PotentialPower"]
            restructured_availability_data=pd.DataFrame(index=sorted(list(set(availability_data["Manufacturer"].tolist()))),columns=["Production-based System Avail."])
            for manufacturer in restructured_availability_data.index:
                tmp=availability_data.loc[availability_data["Manufacturer"]==manufacturer]
                restructured_availability_data.at[manufacturer,"Production-based System Avail."]=tmp["Weighted production-based System Avail."].sum()/tmp["PotentialPower"].sum()

            availability_kpis_figure_3 = go.Figure(data=[
                go.Bar(name="Time-based System Avail.", x=restructured_availability_data.index, y=restructured_availability_data["Production-based System Avail."],showlegend=False),
            ])

            availability_kpis_figure_3.update_yaxes(range=[90,100],title="Production-based System Avail. (%)",showgrid=False)
            availability_kpis_figure_3.update_xaxes(showgrid=False)
    
            availability_kpis_figure_3.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - Production-based System Avail. by manufacturer',x=0.5))
            figure_appearance={'display': 'block'}


        except:
            availability_kpis_figure_3=default_figure

    elif active_cell["column_id"]=="Monthly Year-To-Date production-based System Avail. KPI":

        try:
            
            availability_data=restructured_ytd_availability_kpis.loc[(restructured_ytd_availability_kpis["Date"]==selected_time)]
            availability_data=availability_data[["Manufacturer","Production-based System Avail.","PotentialPower"]]
            availability_data["Weighted production-based System Avail."]=availability_data["Production-based System Avail."]*availability_data["PotentialPower"]
            restructured_availability_data=pd.DataFrame(index=sorted(list(set(availability_data["Manufacturer"].tolist()))),columns=["Production-based System Avail."])
            for manufacturer in restructured_availability_data.index:
                tmp=availability_data.loc[availability_data["Manufacturer"]==manufacturer]
                restructured_availability_data.at[manufacturer,"Production-based System Avail."]=tmp["Weighted production-based System Avail."].sum()/tmp["PotentialPower"].sum()

            availability_kpis_figure_3 = go.Figure(data=[
                go.Bar(name="Time-based System Avail.", x=restructured_availability_data.index, y=restructured_availability_data["Production-based System Avail."],showlegend=False),
            ])

            availability_kpis_figure_3.update_yaxes(range=[90,100],title="Production-based System Avail. (%)",showgrid=False)
            availability_kpis_figure_3.update_xaxes(showgrid=False)
    
            availability_kpis_figure_3.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - YTD production-based System Avail. by manufacturer',x=0.5))
            figure_appearance={'display': 'block'}


        except:
            availability_kpis_figure_3=default_figure
        



    elif active_cell["column_id"]=="Monthly own stops KPI":

        try:
            
            months=list(set(availability_kpis["Date"].tolist()))
            months=[dt.strptime(i,'%B %Y') for i in months]
            months=sorted(months)
            months=[i.strftime('%B %Y') for i in months]
            good_bad_stops_over_time=pd.DataFrame(index=months,columns=["Good","Bad","Very bad"])
            for month in months:
                monthly_good_bad_stops_data=availability_kpis.loc[availability_kpis["Date"]==month]
                monthly_good_bad_stops_data=monthly_good_bad_stops_data[["OwnStops3"]]
                monthly_good_bad_stops_data["OwnStops3"]=monthly_good_bad_stops_data["OwnStops3"].apply(lambda x: [event["Stop optimization factor"] for event in x])
                monthly_good_bad_stops=[item for sublist in monthly_good_bad_stops_data["OwnStops3"].tolist() for item in sublist]
                if len(monthly_good_bad_stops)>0:
                    value=sum(item<=1 for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Good"]=value
                    value=sum((item>1 and item<=2) for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Bad"]=value
                    value=sum(item>2 for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Very bad"]=value
                else:
                    good_bad_stops_over_time.at[month,"Good"]=0
                    good_bad_stops_over_time.at[month,"Bad"]=0
                    good_bad_stops_over_time.at[month,"Very bad"]=0

            availability_kpis_figure_3 = go.Figure(data=[
                go.Bar(name="Good", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Good"],marker=go.bar.Marker(color='green')),
                go.Bar(name="Bad", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Bad"],marker=go.bar.Marker(color='orange')),
                go.Bar(name="Very bad", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Very bad"],marker=go.bar.Marker(color='red')),
            ])

            availability_kpis_figure_3.update_layout(barmode='stack')

            availability_kpis_figure_3.update_yaxes(title="Good/bad own stops (-)",showgrid=False)
            availability_kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            availability_kpis_figure_3.update_layout(title=dict(text='All assets - Good and bad own stops over time',x=0.5))
            figure_appearance={'display': 'block'}


        except:
            availability_kpis_figure_3=default_figure


    elif active_cell["column_id"]=="Monthly scheduled maintenance stops KPI":

        try:
            
            months=list(set(availability_kpis["Date"].tolist()))
            months=[dt.strptime(i,'%B %Y') for i in months]
            months=sorted(months)
            months=[i.strftime('%B %Y') for i in months]
            good_bad_stops_over_time=pd.DataFrame(index=months,columns=["Good","Bad","Very bad"])
            for month in months:
                monthly_good_bad_stops_data=availability_kpis.loc[availability_kpis["Date"]==month]
                monthly_good_bad_stops_data=monthly_good_bad_stops_data[["ScheduledMaintenanceStops3"]]
                monthly_good_bad_stops_data["ScheduledMaintenanceStops3"]=monthly_good_bad_stops_data["ScheduledMaintenanceStops3"].apply(lambda x: [event["Stop optimization factor"] for event in x])
                monthly_good_bad_stops=[item for sublist in monthly_good_bad_stops_data["ScheduledMaintenanceStops3"].tolist() for item in sublist]
                if len(monthly_good_bad_stops)>0:
                    value=sum(item<=1 for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Good"]=value
                    value=sum((item>1 and item<=2) for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Bad"]=value
                    value=sum(item>2 for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Very bad"]=value
                else:
                    good_bad_stops_over_time.at[month,"Good"]=0
                    good_bad_stops_over_time.at[month,"Bad"]=0
                    good_bad_stops_over_time.at[month,"Very bad"]=0

            availability_kpis_figure_3 = go.Figure(data=[
                go.Bar(name="Good", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Good"],marker=go.bar.Marker(color='green')),
                go.Bar(name="Bad", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Bad"],marker=go.bar.Marker(color='orange')),
                go.Bar(name="Very bad", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Very bad"],marker=go.bar.Marker(color='red')),
            ])

            availability_kpis_figure_3.update_layout(barmode='stack')

            availability_kpis_figure_3.update_yaxes(title="Good/bad scheduled maintenance stops (-)",showgrid=False)
            availability_kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            availability_kpis_figure_3.update_layout(title=dict(text='All assets - Good and bad scheduled maintenance stops over time',x=0.5))
            figure_appearance={'display': 'block'}


        except:
            availability_kpis_figure_3=default_figure



    elif active_cell["column_id"]=="Monthly Year-To-Date own stops KPI":

        try:
            
            months=list(set(restructured_ytd_availability_kpis["Date"].tolist()))
            months=[dt.strptime(i,'%B %Y') for i in months]
            months=sorted(months)
            months=[i.strftime('%B %Y') for i in months]
            good_bad_stops_over_time=pd.DataFrame(index=months,columns=["Good","Bad","Very bad"])
            for month in months:
                monthly_good_bad_stops_data=restructured_ytd_availability_kpis.loc[restructured_ytd_availability_kpis["Date"]==month]
                monthly_good_bad_stops_data=monthly_good_bad_stops_data[["OwnStops3"]]
                monthly_good_bad_stops_data["OwnStops3"]=monthly_good_bad_stops_data["OwnStops3"].apply(lambda x: [event["Stop optimization factor"] for event in x])
                monthly_good_bad_stops=[item for sublist in monthly_good_bad_stops_data["OwnStops3"].tolist() for item in sublist]
                if len(monthly_good_bad_stops)>0:
                    value=sum(item<=1 for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Good"]=value
                    value=sum((item>1 and item<=2) for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Bad"]=value
                    value=sum(item>2 for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Very bad"]=value
                else:
                    good_bad_stops_over_time.at[month,"Good"]=0
                    good_bad_stops_over_time.at[month,"Bad"]=0
                    good_bad_stops_over_time.at[month,"Very bad"]=0

            availability_kpis_figure_3 = go.Figure(data=[
                go.Bar(name="Good", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Good"],marker=go.bar.Marker(color='green')),
                go.Bar(name="Bad", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Bad"],marker=go.bar.Marker(color='orange')),
                go.Bar(name="Very bad", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Very bad"],marker=go.bar.Marker(color='red')),
            ])

            availability_kpis_figure_3.update_layout(barmode='stack')

            availability_kpis_figure_3.update_yaxes(title="Good/bad own stops (-)",showgrid=False)
            availability_kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            availability_kpis_figure_3.update_layout(title=dict(text='All assets - YTD good and bad own stops over time',x=0.5))
            figure_appearance={'display': 'block'}


        except:
            availability_kpis_figure_3=default_figure



    elif active_cell["column_id"]=="Monthly Year-To-Date scheduled maintenance stops KPI":

        try:
            
            months=list(set(restructured_ytd_availability_kpis["Date"].tolist()))
            months=[dt.strptime(i,'%B %Y') for i in months]
            months=sorted(months)
            months=[i.strftime('%B %Y') for i in months]
            good_bad_stops_over_time=pd.DataFrame(index=months,columns=["Good","Bad","Very bad"])
            for month in months:
                monthly_good_bad_stops_data=restructured_ytd_availability_kpis.loc[restructured_ytd_availability_kpis["Date"]==month]
                monthly_good_bad_stops_data=monthly_good_bad_stops_data[["ScheduledMaintenanceStops3"]]
                monthly_good_bad_stops_data["ScheduledMaintenanceStops3"]=monthly_good_bad_stops_data["ScheduledMaintenanceStops3"].apply(lambda x: [event["Stop optimization factor"] for event in x])
                monthly_good_bad_stops=[item for sublist in monthly_good_bad_stops_data["ScheduledMaintenanceStops3"].tolist() for item in sublist]
                if len(monthly_good_bad_stops)>0:
                    value=sum(item<=1 for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Good"]=value
                    value=sum((item>1 and item<=2) for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Bad"]=value
                    value=sum(item>2 for item in monthly_good_bad_stops)
                    good_bad_stops_over_time.at[month,"Very bad"]=value
                else:
                    good_bad_stops_over_time.at[month,"Good"]=0
                    good_bad_stops_over_time.at[month,"Bad"]=0
                    good_bad_stops_over_time.at[month,"Very bad"]=0

            availability_kpis_figure_3 = go.Figure(data=[
                go.Bar(name="Good", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Good"],marker=go.bar.Marker(color='green')),
                go.Bar(name="Bad", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Bad"],marker=go.bar.Marker(color='orange')),
                go.Bar(name="Very bad", x=good_bad_stops_over_time.index, y=good_bad_stops_over_time["Very bad"],marker=go.bar.Marker(color='red')),
            ])

            availability_kpis_figure_3.update_layout(barmode='stack')

            availability_kpis_figure_3.update_yaxes(title="Good/bad scheduled maintenance stops (-)",showgrid=False)
            availability_kpis_figure_3.update_xaxes(showgrid=False)
    
    
    
            availability_kpis_figure_3.update_layout(title=dict(text='All assets - YTD good and bad scheduled maintenance stops over time',x=0.5))
            figure_appearance={'display': 'block'}


        except:
            availability_kpis_figure_3=default_figure




    elif active_cell["column_id"]=="Status Codes KPI":
        
        
        try:
            
            current_wt=current_availability_kpis["Wind turbine"].iloc[active_cell["row_id"]]
            current_type=kpis.loc[kpis["Wind turbine"]==current_wt]["Wind turbine type"].iloc[0]
            wts_of_current_type=kpis.loc[kpis["Wind turbine type"]==current_type]
            wts_of_current_type=list(set(wts_of_current_type["Wind turbine"].to_list()))

            status_data = pd.DataFrame(status_data)

            current_status_code=status_data["Code"].iloc[active_cell_2["row_id"]]
            data_for_plotting=availability_kpis.loc[availability_kpis["Wind turbine"].isin(wts_of_current_type)]
            data_for_plotting.index=range(len(data_for_plotting.index))
            events_data=pd.DataFrame()
            for item in range(len(data_for_plotting.index)):
                tmp=data_for_plotting["details"].iloc[item]
                tmp=pd.DataFrame(tmp)
                tmp["WT"]=data_for_plotting["Wind turbine"].iloc[item]
                events_data=pd.concat([events_data,tmp])
            
            events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            events_data["End"]=events_data.apply(lambda row: row["Start"]+timedelta(seconds=21600) if (row["End"]-row["Start"]).total_seconds()<21600 else row["End"],axis=1)
            events_data=events_data[["WT","Code","Start","End"]]
            events_data=events_data.loc[events_data["Code"]==current_status_code]
            events_data.index=range(len(events_data.index))
            
            for wt in wts_of_current_type:
                events_data.loc[len(events_data)] = [wt,current_status_code,np.nan,np.nan]

            category_order=natsorted(wts_of_current_type)
            
            availability_kpis_figure_3 = px.timeline(events_data,x_start="Start",x_end="End",y="WT",range_x=[dt.today().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-12), dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)],category_orders={'WT': category_order},color_discrete_sequence=["red"],labels={"WT":"","Start":"Start","End":"End"})    

            availability_kpis_figure_3.update_yaxes(showgrid=False)
            availability_kpis_figure_3.update_xaxes(showgrid=False)
            height=max(450,30*len(wts_of_current_type))
            availability_kpis_figure_3.update_layout(title=dict(text=str(current_type)+" - Occurence of status " + str(current_status_code) + " over time",x=0.5),height=height)

            figure_appearance={'display': 'block'}



        except:
            availability_kpis_figure_3=default_figure











    elif active_cell["column_id"]=="Warnings KPI":
                
        try:

            current_wt=current_availability_kpis["Wind turbine"].iloc[active_cell["row_id"]]
            current_type=kpis.loc[kpis["Wind turbine"]==current_wt]["Wind turbine type"].iloc[0]
            wts_of_current_type=kpis.loc[kpis["Wind turbine type"]==current_type]
            wts_of_current_type=list(set(wts_of_current_type["Wind turbine"].to_list()))

            status_data = pd.DataFrame(status_data)

            current_status_code=status_data["Code"].iloc[active_cell_2["row_id"]]
            data_for_plotting=availability_kpis.loc[availability_kpis["Wind turbine"].isin(wts_of_current_type)]
            data_for_plotting.index=range(len(data_for_plotting.index))
            events_data=pd.DataFrame()
            for item in range(len(data_for_plotting.index)):
                tmp=data_for_plotting["details"].iloc[item]
                tmp=pd.DataFrame(tmp)
                tmp["WT"]=data_for_plotting["Wind turbine"].iloc[item]
                events_data=pd.concat([events_data,tmp])
            
            events_data["Start"]=pd.to_datetime(events_data["Start"],format='%d/%m/%Y %H:%M')
            events_data["End"]=pd.to_datetime(events_data["End"], format='%d/%m/%Y %H:%M')
            events_data["End"]=events_data.apply(lambda row: row["Start"]+timedelta(seconds=21600) if (row["End"]-row["Start"]).total_seconds()<21600 else row["End"],axis=1)
            events_data=events_data[["WT","Code","Start","End"]]
            events_data=events_data.loc[events_data["Code"]==current_status_code]
            events_data.index=range(len(events_data.index))
            
            for wt in wts_of_current_type:
                events_data.loc[len(events_data)] = [wt,current_status_code,np.nan,np.nan]

            category_order=natsorted(wts_of_current_type)
            
            availability_kpis_figure_3 = px.timeline(events_data,x_start="Start",x_end="End",y="WT",range_x=[dt.today().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-12), dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)],category_orders={'WT': category_order},color_discrete_sequence=["red"],labels={"WT":"","Start":"Start","End":"End"})    

            availability_kpis_figure_3.update_yaxes(showgrid=False)
            availability_kpis_figure_3.update_xaxes(showgrid=False)
            height=max(450,30*len(wts_of_current_type))
            availability_kpis_figure_3.update_layout(title=dict(text=str(current_type)+" - Occurence of warning " + str(current_status_code) + " over time",x=0.5),height=height)

            figure_appearance={'display': 'block'}



        except:
            availability_kpis_figure_3=default_figure






    else:
        availability_kpis_figure_3=default_figure










    
    
    return availability_kpis_figure_3,figure_appearance


















@app.callback(
        [Output('availability-kpis-graph-4', 'figure'),
         Output("availability-kpis-graph-4-container","style")
         ],[Input('availability-kpis-datatable', 'active_cell'),
         Input('availability-kpis-graph','clickData')])
def availability_kpis_graph_4(active_cell,clickData):
    
    if clickData is not None:
        selected_time=clickData['points'][0]['x']
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)
        selected_time=selected_time.strftime("%B %Y")

    graph_month=dt.strptime(selected_time,'%B %Y').strftime("%B")
    graph_year=dt.strptime(selected_time,'%B %Y').year
    figure_appearance={'display': 'none'}


    if active_cell["column_id"]=="Monthly scheduled maintenance stops KPI":

        try:
            
            manufacturers=sorted(list(set(availability_kpis["Manufacturer"].tolist())))
            good_bad_stops_by_manufacturer=pd.DataFrame(index=manufacturers,columns=["Good","Bad","Very bad"])
            for manufacturer in manufacturers:
                monthly_good_bad_stops_data=availability_kpis.loc[(availability_kpis["Manufacturer"]==manufacturer) & (availability_kpis["Date"]==selected_time)]
                monthly_good_bad_stops_data=monthly_good_bad_stops_data[["ScheduledMaintenanceStops3"]]
                monthly_good_bad_stops_data["ScheduledMaintenanceStops3"]=monthly_good_bad_stops_data["ScheduledMaintenanceStops3"].apply(lambda x: [event["Stop optimization factor"] for event in x] if x!={} else [])
                monthly_good_bad_stops=[item for sublist in monthly_good_bad_stops_data["ScheduledMaintenanceStops3"].tolist() for item in sublist]
                if len(monthly_good_bad_stops)>0:
                    value=sum(item<=1 for item in monthly_good_bad_stops)
                    good_bad_stops_by_manufacturer.at[manufacturer,"Good"]=value
                    value=sum((item>1 and item<=2) for item in monthly_good_bad_stops)
                    good_bad_stops_by_manufacturer.at[manufacturer,"Bad"]=value
                    value=sum(item>2 for item in monthly_good_bad_stops)
                    good_bad_stops_by_manufacturer.at[manufacturer,"Very bad"]=value
                else:
                    good_bad_stops_by_manufacturer.at[manufacturer,"Good"]=0
                    good_bad_stops_by_manufacturer.at[manufacturer,"Bad"]=0
                    good_bad_stops_by_manufacturer.at[manufacturer,"Very bad"]=0

            availability_kpis_figure_4 = go.Figure(data=[
                go.Bar(name="Good", x=good_bad_stops_by_manufacturer.index, y=good_bad_stops_by_manufacturer["Good"],marker=go.bar.Marker(color='green')),
                go.Bar(name="Bad", x=good_bad_stops_by_manufacturer.index, y=good_bad_stops_by_manufacturer["Bad"],marker=go.bar.Marker(color='orange')),
                go.Bar(name="Very bad", x=good_bad_stops_by_manufacturer.index, y=good_bad_stops_by_manufacturer["Very bad"],marker=go.bar.Marker(color='red')),
            ])

            availability_kpis_figure_4.update_layout(barmode='stack')

            availability_kpis_figure_4.update_yaxes(title="Good/bad scheduled maintenance stops (-)",showgrid=False)
            availability_kpis_figure_4.update_xaxes(showgrid=False)
    
    
            availability_kpis_figure_4.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - Good and bad scheduled maintenance stops by manufacturer',x=0.5))
            figure_appearance={'display': 'block'}


        except:
            availability_kpis_figure_4=default_figure


    elif active_cell["column_id"]=="Monthly Year-To-Date scheduled maintenance stops KPI":

        try:
            
            manufacturers=sorted(list(set(restructured_ytd_availability_kpis["Manufacturer"].tolist())))
            good_bad_stops_by_manufacturer=pd.DataFrame(index=manufacturers,columns=["Good","Bad","Very bad"])
            for manufacturer in manufacturers:
                monthly_good_bad_stops_data=restructured_ytd_availability_kpis.loc[(restructured_ytd_availability_kpis["Manufacturer"]==manufacturer) & (restructured_ytd_availability_kpis["Date"]==selected_time)]
                monthly_good_bad_stops_data=monthly_good_bad_stops_data[["ScheduledMaintenanceStops3"]]
                monthly_good_bad_stops_data["ScheduledMaintenanceStops3"]=monthly_good_bad_stops_data["ScheduledMaintenanceStops3"].apply(lambda x: [event["Stop optimization factor"] for event in x] if x!={} else [])
                monthly_good_bad_stops=[item for sublist in monthly_good_bad_stops_data["ScheduledMaintenanceStops3"].tolist() for item in sublist]
                if len(monthly_good_bad_stops)>0:
                    value=sum(item<=1 for item in monthly_good_bad_stops)
                    good_bad_stops_by_manufacturer.at[manufacturer,"Good"]=value
                    value=sum((item>1 and item<=2) for item in monthly_good_bad_stops)
                    good_bad_stops_by_manufacturer.at[manufacturer,"Bad"]=value
                    value=sum(item>2 for item in monthly_good_bad_stops)
                    good_bad_stops_by_manufacturer.at[manufacturer,"Very bad"]=value
                else:
                    good_bad_stops_by_manufacturer.at[manufacturer,"Good"]=0
                    good_bad_stops_by_manufacturer.at[manufacturer,"Bad"]=0
                    good_bad_stops_by_manufacturer.at[manufacturer,"Very bad"]=0

            availability_kpis_figure_4 = go.Figure(data=[
                go.Bar(name="Good", x=good_bad_stops_by_manufacturer.index, y=good_bad_stops_by_manufacturer["Good"],marker=go.bar.Marker(color='green')),
                go.Bar(name="Bad", x=good_bad_stops_by_manufacturer.index, y=good_bad_stops_by_manufacturer["Bad"],marker=go.bar.Marker(color='orange')),
                go.Bar(name="Very bad", x=good_bad_stops_by_manufacturer.index, y=good_bad_stops_by_manufacturer["Very bad"],marker=go.bar.Marker(color='red')),
            ])

            availability_kpis_figure_4.update_layout(barmode='stack')

            availability_kpis_figure_4.update_yaxes(title="Good/bad scheduled maintenance stops (-)",showgrid=False)
            availability_kpis_figure_4.update_xaxes(showgrid=False)
    
    
    
            availability_kpis_figure_4.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - YTD good and bad scheduled maintenance stops by manufacturer',x=0.5))
            figure_appearance={'display': 'block'}


        except:
            availability_kpis_figure_4=default_figure


    else:
        availability_kpis_figure_4=default_figure





    return availability_kpis_figure_4,figure_appearance














@app.callback(
        [Output('availability-kpis-graph-5', 'figure'),
         Output("availability-kpis-graph-5-container","style")
         ],[Input('availability-kpis-datatable', 'active_cell')])
def availability_kpis_graph_5(active_cell):
    
    figure_appearance={'display': 'none'}

    if active_cell["column_id"]=="Monthly own stops KPI":

        try:
            
            tmp_for_JB=availability_kpis.copy()
            tmp_for_JB=tmp_for_JB[["Month","OwnStops2","DaysInMonth","PotentialPower"]]
            tmp_for_JB["Duration"]=tmp_for_JB["OwnStops2"].apply(lambda x: sum(event["Duration"] for event in x))
            tmp_for_JB["Loss"]=tmp_for_JB["OwnStops2"].apply(lambda x: sum(event["Loss"] for event in x))
            
            months_for_graph=sorted(list(set(tmp_for_JB["Month"].tolist())))
            
            vals_for_graph=pd.DataFrame(index=months_for_graph,columns=["Optimization factor"])
            
            for item in months_for_graph:
                tmp_tmp_for_JB=tmp_for_JB.loc[tmp_for_JB["Month"]==item]
                tmp_tmp_for_JB=tmp_tmp_for_JB.sum()                
                vals_for_graph.at[item,"Optimization factor"]=(tmp_tmp_for_JB.loc["Loss"]/tmp_tmp_for_JB.loc["PotentialPower"])/(tmp_tmp_for_JB.loc["Duration"]/tmp_tmp_for_JB.loc["DaysInMonth"]/60/60/24)
             
            availability_kpis_figure_5 = go.Figure()
            
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name='Assessment of own stops',
                    x=vals_for_graph.index,
                    y=vals_for_graph["Optimization factor"],
                    mode='lines+markers',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=False
                ))
    
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name="Very bad",
                    x=vals_for_graph.index,
                    y=[2 for i in vals_for_graph.index],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))
    
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name="Bad",
                    x=vals_for_graph.index,
                    y=[1 for i in vals_for_graph.index],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))
    
            availability_kpis_figure_5.update_yaxes(showgrid=False,showticklabels=False)
            availability_kpis_figure_5.update_xaxes(showgrid=False)
    
            availability_kpis_figure_5.update_layout(title=dict(text='All assets - Monthly own stops optimization factor',x=0.5))
            figure_appearance={'display': 'block'}

        except:
            availability_kpis_figure_5=default_figure

    elif active_cell["column_id"]=="Monthly scheduled maintenance stops KPI":

        try:
            
            tmp_for_JB=availability_kpis.copy()
            tmp_for_JB=tmp_for_JB[["Month","ScheduledMaintenanceStops2","DaysInMonth","PotentialPower"]]
            tmp_for_JB["Duration"]=tmp_for_JB["ScheduledMaintenanceStops2"].apply(lambda x: sum(event["Duration"] for event in x))
            tmp_for_JB["Loss"]=tmp_for_JB["ScheduledMaintenanceStops2"].apply(lambda x: sum(event["Loss"] for event in x))
            
            months_for_graph=sorted(list(set(tmp_for_JB["Month"].tolist())))
            
            vals_for_graph=pd.DataFrame(index=months_for_graph,columns=["Optimization factor"])
            
            for item in months_for_graph:
                tmp_tmp_for_JB=tmp_for_JB.loc[tmp_for_JB["Month"]==item]
                tmp_tmp_for_JB=tmp_tmp_for_JB.sum()                
                vals_for_graph.at[item,"Optimization factor"]=(tmp_tmp_for_JB.loc["Loss"]/tmp_tmp_for_JB.loc["PotentialPower"])/(tmp_tmp_for_JB.loc["Duration"]/tmp_tmp_for_JB.loc["DaysInMonth"]/60/60/24)
             
            availability_kpis_figure_5 = go.Figure()
            
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name='Assessment of scheduled maintenance stops',
                    x=vals_for_graph.index,
                    y=vals_for_graph["Optimization factor"],
                    mode='lines+markers',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=False
                ))
    
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name="Very bad",
                    x=vals_for_graph.index,
                    y=[2 for i in vals_for_graph.index],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))
    
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name="Bad",
                    x=vals_for_graph.index,
                    y=[1 for i in vals_for_graph.index],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))
    
    
            availability_kpis_figure_5.update_yaxes(showgrid=False,showticklabels=False)
            availability_kpis_figure_5.update_xaxes(showgrid=False)
    
    
    
            availability_kpis_figure_5.update_layout(title=dict(text='All assets - Monthly scheduled maintenance stops optimization factor',x=0.5))
            figure_appearance={'display': 'block'}




        except:
            availability_kpis_figure_5=default_figure


    elif active_cell["column_id"]=="Monthly Year-To-Date own stops KPI":

        try:
            
            tmp_for_JB=restructured_ytd_availability_kpis.copy()
            tmp_for_JB=tmp_for_JB[["Month","OwnStops2","DaysInMonth","PotentialPower"]]
            tmp_for_JB["Duration"]=tmp_for_JB["OwnStops2"].apply(lambda x: sum(event["Duration"] for event in x))
            tmp_for_JB["Loss"]=tmp_for_JB["OwnStops2"].apply(lambda x: sum(event["Loss"] for event in x))
            
            months_for_graph=sorted(list(set(tmp_for_JB["Month"].tolist())))
            
            vals_for_graph=pd.DataFrame(index=months_for_graph,columns=["Optimization factor"])
            
            for item in months_for_graph:
                tmp_tmp_for_JB=tmp_for_JB.loc[tmp_for_JB["Month"]==item]
                tmp_tmp_for_JB=tmp_tmp_for_JB.sum()                
                vals_for_graph.at[item,"Optimization factor"]=(tmp_tmp_for_JB.loc["Loss"]/tmp_tmp_for_JB.loc["PotentialPower"])/(tmp_tmp_for_JB.loc["Duration"]/tmp_tmp_for_JB.loc["DaysInMonth"]/60/60/24)
             
            availability_kpis_figure_5 = go.Figure()
            
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name='Assessment of own stops',
                    x=vals_for_graph.index,
                    y=vals_for_graph["Optimization factor"],
                    mode='lines+markers',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=False
                ))
    
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name="Very bad",
                    x=vals_for_graph.index,
                    y=[2 for i in vals_for_graph.index],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))
    
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name="Bad",
                    x=vals_for_graph.index,
                    y=[1 for i in vals_for_graph.index],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))
    
    
            availability_kpis_figure_5.update_yaxes(showgrid=False,showticklabels=False)
            availability_kpis_figure_5.update_xaxes(showgrid=False)
    
    
    
            availability_kpis_figure_5.update_layout(title=dict(text='All assets - Monthly YTD own stops optimization factor',x=0.5))
            figure_appearance={'display': 'block'}




        except:
            availability_kpis_figure_5=default_figure

    elif active_cell["column_id"]=="Monthly Year-To-Date scheduled maintenance stops KPI":

        try:
            
            tmp_for_JB=restructured_ytd_availability_kpis.copy()
            tmp_for_JB=tmp_for_JB[["Month","ScheduledMaintenanceStops2","DaysInMonth","PotentialPower"]]
            tmp_for_JB["Duration"]=tmp_for_JB["ScheduledMaintenanceStops2"].apply(lambda x: sum(event["Duration"] for event in x))
            tmp_for_JB["Loss"]=tmp_for_JB["ScheduledMaintenanceStops2"].apply(lambda x: sum(event["Loss"] for event in x))
            
            months_for_graph=sorted(list(set(tmp_for_JB["Month"].tolist())))
            
            vals_for_graph=pd.DataFrame(index=months_for_graph,columns=["Optimization factor"])
            
            for item in months_for_graph:
                tmp_tmp_for_JB=tmp_for_JB.loc[tmp_for_JB["Month"]==item]
                tmp_tmp_for_JB=tmp_tmp_for_JB.sum()                
                vals_for_graph.at[item,"Optimization factor"]=(tmp_tmp_for_JB.loc["Loss"]/tmp_tmp_for_JB.loc["PotentialPower"])/(tmp_tmp_for_JB.loc["Duration"]/tmp_tmp_for_JB.loc["DaysInMonth"]/60/60/24)
             
            availability_kpis_figure_5 = go.Figure()
            
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name='Assessment of scheduled maintenance stops',
                    x=vals_for_graph.index,
                    y=vals_for_graph["Optimization factor"],
                    mode='lines+markers',
                    line={
                            'shape':"spline",
                            'smoothing':1,
                            'color':'rgb(31, 119, 180)'
                            },
                    showlegend=False
                ))
    
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name="Very bad",
                    x=vals_for_graph.index,
                    y=[2 for i in vals_for_graph.index],
                    mode='lines',
                    line=dict(color="Red",dash="dashdot"),
                    showlegend=False
                ))
    
            availability_kpis_figure_5.add_trace(go.Scatter(
                    name="Bad",
                    x=vals_for_graph.index,
                    y=[1 for i in vals_for_graph.index],
                    mode='lines',
                    line=dict(color="Orange",dash="dashdot"),
                    showlegend=False
                ))
    
    
            availability_kpis_figure_5.update_yaxes(showgrid=False,showticklabels=False)
            availability_kpis_figure_5.update_xaxes(showgrid=False)
    
    
    
            availability_kpis_figure_5.update_layout(title=dict(text='All assets - Monthly YTD scheduled maintenance stops optimization factor',x=0.5))
            figure_appearance={'display': 'block'}




        except:
            availability_kpis_figure_5=default_figure

    else:
        availability_kpis_figure_5=default_figure
        
    return availability_kpis_figure_5,figure_appearance









@app.callback(
        [Output('production-kpis-graph', 'figure'),
         Output("production-kpis-graph-container","style")
         ],[Input('production-kpis-datatable', 'active_cell')])
def production_kpis_graph(active_cell):
    selected_wf=current_production_kpis["Wind farm"].iloc[active_cell["row_id"]]
    figure_appearance={'display': 'none'}
    if active_cell["column_id"]=="Monthly production KPI":
        wf_data=production_kpis.loc[production_kpis["WF"]==selected_wf]
        monthly_productions=wf_data["Monthly productions"].iloc[0]
        monthly_p50s=wf_data["Monthly P50s"].iloc[0]
        monthly_p75s=wf_data["Monthly P75s"].iloc[0]
        monthly_p25s=wf_data["Monthly P25s"].iloc[0]
        monthly_p90s=wf_data["Monthly P90s"].iloc[0]
        monthly_p10s=wf_data["Monthly P10s"].iloc[0]
        monthly_productions=pd.DataFrame.from_dict(monthly_productions, orient='index',columns=["Production"])
        monthly_p50s=pd.DataFrame.from_dict(monthly_p50s, orient='index',columns=["P50"])
        monthly_p75s=pd.DataFrame.from_dict(monthly_p75s, orient='index',columns=["P75"])
        monthly_p25s=pd.DataFrame.from_dict(monthly_p25s, orient='index',columns=["P25"])
        monthly_p90s=pd.DataFrame.from_dict(monthly_p90s, orient='index',columns=["P90"])
        monthly_p10s=pd.DataFrame.from_dict(monthly_p10s, orient='index',columns=["P10"])
        data=pd.concat([monthly_productions,monthly_p50s,monthly_p75s,monthly_p25s,monthly_p90s,monthly_p10s], axis=1)
        data=data.dropna(subset=["Production"])


        production_kpis_figure = go.Figure()
        
        production_kpis_figure.add_trace(go.Scatter(
                name='Production',
                legendgroup = 'a',
                x=data.index,
                y=data["Production"],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))

        production_kpis_figure.add_trace(go.Scatter(
                name='P50',
                legendgroup = 'b',
                x=data.index,
                y=data["P50"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Green',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        production_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'c',
                x=data.index,
                y=data["P75"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        production_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'c',
                x=data.index,
                y=data["P25"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))

        production_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'd',
                x=data.index,
                y=data["P90"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        production_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'd',
                x=data.index,
                y=data["P10"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))


        production_kpis_figure.update_yaxes(title="Energy production (MWh)",showgrid=False)
#        production_kpis_figure.update_xaxes(showgrid=False)
        production_kpis_figure.update_xaxes(showgrid=False,range=[dt.today().replace(month=1,day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-12), dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)])
        production_kpis_figure.update_layout(title=dict(text=str(selected_wf)+' - Historical productions against budget',x=0.5))
        figure_appearance={'display': 'block'}
        
        
        




    elif active_cell["column_id"]=="Monthly Year-To-Date production KPI":
        wf_data=production_kpis.loc[production_kpis["WF"]==selected_wf]
        monthly_ytd_productions=wf_data["Monthly YTD productions"].iloc[0]
        monthly_ytd_p50s=wf_data["Monthly YTD P50s"].iloc[0]
        monthly_ytd_p75s=wf_data["Monthly YTD P75s"].iloc[0]
        monthly_ytd_p25s=wf_data["Monthly YTD P25s"].iloc[0]
        monthly_ytd_p90s=wf_data["Monthly YTD P90s"].iloc[0]
        monthly_ytd_p10s=wf_data["Monthly YTD P10s"].iloc[0]
        monthly_ytd_productions=pd.DataFrame.from_dict(monthly_ytd_productions, orient='index',columns=["Production"])
        monthly_ytd_p50s=pd.DataFrame.from_dict(monthly_ytd_p50s, orient='index',columns=["P50"])
        monthly_ytd_p75s=pd.DataFrame.from_dict(monthly_ytd_p75s, orient='index',columns=["P75"])
        monthly_ytd_p25s=pd.DataFrame.from_dict(monthly_ytd_p25s, orient='index',columns=["P25"])
        monthly_ytd_p90s=pd.DataFrame.from_dict(monthly_ytd_p90s, orient='index',columns=["P90"])
        monthly_ytd_p10s=pd.DataFrame.from_dict(monthly_ytd_p10s, orient='index',columns=["P10"])
        data=pd.concat([monthly_ytd_productions,monthly_ytd_p50s,monthly_ytd_p75s,monthly_ytd_p25s,monthly_ytd_p90s,monthly_ytd_p10s], axis=1)
        data=data.dropna(subset=["Production"])




        production_kpis_figure = go.Figure()
        
        
        production_kpis_figure.add_trace(go.Scatter(
                name='Production',
                legendgroup = 'a',
                x=data.index,
                y=data["Production"],
                mode='lines+markers',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'rgb(31, 119, 180)'
                        },
                showlegend=True
            ))

        production_kpis_figure.add_trace(go.Scatter(
                name='P50',
                legendgroup = 'b',
                x=data.index,
                y=data["P50"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Green',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        production_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'c',
                x=data.index,
                y=data["P75"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        production_kpis_figure.add_trace(go.Scatter(
                name='P75/P25',
                legendgroup = 'c',
                x=data.index,
                y=data["P25"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Orange',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))

        production_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'd',
                x=data.index,
                y=data["P90"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=True
            ))

        production_kpis_figure.add_trace(go.Scatter(
                name='P90/P10',
                legendgroup = 'd',
                x=data.index,
                y=data["P10"],
                mode='lines',
                line={
                        'shape':"spline",
                        'smoothing':1,
                        'color':'Red',
                        'dash':"dashdot"
                        },
                showlegend=False
            ))

        production_kpis_figure.update_yaxes(title="Energy production (MWh)",showgrid=False)
#        production_kpis_figure.update_xaxes(showgrid=False)
        production_kpis_figure.update_xaxes(showgrid=False,range=[dt.today().replace(month=1,day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-12), dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)])
        production_kpis_figure.update_layout(title=dict(text=str(selected_wf)+' - Historical YTD productions against budget',x=0.5))
        figure_appearance={'display': 'block'}




        
    else:
        production_kpis_figure=default_figure

    
    
    return production_kpis_figure,figure_appearance
    
    
@app.callback(
        [Output('production-kpis-graph-2', 'figure'),
         Output("production-kpis-graph-2-container","style")
         ],[Input('production-kpis-datatable', 'active_cell'),
         Input('production-kpis-graph','clickData')])
def production_kpis_graph_2(active_cell,clickData):    

    selected_wf=current_production_kpis["Wind farm"].iloc[active_cell["row_id"]]
    
    if clickData is not None:
        selected_time=clickData['points'][0]['x']
        selected_time=dt.strptime(selected_time,'%Y-%m-%d')
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)

    graph_month=selected_time.strftime("%B")
    graph_year=selected_time.year
    figure_appearance={'display': 'none'}

    if active_cell["column_id"]=="Monthly production KPI":
        
        data=pd.DataFrame(index=wf_names)
        data["Wind farm"]=data.index
        data["P50"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly P50s"].iloc[0])
        data["P50"]=data["P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Production"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly productions"].iloc[0])
        data["Production"]=data["Production"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Wind Indexes"].iloc[0])
        data["WI"]=data["WI"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI_P50"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Monthly P50s"].iloc[0])
        data["WI_P50"]=data["WI_P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Potential production with normal wind resource"]=data["Production"]/data["WI"]*data["WI_P50"]
        data["Gap to wind resource"]=100*(data["Production"]-data["Potential production with normal wind resource"])/data["Potential production with normal wind resource"]
        data["Gap to losses"]=100*(data["Potential production with normal wind resource"]-data["P50"])/data["P50"]
        data["Final gap to budget"]=100*(data["Production"]-data["P50"])/data["P50"]
        data=data.loc[data["Wind farm"]==selected_wf]
        data=data[["Gap to wind resource","Gap to losses","Final gap to budget"]]
        data=pd.DataFrame.transpose(data)

        production_kpis_figure_2 = go.Figure()

        production_kpis_figure_2.add_trace(go.Bar(
                x=data.index,
                y=data[selected_wf],
                showlegend=False
            ))
        
        production_kpis_figure_2.update_yaxes(title="Gap (%)",showgrid=False)
        production_kpis_figure_2.update_xaxes(showgrid=False)

        production_kpis_figure_2.update_layout(title=dict(text=str(selected_wf)+" - "+str(graph_month)+" "+str(graph_year)+' - Gap to production budget',x=0.5))
        figure_appearance={'display': 'block'}

    elif active_cell["column_id"]=="Monthly Year-To-Date production KPI":
        
        data=pd.DataFrame(index=wf_names)
        data["Wind farm"]=data.index
        data["P50"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly YTD P50s"].iloc[0])
        data["P50"]=data["P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Production"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly YTD productions"].iloc[0])
        data["Production"]=data["Production"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Monthly YTD Wind Indexes"].iloc[0])
        data["WI"]=data["WI"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI_P50"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Monthly YTD P50s"].iloc[0])
        data["WI_P50"]=data["WI_P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Potential production with normal wind resource"]=data["Production"]/data["WI"]*data["WI_P50"]
        data["Gap to wind resource"]=100*(data["Production"]-data["Potential production with normal wind resource"])/data["Potential production with normal wind resource"]
        data["Gap to losses"]=100*(data["Potential production with normal wind resource"]-data["P50"])/data["P50"]
        data["Final gap to budget"]=100*(data["Production"]-data["P50"])/data["P50"]
        data=data.loc[data["Wind farm"]==selected_wf]
        data=data[["Gap to wind resource","Gap to losses","Final gap to budget"]]
        data=pd.DataFrame.transpose(data)

#        kpis["Asset Manager"]=kpis["WF"].apply(lambda x: metadata.at[x,"AM"])

        production_kpis_figure_2 = go.Figure()

        production_kpis_figure_2.add_trace(go.Bar(
                x=data.index,
                y=data[selected_wf],
                showlegend=False
            ))
        
        production_kpis_figure_2.update_yaxes(title="Gap (%)",showgrid=False)
        production_kpis_figure_2.update_xaxes(showgrid=False)

        production_kpis_figure_2.update_layout(title=dict(text=str(selected_wf)+" - "+str(graph_month)+" "+str(graph_year)+' - YTD gap to production budget',x=0.5))
        figure_appearance={'display': 'block'}

        
    else:
        production_kpis_figure_2=default_figure

    
    
    return production_kpis_figure_2,figure_appearance











@app.callback(
        [Output('production-kpis-graph-3', 'figure'),
         Output("production-kpis-graph-3-container","style")
         ],[Input('production-kpis-datatable', 'active_cell'),
         Input('production-kpis-graph','clickData')])
def production_kpis_graph_3(active_cell,clickData):    
   
    if clickData is not None:
        selected_time=clickData['points'][0]['x']
        selected_time=dt.strptime(selected_time,'%Y-%m-%d')
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)

    graph_month=selected_time.strftime("%B")
    graph_year=selected_time.year
    figure_appearance={'display': 'none'}

    if active_cell["column_id"]=="Monthly production KPI":
        
        data=pd.DataFrame(index=wf_names)
        data["Wind farm"]=data.index
        data["Country"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Country"].iloc[0])
        data["P50"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly P50s"].iloc[0])
        data["P50"]=data["P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Production"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly productions"].iloc[0])
        data["Production"]=data["Production"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Wind Indexes"].iloc[0])
        data["WI"]=data["WI"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI_P50"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Monthly P50s"].iloc[0])
        data["WI_P50"]=data["WI_P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Potential production with normal wind resource"]=data["Production"]/data["WI"]*data["WI_P50"]
        data=data[["Country","Potential production with normal wind resource","Production","P50"]]
        data["Valid"]=data.isnull().sum(axis=1)
        data["Valid"]=data["Valid"].apply(lambda x: 1 if x==0 else 0)
        data["Count"]=1
        data["Potential production with normal wind resource"]=data.apply(lambda row: row["Potential production with normal wind resource"] if row["Valid"]==1 else np.nan,axis=1)
        data["Production"]=data.apply(lambda row: row["Production"] if row["Valid"]==1 else np.nan,axis=1)
        data["P50"]=data.apply(lambda row: row["P50"] if row["Valid"]==1 else np.nan,axis=1)
        data=data.groupby(["Country"]).sum()
        data.sort_index(inplace=True)
        data["Country"]=data.index
        data["Gap to wind resource"]=100*(data["Production"]-data["Potential production with normal wind resource"])/data["Potential production with normal wind resource"]
        data["Gap to losses"]=100*(data["Potential production with normal wind resource"]-data["P50"])/data["P50"]
        data["Final gap to budget"]=100*(data["Production"]-data["P50"])/data["P50"]
        data["Indexes"]=data.apply(lambda row: str(row["Country"])+" ("+str(row["Valid"])+"/"+str(row["Count"])+" WFs)" if row["Valid"]<row["Count"] else str(row["Country"]),axis=1)
        data.index=data["Indexes"]

        production_kpis_figure_3 = go.Figure()
        production_kpis_figure_3.add_trace(go.Bar(
                name="Gap to wind resource",
                x=data.index,
                y=data["Gap to wind resource"],
            ))
        production_kpis_figure_3.add_trace(go.Bar(
                name="Gap to losses",
                x=data.index,
                y=data["Gap to losses"],
            ))
        production_kpis_figure_3.add_trace(go.Bar(
                name="Final gap to budget",
                x=data.index,
                y=data["Final gap to budget"],
            ))
        
        production_kpis_figure_3.update_yaxes(title="Gap (%)",showgrid=False)
        production_kpis_figure_3.update_xaxes(showgrid=False)
        production_kpis_figure_3.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - Gap to production budget by country',x=0.5))
        figure_appearance={'display': 'block'}

    elif active_cell["column_id"]=="Monthly Year-To-Date production KPI":
        
        data=pd.DataFrame(index=wf_names)
        data["Wind farm"]=data.index
        data["Country"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Country"].iloc[0])
        data["P50"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly YTD P50s"].iloc[0])
        data["P50"]=data["P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Production"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly YTD productions"].iloc[0])
        data["Production"]=data["Production"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Monthly YTD Wind Indexes"].iloc[0])
        data["WI"]=data["WI"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI_P50"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Monthly YTD P50s"].iloc[0])
        data["WI_P50"]=data["WI_P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Potential production with normal wind resource"]=data["Production"]/data["WI"]*data["WI_P50"]
        data=data[["Country","Potential production with normal wind resource","Production","P50"]]
        data["Valid"]=data.isnull().sum(axis=1)
        data["Valid"]=data["Valid"].apply(lambda x: 1 if x==0 else 0)
        data["Count"]=1
        data["Potential production with normal wind resource"]=data.apply(lambda row: row["Potential production with normal wind resource"] if row["Valid"]==1 else np.nan,axis=1)
        data["Production"]=data.apply(lambda row: row["Production"] if row["Valid"]==1 else np.nan,axis=1)
        data["P50"]=data.apply(lambda row: row["P50"] if row["Valid"]==1 else np.nan,axis=1)
        data=data.groupby(["Country"]).sum()
        data.sort_index(inplace=True)
        data["Country"]=data.index
        data["Gap to wind resource"]=100*(data["Production"]-data["Potential production with normal wind resource"])/data["Potential production with normal wind resource"]
        data["Gap to losses"]=100*(data["Potential production with normal wind resource"]-data["P50"])/data["P50"]
        data["Final gap to budget"]=100*(data["Production"]-data["P50"])/data["P50"]
        data["Indexes"]=data.apply(lambda row: str(row["Country"])+" ("+str(row["Valid"])+"/"+str(row["Count"])+" WFs)" if row["Valid"]<row["Count"] else str(row["Country"]),axis=1)
        data.index=data["Indexes"]

        production_kpis_figure_3 = go.Figure()
        production_kpis_figure_3.add_trace(go.Bar(
                name="Gap to wind resource",
                x=data.index,
                y=data["Gap to wind resource"],
            ))
        production_kpis_figure_3.add_trace(go.Bar(
                name="Gap to losses",
                x=data.index,
                y=data["Gap to losses"],
            ))
        production_kpis_figure_3.add_trace(go.Bar(
                name="Final gap to budget",
                x=data.index,
                y=data["Final gap to budget"],
            ))
        
        production_kpis_figure_3.update_yaxes(title="Gap (%)",showgrid=False)
        production_kpis_figure_3.update_xaxes(showgrid=False)
        production_kpis_figure_3.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - YTD gap to production budget by country',x=0.5))
        figure_appearance={'display': 'block'}

        
    else:
        production_kpis_figure_3=default_figure

    
    
    return production_kpis_figure_3,figure_appearance



@app.callback(
        [Output('production-kpis-graph-4', 'figure'),
         Output("production-kpis-graph-4-container","style")
         ],[Input('production-kpis-datatable', 'active_cell'),
         Input('production-kpis-graph','clickData')])
def production_kpis_graph_4(active_cell,clickData):    

    if clickData is not None:
        selected_time=clickData['points'][0]['x']
        selected_time=dt.strptime(selected_time,'%Y-%m-%d')
    else:
        selected_time=dt.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)+relativedelta(months=-1)

    graph_month=selected_time.strftime("%B")
    graph_year=selected_time.year
    figure_appearance={'display': 'none'}

    if active_cell["column_id"]=="Monthly production KPI":
        
        data=pd.DataFrame(index=wf_names)
        data["Wind farm"]=data.index
        data["Portfolio"]=data["Wind farm"].apply(lambda x: metadata.at[x,"Portfolio"])
        data["P50"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly P50s"].iloc[0])
        data["P50"]=data["P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Production"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly productions"].iloc[0])
        data["Production"]=data["Production"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Wind Indexes"].iloc[0])
        data["WI"]=data["WI"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI_P50"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Monthly P50s"].iloc[0])
        data["WI_P50"]=data["WI_P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Potential production with normal wind resource"]=data["Production"]/data["WI"]*data["WI_P50"]
        data=data[["Portfolio","Potential production with normal wind resource","Production","P50"]]
        data["Valid"]=data.isnull().sum(axis=1)
        data["Valid"]=data["Valid"].apply(lambda x: 1 if x==0 else 0)
        data["Count"]=1
        data["Potential production with normal wind resource"]=data.apply(lambda row: row["Potential production with normal wind resource"] if row["Valid"]==1 else np.nan,axis=1)
        data["Production"]=data.apply(lambda row: row["Production"] if row["Valid"]==1 else np.nan,axis=1)
        data["P50"]=data.apply(lambda row: row["P50"] if row["Valid"]==1 else np.nan,axis=1)
        data=data.groupby(["Portfolio"]).sum()
        data.sort_index(inplace=True)
        data["Portfolio"]=data.index
        data["Gap to wind resource"]=100*(data["Production"]-data["Potential production with normal wind resource"])/data["Potential production with normal wind resource"]
        data["Gap to losses"]=100*(data["Potential production with normal wind resource"]-data["P50"])/data["P50"]
        data["Final gap to budget"]=100*(data["Production"]-data["P50"])/data["P50"]
        data["Indexes"]=data.apply(lambda row: str(row["Portfolio"])+" ("+str(row["Valid"])+"/"+str(row["Count"])+" WFs)" if row["Valid"]<row["Count"] else str(row["Portfolio"]),axis=1)
        data.index=data["Indexes"]

        production_kpis_figure_4 = go.Figure()
        production_kpis_figure_4.add_trace(go.Bar(
                name="Gap to wind resource",
                x=data.index,
                y=data["Gap to wind resource"],
            ))
        production_kpis_figure_4.add_trace(go.Bar(
                name="Gap to losses",
                x=data.index,
                y=data["Gap to losses"],
            ))
        production_kpis_figure_4.add_trace(go.Bar(
                name="Final gap to budget",
                x=data.index,
                y=data["Final gap to budget"],
            ))
        
        production_kpis_figure_4.update_yaxes(title="Gap (%)",showgrid=False)
        production_kpis_figure_4.update_xaxes(showgrid=False)
        production_kpis_figure_4.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - Gap to production budget by portfolio',x=0.5))
        figure_appearance={'display': 'block'}

    elif active_cell["column_id"]=="Monthly Year-To-Date production KPI":
        
        data=pd.DataFrame(index=wf_names)
        data["Wind farm"]=data.index
        data["Portfolio"]=data["Wind farm"].apply(lambda x: metadata.at[x,"Portfolio"])
        data["P50"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly YTD P50s"].iloc[0])
        data["P50"]=data["P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Production"]=data["Wind farm"].apply(lambda x: production_kpis.loc[production_kpis["WF"]==x]["Monthly YTD productions"].iloc[0])
        data["Production"]=data["Production"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Monthly YTD Wind Indexes"].iloc[0])
        data["WI"]=data["WI"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["WI_P50"]=data["Wind farm"].apply(lambda x: resource_kpis.loc[resource_kpis["Wind farm"]==x]["Monthly YTD P50s"].iloc[0])
        data["WI_P50"]=data["WI_P50"].apply(lambda x: x[selected_time] if selected_time in x.keys() else np.nan)
        data["Potential production with normal wind resource"]=data["Production"]/data["WI"]*data["WI_P50"]
        data=data[["Portfolio","Potential production with normal wind resource","Production","P50"]]
        data["Valid"]=data.isnull().sum(axis=1)
        data["Valid"]=data["Valid"].apply(lambda x: 1 if x==0 else 0)
        data["Count"]=1
        data["Potential production with normal wind resource"]=data.apply(lambda row: row["Potential production with normal wind resource"] if row["Valid"]==1 else np.nan,axis=1)
        data["Production"]=data.apply(lambda row: row["Production"] if row["Valid"]==1 else np.nan,axis=1)
        data["P50"]=data.apply(lambda row: row["P50"] if row["Valid"]==1 else np.nan,axis=1)
        data=data.groupby(["Portfolio"]).sum()
        data.sort_index(inplace=True)
        data["Portfolio"]=data.index
        data["Gap to wind resource"]=100*(data["Production"]-data["Potential production with normal wind resource"])/data["Potential production with normal wind resource"]
        data["Gap to losses"]=100*(data["Potential production with normal wind resource"]-data["P50"])/data["P50"]
        data["Final gap to budget"]=100*(data["Production"]-data["P50"])/data["P50"]
        data["Indexes"]=data.apply(lambda row: str(row["Portfolio"])+" ("+str(row["Valid"])+"/"+str(row["Count"])+" WFs)" if row["Valid"]<row["Count"] else str(row["Portfolio"]),axis=1)
        data.index=data["Indexes"]

        production_kpis_figure_4 = go.Figure()
        production_kpis_figure_4.add_trace(go.Bar(
                name="Gap to wind resource",
                x=data.index,
                y=data["Gap to wind resource"],
            ))
        production_kpis_figure_4.add_trace(go.Bar(
                name="Gap to losses",
                x=data.index,
                y=data["Gap to losses"],
            ))
        production_kpis_figure_4.add_trace(go.Bar(
                name="Final gap to budget",
                x=data.index,
                y=data["Final gap to budget"],
            ))
        
        production_kpis_figure_4.update_yaxes(title="Gap (%)",showgrid=False)
        production_kpis_figure_4.update_xaxes(showgrid=False)
        production_kpis_figure_4.update_layout(title=dict(text=str(graph_month)+" "+str(graph_year)+' - YTD gap to production budget by portfolio',x=0.5))
        figure_appearance={'display': 'block'}

        
    else:
        production_kpis_figure_4=default_figure

    
    
    return production_kpis_figure_4,figure_appearance






        
if __name__ == '__main__':
#    app.run_server(host='0.0.0.0',debug=True,dev_tools_ui=False)
    app.run_server(debug=True,dev_tools_ui=False)