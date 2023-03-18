# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:27:46 2023

@author: marci
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine 
import pymysql.connections
import pymysql.cursors
import time
import contextlib
import matplotlib.pyplot as plt
os.chdir("C:\\Users\\marci\\dev\\FinalProject_IronHack\\dansmarue")
data_path = "data/dmr_historique"

path_output= "model_prep"


@contextlib.contextmanager
def connect():
    pw = "azerty" #getpass.getpass()
    #'mysql+pymysql://root:' + pw + '@127.0.0.1:3306/'
    user="root"
    host = "127.0.0.1"
    cnx = pymysql.connections.Connection(host=host, user=user, password=pw)
    cnx.select_db("dansmarue")
    yield cnx
    cnx.commit()
    cnx.close()

 
# Yield successive n-sized
# chunks from l.
def chunks(it, n):
    l = []
    for item in it:
        l.append(item)
        if len(l) == n:
            yield l
            l = []
    if len(l) > 0:
        yield l

def create_connection_sql(schema="dansmarue"):
    connection_string = 'mysql+pymysql://root:' + "azerty" + f'@127.0.0.1:3306/{schema}'
    engine = create_engine(connection_string)
    return engine
def haversine_distance(lon1, lat1, lon2, lat2,**kwargs):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    All args must be of equal length.
    
    Sourced from https://stackoverflow.com/a/29546836/11637704
    
    Thanks to derricw!
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
from sklearn.cluster import DBSCAN

def dbscan_cluster(latitudes,longitudes,epsilon,min_samples,**kwargs):
    '''
    Function to perform DBSCAN clustering for given parameters.
    
    '''
    # convert epsilon from km to radians
    kms_per_radian = 6371.0088
    epsilon /= kms_per_radian
    
    # set up the algorithm
    dbscan = DBSCAN(
        eps = epsilon,
        min_samples = min_samples,
        algorithm = 'ball_tree',
        metric = 'haversine',
        **kwargs
    )
    
    # fit the algorithm
    dbscan.fit(
        np.radians(
            [x for x in zip(latitudes,longitudes)]
        )
    )
    
    # return the cluster labels
    return pd.Series(dbscan.labels_)
#%%CONNECT TO SQL
year_analysis = 2022
schema = "dansmarue"
table_name = f"dmr_{year_analysis}_clean"
conn = create_connection_sql(schema="dansmarue")
#%%CONNECT TO SQL
read_from_schema = True

year_analysis = 2022
if read_from_schema:
    schema = "dansmarue"
    table_name = "dmr_all" #f"dmr_{year_analysis}_clean"
    conn = create_connection_sql(schema="dansmarue")
    #LOAD SAMPLE OF DATA TO 
    #table_name = "dmr_all"
    print(f"""Reading {table_name}""")
    query_str = f"""SELECT * FROM {table_name} WHERE  EXTRACT(YEAR FROM date_input) >2020"""
    df=pd.read_sql(query_str, conn)
else:
    df = pd.read_csv("../model_prep/dmr_2022_simplify.csv")
#WE CLEAN REJECTED
df =df[df["etat"]!="Rejeté"]
df.etat.value_counts()

#DROP NOT USEFUL COLUMNS
list_to_drop=['code_postal','numero', 'prefixe', 'intervenant', 'extrainfo',"id_dmr"]
df.drop(columns = list_to_drop,inplace=True)
#df["date_input"] = pd.to_datetime(df["date_input"],format="%Y-%m-%d")
#%% ONLY COUNT
import pickle
def window_input(window_length:int, data:pd.DataFrame,target_column:str, feature_column:str)->pd.DataFrame:
    df = data.copy()
    i = 1
    while i < window_length:
        df[f"{feature_column}_{i}"] = df[target_column].shift(-i)
        i +=1
    if i==window_length:
        df[f"{feature_column}_0"] = df[target_column].shift(-i)
    df = df.dropna(axis=0)
    return df

per_category = False

df["date_input"] = pd.to_datetime(df["date_input"],format="%Y-%m-%d")

categorical_model = True

if categorical_model:
    from sklearn.preprocessing import OrdinalEncoder

    ord_enc = OrdinalEncoder()
    df["category_EN_code"] = ord_enc.fit_transform(df[["category_EN"]])
    with open(f"{path_output}/encode_category_cat.pkl", "wb") as f:
        pickle.dump(ord_enc, f)
    # df["category_EN"] = df["category_EN"].astype('category')
    # df["category_EN_code"] = df["category_EN"].cat.codes
    target_col = "category_EN_code"
    df.drop(columns="category_EN", inplace=True)
else:
    if per_category:
        categories_model = ['Autos, motos, vélos...', 'Objets abandonnés',
                       'Graffitis, tags, affiches et autocollants']
        dic_cat_df = {}
        
        for cate_i in categories_model:
            df_cat = df[df["category_FR"]==cate_i]
            df_ts = df_cat.groupby(df_cat.date_input.dt.date).count().iloc[:,0]
            
            df_ts_p_quartier = df.groupby([df.date_input.dt.date, "quartier_id"]).count().iloc[:,0]
            df_ts_p_quartier.name = "target"
            df_ts_p_quartier = df_ts_p_quartier.reset_index()
            df_ts_p_quartier["quartier_id"] = df_ts_p_quartier["quartier_id"].astype(int)
    else:
        df_ts = df.groupby(df.date_input.dt.date).count().iloc[:,0]
        
        df_ts_p_quartier = df.groupby([df.date_input.dt.date, "quartier_id"]).count().iloc[:,0]
        df_ts_p_quartier.name = "target"
        df_ts_p_quartier = df_ts_p_quartier.reset_index()
        #df_ts_p_quartier["quartier_id"] = df_ts_p_quartier["quartier_id"].astype(int)
        
        df_ts_p_quartier = window_input(7, df_ts_p_quartier,"target",feature_column="y")

    

#%% OPEN FEATURES DATASET
one_hot_encode= False

features = pd.read_csv(f"{path_output}/feature_set_quartier_2.csv", dtype = {'quartier_id': str} )#,
                       #sep=";",decimal=",", encoding="utf-8-sig")
traffic_data = pd.read_csv(f"{path_output}/traffic_pquartier_2021_2022.csv", dtype = {'quartier_id': str}   )
traffic_data["date"] = pd.to_datetime(traffic_data["date"],format="%Y-%m-%d")

features.drop(columns="name",inplace=True)
features["surface_km2"] = features["surface_km2"].astype(float)
#Standarize every column in per km2
for icol in ['number_shops', 'number_amenities',
       'n_places']:
    features[icol] = features[icol].astype(float)
    features[f"{icol}_p_km2"] = features[icol]/features["surface_km2"]
features['green_space_surf_km2'] = features['green_space_surf_m2']/1000000
    
features_to_pass_model = ['quartier_id', 'surface_km2', 
       'surface_parking_m2', 
       'green_space_surf_m2', 'zti', 'density_pop_2021', 'menages_km2',
       'pop_2021', 'number_shops_p_km2', 'number_amenities_p_km2',
       'n_places_p_km2',"Loyers de référence"]

features_df = features.loc[:,features_to_pass_model]

if not categorical_model:

    #JOINT TO COUNT PER QUARTIER
    count_p_quartier_feat = pd.merge(left = df_ts_p_quartier, right = features_df, 
                                     how = "left",
                                     left_on="quartier_id",
                                     right_on="quartier_id")
    
    #JOIN_TRAFFIC
    count_p_quartier_feat["date_input"] = pd.to_datetime(count_p_quartier_feat["date_input"],format="%Y-%m-%d")
    
    count_p_quartier_final = pd.merge(left = count_p_quartier_feat, right = traffic_data, 
                                     how = "left",
                                     left_on=["date_input","quartier_id"],
                                     right_on=["date","quartier_id"])
    
    
    #ADDITIONALE FEATURES
    count_p_quartier_final["day_of_week"] =  count_p_quartier_final["date_input"].dt.weekday
    count_p_quartier_final["month"] =  count_p_quartier_final["date_input"].dt.month
    
    if one_hot_encode :
        col_to_encode="quartier_id"
        count_p_quartier_final = pd.concat([count_p_quartier_final,
                                            pd.get_dummies(count_p_quartier_final[[col_to_encode]].astype(str), drop_first=True)],axis=1)
    #count_p_quartier_final.drop(columns=col_to_encode,inplace=True)
    count_p_quartier_final.isnull().sum()
    count_p_quartier_final["green_space_surf_m2"].fillna(0,inplace=True)
    count_p_quartier_final.dropna(inplace=True)
    count_p_quartier_final.to_csv(f"{path_output}/model_dataset_7.csv",index=False)
    #ANALYSE FEATURES
    df_corr = count_p_quartier_final.iloc[:,1:].corr()
    sns.heatmap(df_corr, annot=True)

else:
    
    #JOINT QUARTIER CHARACTERISTICS TO OG_DATA
    #list_to_drop = 
    #FIRST TRY WE USE ALL POSSIBLE COLUMNS
    cluster_df_trainset = df [[ 'lon', 'lat',
    'quartier_id',  'deltadays_signal_etat', 'category_EN_code' ]]
    #ADDITIONALE FEATURES
    day_of_week_columns = pd.get_dummies(df["date_input"].dt.weekday,drop_first=True)
    cols_names = [f"dayofweek_{x}" for x in day_of_week_columns.columns]
    day_of_week_columns.columns =cols_names
    cluster_df_trainset = cluster_df_trainset.merge(day_of_week_columns, left_index=True, right_index=True)

    m_of_y_columns = pd.get_dummies(df["date_input"].dt.month,drop_first=True)
    cols_names = [f"month_{x}" for x in m_of_y_columns.columns]
    m_of_y_columns.columns =cols_names
    cluster_df_trainset = cluster_df_trainset.merge(m_of_y_columns, left_index=True, right_index=True)
    
    
    
    cluster_df_trainset = pd.merge(left = cluster_df_trainset, right = features_df, 
                                     how = "left",
                                     left_on="quartier_id",
                                     right_on="quartier_id")
    cluster_df_trainset["green_space_surf_m2"].fillna(0,inplace=True)
    cluster_df_trainset.isnull().sum()
    cluster_df_trainset.drop(columns="quartier_id",inplace=True)
    
    cluster_df_trainset.dropna(inplace=True)
    cluster_df_trainset.to_csv(f"{path_output}/model_dataset_8_class.csv",index=False)


1642528