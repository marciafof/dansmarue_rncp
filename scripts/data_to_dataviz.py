# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 23:47:25 2023
Preparation of data for DataViz
@author: marci
"""

import os
#import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine 
import pymysql.connections
import pymysql.cursors
import time
import contextlib
#%%
os.chdir("C:\\Users\\marci\\dev\\FinalProject_IronHack\\dansmarue")
data_path = "data/dmr_historique"

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

#%%
path_output= "output_for_tableau"

#%% TIME SERIES OF YEAR
schema = "dansmarue"
table_name = "dmr_all" #f"dmr_{year_analysis}_clean"
conn = create_connection_sql(schema="dansmarue")
#LOAD SAMPLE OF DATA TO 
#table_name = "dmr_all"
print(f"""Reading {table_name}""")
query_str = f"""SELECT * FROM {table_name}"""
df=pd.read_sql(query_str, conn)
df.columns
cols_simplified = ['lon', 'lat',
'date_input', 'quartier_id','category_EN','category_FR','code_postal_id','code_postal']
df["date_input"] = pd.to_datetime(df["date_input"],format="%Y-%m-%d")
df_to_tableau = df.loc[:,cols_simplified]
df.groupby(df.date_input.dt.date).count().iloc[:,0].plot()
df_to_tableau.to_csv(f"{path_output}/dmr_all_simplified.csv")

#%% TAKING A SAMPLE OF ONE DAY
schema = "dansmarue"
table_name = "dmr_all" #f"dmr_{year_analysis}_clean"
conn = create_connection_sql(schema="dansmarue")
#LOAD SAMPLE OF DATA TO 
#table_name = "dmr_all"
print(f"""Reading {table_name}""")
query_str = f"""SELECT * FROM {table_name} WHERE  EXTRACT(YEAR FROM date_input) >2020"""
df=pd.read_sql(query_str, conn)

#%%READING POPULATION AND JOIN CODE POSTAL AND QUARTIER

schema = "dansmarue"
conn = create_connection_sql(schema="dansmarue")
df_code_postal = pd.read_sql('SELECT * FROM code_postal_new', conn)
df_quartier=pd.read_sql('SELECT * FROM quartier', conn)
#get population data
df_pop = pd.read_csv("data/insee/population_insee_2019_2021.csv",sep=";", decimal=".")
df_pop.dropna(axis=0,inplace=True)
df_pop["menages_p_km2"] = df_pop['Nombre de ménages 2019']/df_pop["Superficie en 2019, en km²"]

#GroupBy PARIS CENTRE --> 1+2+3+4
df_pop.set_index("code_postal_id",inplace=True)
df_pop_centre = df_pop.loc[["1_OLD","2_OLD","3_OLD","4_OLD"],["Population en 2019","Population en 2021",
                                                           'Superficie en 2019, en km²','Nombre de ménages 2019']].sum()
s_pop_centre =pd.Series(data =["1", None, None, "PARIS CENTRE",df_pop_centre["Population en 2019"],
                  df_pop_centre["Population en 2021"],None,None,
                  df_pop_centre["Superficie en 2019, en km²"],
                  df_pop_centre["Nombre de ménages 2019"]],
          index=['code_postal_id',"code_postal", 'numero_INSEE', 'Name INSEE', 'Population en 2019',
                                                 'Population en 2021',
                                                 "Densité de la population (nombre d'habitants au km²) en 2019",
                                                "Densité de la population (nombre d'habitants au km²) en 2021",
                                                 'Superficie en 2019, en km²','Nombre de ménages 2019'])

s_pop_centre["Densité de la population (nombre d'habitants au km²) en 2019"] = s_pop_centre["Population en 2019"]/s_pop_centre["Superficie en 2019, en km²"]
s_pop_centre["Densité de la population (nombre d'habitants au km²) en 2021"] = s_pop_centre["Population en 2021"]/s_pop_centre["Superficie en 2019, en km²"]
s_pop_centre["menages_p_km2"] = s_pop_centre['Nombre de ménages 2019']/s_pop_centre["Superficie en 2019, en km²"]

df_pop.reset_index(inplace=True)
df_pop = pd.concat([df_pop,s_pop_centre.to_frame().T ],axis=0)

df_pop_arrond = df_code_postal.loc[:,['code_postal_id', 'code_postal_long', 'numero_insee']]
df_pop_arrond = pd.merge(right=df_pop_arrond, left=df_quartier[['quartier_id',"code_postal_id","name"]],
                         left_on="code_postal_id", right_on = "code_postal_id",how="left" )

df_pop['code_postal_id'] = df_pop['code_postal_id'].astype("str")

df_pop = df_pop[['code_postal_id',"Population en 2021",
                                                           "Densité de la population (nombre d'habitants au km²) en 2021",
                                                           'Superficie en 2019, en km²', 'Nombre de ménages 2019']]
df_merge= pd.merge(left=df_pop_arrond, right =df_pop, how="left",
                         left_on="code_postal_id", right_on = "code_postal_id")
df_merge.columns =['quartier_id', 'code_postal_id', 'name', 'code_postal_long',
       'numero_insee', 'pop_2021',
       'density_pop_2021',
       'superficie_km2',"menages_km2"]
df_merge.to_csv(f"{path_output}/pop_density_parrond.csv",index=False)

#%%GET QUARTIER ARRONDISSEMENTS TABLE AND JOIN WITH CODE_POSTAL
year_analysis = 2022
schema = "dansmarue"
table_name = f"dmr_{year_analysis}_clean"
conn = create_connection_sql(schema="dansmarue")
#%% CREATE AGGREGATED TABLE FOR HISTORICAL EVOLUTION
df_pop = pd.read_csv(f"{path_output}/pop_density_parrond.csv")

years_analysis = [2016,2017,2018,2019,2020,2021,2022]

df_historique = pd.DataFrame()
for year_analysis in years_analysis:
    table_name = f"dmr_{year_analysis}_clean"
    print(f"""Reading {table_name}""")
    df_i=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)
    df_i["date_input"] = pd.to_datetime(df_i["date_input"],format="%Y-%m-%d")


    df_type_year_quartier = df_i.groupby([df_i.date_input.dt.year,
                                          df_i.date_input.dt.month,
                                          'code_postal_id',"quartier_id","category_FR"]).count().iloc[:,0]
    df_type_year_quartier.name = "count"
    df_type_year_quartier = df_type_year_quartier.reset_index(['code_postal_id',"quartier_id","category_FR"])
    df_type_year_quartier = df_type_year_quartier.reset_index(names=["year","month"])
    df_historique = pd.concat([df_historique,df_type_year_quartier],axis=0)
df_historique.to_csv(f"{path_output}/count_ptype_pquartier_pyear.csv",index=False)

def divide_supf_codepostal(x, df_pop):
    supf = np.mean(df_pop[df_pop["code_postal_id"]==  int(x["code_postal_id"])].loc[:,"superficie_km2"])
    print(supf)
    # print(df_pop[df_pop["code_postal_id"]==  int(x["code_postal_id"])].loc[:,"superficie_km2"])
    return x["count"]/supf
df_historique = pd.DataFrame()
for year_analysis in years_analysis:
    table_name = f"dmr_{year_analysis}_clean"
    print(f"""Reading {table_name}""")
    df_i=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)
    df_i["date_input"] = pd.to_datetime(df_i["date_input"],format="%Y-%m-%d")
    df_type_year_quartier = df_i.groupby([df_i.date_input.dt.year,
                                          df_i.date_input.dt.month,
                                          'code_postal_id',"category_FR"]).count().iloc[:,0]
    df_type_year_quartier.name = "count"
    df_type_year_quartier = df_type_year_quartier.reset_index(["code_postal_id","category_FR"])
    
    df_type_year_quartier["count_by_km2_arron"] = df_type_year_quartier.apply(divide_supf_codepostal, axis=1, args=( df_pop,))
    df_type_year_quartier = df_type_year_quartier.reset_index(names=["year","month"])
    df_historique = pd.concat([df_historique,df_type_year_quartier],axis=0)
    #break
df_historique.to_csv(f"{path_output}/count_ptype_pyear_km2.csv",index=False)
#%% COUNT PER YEAR PER SURFACE OF ARRONDISSEMENT
df_historique = pd.DataFrame()
for year_analysis in years_analysis:
    table_name = f"dmr_{year_analysis}_clean"
    print(f"""Reading {table_name}""")
    df_i=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)
    df_i["date_input"] = pd.to_datetime(df_i["date_input"],format="%Y-%m-%d")
    df_type_year_quartier = df_i.groupby([df_i.date_input.dt.year,
                                          'code_postal_id']).count().iloc[:,0]
    df_type_year_quartier.name = "count"
    df_type_year_quartier = df_type_year_quartier.reset_index(['code_postal_id'])
    
    df_type_year_quartier["countpy_km2_arr"] = df_type_year_quartier.apply(divide_supf_codepostal, axis=1, args=( df_pop,))
    df_type_year_quartier = df_type_year_quartier.reset_index(names=["year"])
    df_historique = pd.concat([df_historique,df_type_year_quartier],axis=0)

df_historique.to_csv(f"{path_output}/count_pyear_km2_parr.csv",index=False)

#%% GET MAX PER YEAR PER MONTH PER DAY
def divide_supf_codepostal(x, df_pop):
    supf = np.mean(df_pop[df_pop["code_postal_id"]==  int(x["code_postal_id"])].loc[:,"superficie_km2"])
    return x["count"]/supf

df_historique = pd.DataFrame()
for year_analysis in years_analysis:
    table_name = f"dmr_{year_analysis}_clean"
    print(f"""Reading {table_name}""")
    df_i=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)
    df_i["date_input"] = pd.to_datetime(df_i["date_input"],format="%Y-%m-%d")
    df_type_year_quartier = df_i.groupby([df_i.date_input.dt.year,
                                          'code_postal_id',"category_FR"]).count().iloc[:,0]
    df_type_year_quartier.name = "count"
    df_type_year_quartier = df_type_year_quartier.reset_index(["code_postal_id","category_FR"])
    
    df_type_year_quartier["count_by_km2_arron"] = df_type_year_quartier.apply(divide_supf_codepostal, axis=1, args=( df_pop,))
    df_type_year_quartier = df_type_year_quartier.reset_index(names=["year"])
    
    list_gdfs = []
    for categor_,gdf in df_type_year_quartier.groupby(["code_postal_id"]):
        cat_max = gdf.sort_values(by="count",ascending=False).iloc[0,:]["category_FR"]
        gdf["cat_max_pyear" ] = cat_max
        list_gdfs.append(gdf)
    df_historique = pd.concat(list_gdfs,axis=0)
df_historique.to_csv(f"{path_output}/maxcat_pyear_arrond.csv",index=False)

df_historique = pd.DataFrame()
for year_analysis in years_analysis:
    table_name = f"dmr_{year_analysis}_clean"
    print(f"""Reading {table_name}""")
    df_i=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)
    df_i["date_input"] = pd.to_datetime(df_i["date_input"],format="%Y-%m-%d")
    df_type_year_quartier = df_i.groupby([df_i.date_input.dt.year,
                                          'code_postal_id',"quartier_id","category_FR"]).count().iloc[:,0]
    df_type_year_quartier.name = "count"
    df_type_year_quartier = df_type_year_quartier.reset_index(["code_postal_id","quartier_id","category_FR"])
    
    df_type_year_quartier["count_by_km2_arron"] = df_type_year_quartier.apply(divide_supf_codepostal, axis=1, args=( df_pop,))
    df_type_year_quartier = df_type_year_quartier.reset_index(names=["year"])
    
    list_gdfs = []
    for categor_,gdf in df_type_year_quartier.groupby(["code_postal_id","quartier_id"]):
        print(gdf)
        cat_max = gdf.sort_values(by="count",ascending=False).iloc[0,:]["category_FR"]
        gdf["cat_max_pyear" ] = cat_max
        list_gdfs.append(gdf)
    df_historique = pd.concat(list_gdfs,axis=0)
df_historique.to_csv(f"{path_output}/maxcat_pyear_quartier.csv",index=False)
#%% CHECK DAY OF WEEKEND
def weekday_or_weekend(datei):
    if datei.weekday() > 4:
        return 1
    else:
        return 0
years_analysis = [2016,2017,2018,2019,2020,2021,2022]

df_historique = pd.DataFrame()
for year_analysis in years_analysis:
    table_name = f"dmr_{year_analysis}_clean"
    print(f"""Reading {table_name}""")
    df_i=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)
    df_i["date_input"] = pd.to_datetime(df_i["date_input"],format="%Y-%m-%d")
    df_i["weekend"] = df_i["date_input"].apply(weekday_or_weekend)

    df_type_year_quartier = df_i.groupby([df_i.date_input.dt.year,
                                          df_i.date_input.dt.month,
                                          df_i.weekend,
                                          'code_postal_id',"category_FR","subcategory_FR"]).count().iloc[:,0]
    df_type_year_quartier.name = "count"
    df_type_year_quartier = df_type_year_quartier.reset_index(['code_postal_id',"category_FR","subcategory_FR"])
    df_type_year_quartier = df_type_year_quartier.reset_index(names=["year","month","weekend"])
    df_historique = pd.concat([df_historique,df_type_year_quartier],axis=0)
df_historique.to_csv(f"{path_output}/count_pcategory_parrond_pyear_wknd.csv",index=False)

def weekday_or_weekend(datei):
    if datei.weekday() > 4:
        return 1
    else:
        return 0
years_analysis = [2016,2017,2018,2019,2020,2021,2022]

df_historique = pd.DataFrame()
for year_analysis in years_analysis:
    table_name = f"dmr_{year_analysis}_clean"
    print(f"""Reading {table_name}""")
    df_i=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)
    df_i["date_input"] = pd.to_datetime(df_i["date_input"],format="%Y-%m-%d")
    df_i["dayofweek"] = df_i["date_input"].apply(lambda x: x.weekday())

    df_type_year_quartier = df_i.groupby([df_i.date_input.dt.year,
                                          df_i.date_input.dt.month,
                                          df_i.dayofweek,
                                          'code_postal_id',"category_FR","subcategory_FR"]).count().iloc[:,0]
    df_type_year_quartier.name = "count"
    df_type_year_quartier = df_type_year_quartier.reset_index(['code_postal_id',"category_FR","subcategory_FR"])
    df_type_year_quartier = df_type_year_quartier.reset_index(names=["year","month","dayofweek"])
    df_historique = pd.concat([df_historique,df_type_year_quartier],axis=0)
df_historique.to_csv(f"{path_output}/count_pcategory_parrond_pyear_dayofweek.csv",index=False)
#%% UNDERSTANDING SUBCATEGORIES BY ARRONDISSEMENT
years_analysis = [2016,2017,2018,2019,2020,2021,2022]

df_historique = pd.DataFrame()
for year_analysis in years_analysis:
    table_name = f"dmr_{year_analysis}_clean"
    print(f"""Reading {table_name}""")
    df_i=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)
    df_i["date_input"] = pd.to_datetime(df_i["date_input"],format="%Y-%m-%d")


    df_type_year_quartier = df_i.groupby([df_i.date_input.dt.year,
                                          df_i.date_input.dt.month,
                                          'code_postal_id',"category_FR","subcategory_FR"]).count().iloc[:,0]
    df_type_year_quartier.name = "count"
    df_type_year_quartier = df_type_year_quartier.reset_index(['code_postal_id',"category_FR","subcategory_FR"])
    df_type_year_quartier = df_type_year_quartier.reset_index(names=["year","month"])
    df_historique = pd.concat([df_historique,df_type_year_quartier],axis=0)
df_historique.to_csv(f"{path_output}/count_pcategory_parrond_pyear.csv",index=False)
#%% ETAT AND DATE ETAT ONLY FOR 2021 AND 2022
df_historique = pd.DataFrame()
years_analysis = [2021,2022]

for year_analysis in years_analysis:
    table_name = f"dmr_{year_analysis}_clean"
    print(f"""Reading {table_name}""")
    df_i=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)
    df_i["date_input"] = pd.to_datetime(df_i["date_input"],format="%Y-%m-%d")


    df_type_year_quartier = df_i.groupby([df_i.date_input.dt.year,
                                          df_i.date_input.dt.month,
                                          'code_postal_id',"quartier_id",
                                          "category_FR","etat"]).count().iloc[:,0]
    df_type_year_quartier.name = "count"
    df_type_year_quartier = df_type_year_quartier.reset_index(['code_postal_id',"quartier_id","category_FR","etat"])
    df_type_year_quartier = df_type_year_quartier.reset_index(names=["year","month"])
    df_historique = pd.concat([df_historique,df_type_year_quartier],axis=0)
df_historique.to_csv(f"{path_output}/count_pcategory_petat_pyear.csv",index=False)

#%% ETAT AND DATE ETAT ONLY FOR 2021 AND 2022
df_historique = pd.DataFrame()
years_analysis = [2021,2022]

for year_analysis in years_analysis:
    table_name = f"dmr_{year_analysis}_clean"
    print(f"""Reading {table_name}""")
    df_i=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)
    
    df_to_tableau = df_i.loc[:,['id_dmr','category_FR','code_postal_id',
                                'lon', 'lat',
                                'etat', 
                                 'date_input', 'date_etat',
                                'deltadays_signal_etat']]
        
    df_to_tableau.to_csv(f"{path_output}/dmr_{year_analysis}_simplify.csv",index=False)
#%%


#%%GET ANNUAL DATA
annual_signaled=pd.read_sql(f"""SELECT COUNT(`id_dmr`) as count_events FROM {table_name}""", conn)
#COUNT STATE OF SIGNAL AND PERCENTAGE OF TOTAL
start=time.time()
count_etat = pd.read_sql(f"""SELECT `etat`
                             , COUNT(1) AS total
                             , COUNT(1) / t.cnt * 100 AS `percentage`
                         FROM {table_name} 
                         CROSS
                         JOIN (SELECT COUNT(1) AS cnt FROM {table_name} ) t
                         GROUP BY `etat`, t.cnt
                         """, conn)
end=time.time()
print(f"{end-start}")     






#%%
df=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)

#%% ESTIMATING TIME BETWEEN SIGNALING AND CHANGE IN STATE
df["deltat_signal_etat"] = df["date_etat"] - df['date_input']
#In number of days
df["deltadays_signal_etat"] =(df['deltat_signal_etat'].astype('timedelta64[s]'))/86400
median_delta_t = np.median(df["deltadays_signal_etat"])
print(f"The median of the days between the signal and the event state change is {median_delta_t}")

df["date_input"] = pd.to_datetime(df["date_input"],format="%Y-%m-%d")
df["date_etat"] = pd.to_datetime(df["date_etat"],format="%Y-%m-%d")

#%%

import plotly.figure_factory as ff
import plotly.express as px

px.set_mapbox_access_token(open(".mapbox_token").read())
df = px.data.carshare()

fig = ff.create_hexbin_mapbox(
    data_frame=df, lat="centroid_lat", lon="centroid_lon",
    nx_hexagon=10, opacity=0.5, labels={"color": "Point Count"},
    min_count=1,
)
fig.show()
#%% GET OSM
conn = create_connection_sql(schema="dansmarue")

df_osm = pd.read_sql("""SELECT * FROM amenity_georef""", conn)
df_osm.to_csv(f"{path_output}/amenity_georef.csv",index=False)
