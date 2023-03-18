# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:50:08 2023
Preparation of the model plan
@author: marci
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
from sqlalchemy import create_engine 
import pymysql.connections
import pymysql.cursors
import time
import contextlib
import matplotlib.pyplot as plt

#%% FUNCTIONS
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
#%%
def parse_soustype(row, column_name="subcategory_FR"):
    sentence_to_parse = row[column_name]
    if isinstance(sentence_to_parse, str):
        sentence_to_parse = sentence_to_parse.replace("[]","")
        if len(sentence_to_parse.split(":")) > 1:
            main_category = sentence_to_parse.split(":")[0]
            extra_details = sentence_to_parse.split(":")[1]
            return main_category.strip(), extra_details.strip()
            
        else:
            main_category = sentence_to_parse.split(":")[0]
            extra_details = np.nan
            return main_category.strip(), extra_details
    else:
        return None, None
#%%CONNECT TO SQL
year_analysis = 2022
schema = "dansmarue"
table_name = f"dmr_{year_analysis}_clean"
conn = create_connection_sql(schema="dansmarue")
#%% LOAD SAMPLE OF DATA TO 
table_name = "dmr_all"
print(f"""Reading {table_name}""")
df=pd.read_sql(f"""SELECT * FROM {table_name}""", conn)
df["date_input"] = pd.to_datetime(df["date_input"],format="%Y-%m-%d")

pd.set_option('display.max_columns', df.columns.size)

#%% CLEAN SUBCATEGORY
# path_output= "model_prep"
# df["Year"] = df.date_input.dt.year
# df_subcat_count =pd.pivot_table(df, index= ["category_FR","subcategory_FR"], 
#                                 columns ="Year", values = "id_dmr",
#                                 aggfunc="count", margins=True,
#                                 margins_name="Total")
# for year_ in df["Year"].unique():
#     df_subcat_count[f"{year_}_perc_total"] = (df_subcat_count[year_] / df_subcat_count.groupby(level=0)[year_].transform(sum) * 100)

# #To_CSV to analyse text in further detail
# df_subcat_count.to_csv(f"{path_output}/count_by_subcat_pyear.csv")


# df_subcat_count = df.groupby([df.date_input.dt.year, "category_FR","subcategory_FR"]).count().iloc[:,0]
# df_subcat_count.name = "count"
# df_subcat_count = df_subcat_count.T.to_frame()
# df_subcat_count.sort_values(by="count",ascending=False,inplace=True)
# if table_name == "dmr_all":
#     df_subcat_count.reset_index("date_input",names="year",inplace=True)
    
# else:
#     df_subcat_count["perc_of_total"] = df_subcat_count["count"]/df_subcat_count["count"].sum()*100

# #To_CSV to analyse text in further detail
# df_subcat_count.to_csv

# df_subcat = df["subcategory_FR"].value_counts().sort_values(ascending=False)
# fig, ax = plt.subplots(1,1, figsize=(18,12),dpi=100)
# sns.countplot(data=df, x="subcategory_FR", hue="category_FR", ax =ax)
# plt.sca(ax)
# plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees

#%% JOIN SHOP AMENITIES TO quartier_id
features_per_quartier = pd.DataFrame()

#READING QUARTIER AND CODE POSTAL
schema = "dansmarue"
conn = create_connection_sql(schema="dansmarue")
df_code_postal = pd.read_sql('SELECT * FROM code_postal_new', conn)
df_quartier=pd.read_sql('SELECT * FROM quartier', conn)
features_per_quartier = df_quartier[['quartier_id', 'name',"surface"]]
features_per_quartier.rename(columns={"surface":"surface_km2"},inplace=True)

print(f"""Reading {table_name}""")
shop_df=pd.read_sql("""SELECT * FROM shop_georef""", conn)
amenity_df=pd.read_sql("""SELECT * FROM amenity_georef""", conn)

shop_pquartier = pd.pivot_table(shop_df, index="quartier_id", columns = "subcategory", aggfunc="count",
                                margins=True)
count_shop_pquartier =shop_pquartier.iloc[:-1,-1]
count_shop_pquartier.name = "number_shops"
count_shop_pquartier = count_shop_pquartier.T.to_frame().reset_index()

amenity_pquartier = pd.pivot_table(amenity_df, index="quartier_id", columns = "subcategory", aggfunc="count",
                                   margins=True)
count_amenity_pquartier =amenity_pquartier.iloc[:-1,-1]
count_amenity_pquartier.name = "number_amenities"
count_amenity_pquartier = count_amenity_pquartier.T.to_frame().reset_index()

features_per_quartier = pd.merge(left=features_per_quartier, right =count_shop_pquartier, how="left", left_on = "quartier_id" ,right_on="quartier_id")

features_per_quartier = pd.merge(left=features_per_quartier, right =count_amenity_pquartier, how="left", left_on = "quartier_id" ,right_on="quartier_id")

#%%PARKINGS

def feature_to_join(df_feature,col_name,feat_name):
    feature_to_join = pd.pivot_table(df_feature, index="quartier_id", columns = col_name, aggfunc="count",
                                       margins=True)
    count_pquartier =feature_to_join.iloc[:-1,-1]
    count_pquartier.name = feat_name
    return count_pquartier.T.to_frame().reset_index()
import json
from shapely.geometry import shape
from shapely.geometry import Point, MultiPoint
import time
df_parking = pd.read_csv(f"data/stationnement-voie-publique-emplacements.csv",sep=";",encoding="utf-8-sig" )
df_parking = df_parking[['Régime prioritaire', 'Régime particulier', 'Nombre places réelles','Surface calculée' ,
                         'geo_shape',"geo_point_2d"]]
df_parking["geom"] = df_parking["geo_shape"].apply(lambda x: shape(json.loads(x)))
df_parking[["lat","lon"]] = df_parking["geo_point_2d"].str.split(",", expand=True)
df_parking[["lat","lon"]] = df_parking[["lat","lon"]].astype(float)
df_parking["quartier_id"] = None
df_parking["code_postal_id"] = None

df_elem_copy= df_parking.copy()
print("For efficiency we select points around each neighbourhood (quartier)")
start_time = time.time()
for iquartier, rowquartier in df_quartier.iterrows():
    quartier_polygon = shape(json.loads(rowquartier["geometry"]))
    bbox_quartier = quartier_polygon.bounds
    print("First selection from bbox_quartier to optimize treatment")
    df_elem_inquartier = df_elem_copy[(df_elem_copy["lon"]>=bbox_quartier[0])&(df_elem_copy["lon"]<=bbox_quartier[2])]
    df_elem_inquartier = df_elem_inquartier[(df_elem_inquartier["lat"]>=bbox_quartier[1])&(df_elem_inquartier["lat"]<=bbox_quartier[3])]
    df_elem_inquartier["in_quartier"] = df_elem_inquartier["geom"].apply(lambda x:quartier_polygon.contains(x))
    #getting ids for elements in  quartier
    ids_in_quartier = df_elem_inquartier[df_elem_inquartier["in_quartier"]].index.to_list()
    #Assigning corresponding quartier and code postal
    df_parking.loc[ids_in_quartier,"quartier_id"] = rowquartier["quartier_id"]
    df_parking.loc[ids_in_quartier,"code_postal_id"] = rowquartier["code_postal_id"]
    #dropping matched elements for a faster analaysis in next loop
    df_elem_copy.drop(index=ids_in_quartier,inplace=True)
end_time =  time.time()
print(f"Time in seconds to assign Ids: {end_time-start_time}")
print(f"""Number of not assigned points = {df_parking["quartier_id"].isna().sum()}""")
df_parking = df_parking[~df_parking["quartier_id"].isna()]
df_parking.to_csv(f"{path_output}/parkings_p_quartier.csv",index=False)

list_dfs = []
for it, gdf in df_parking.groupby(["Régime prioritaire","quartier_id"]):
    print(gdf)
    total_parkings = gdf["Nombre places réelles"].sum()
    total_surface = gdf["Surface calculée"].sum()
    s = {}
    s["Régime prioritaire"] = it[0]
    s["quartier_id"] = it[1]
    s["n_places"] =total_parkings
    s["surface_parking_m2"] = total_surface

    list_dfs.append(pd.DataFrame(s,index=[0]))
df_feature = pd.concat(list_dfs, axis=0)
#df_feature["quartier_id"]  = df_feature["quartier_id"].astype(int)

df_feature = df_feature.groupby(["quartier_id"]).sum()
df_feature.reset_index(inplace=True)


#col_feature = feature_to_join(df_parking,"Régime prioritaire","number_parkings")
features_per_quartier = pd.merge(left=features_per_quartier, right =df_feature, 
                                 how="left", left_on = "quartier_id" ,right_on="quartier_id")

#%%GREEN ZONES
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from shapely.geometry import shape

def to_shapely(x):
    try:
        return shape(json.loads(x))
    except:
        None
        

df_parks = pd.read_csv(f"data/espaces_verts.csv",sep=";",encoding="utf-8-sig" )
df_parks = df_parks[['Identifiant espace vert',"Typologie d'espace vert",
                     'Surface calculée', 'Superficie totale réelle','Geo Shape']]

df_parks["geom"] = df_parks["Geo Shape"].apply(lambda x: to_shapely(x))
df_parks["lat"] = df_parks["geom"].apply(lambda x: x.centroid.y if x else None)
df_parks["lon"] = df_parks["geom"].apply(lambda x: x.centroid.x if x else None)
df_parks["quartier_id"] = None
df_parks["code_postal_id"] = None

df_elem_copy= df_parks.copy()
print("For efficiency we select points around each neighbourhood (quartier)")
start_time = time.time()
for iquartier, rowquartier in df_quartier.iterrows():
    quartier_polygon = shape(json.loads(rowquartier["geometry"]))
    bbox_quartier = quartier_polygon.bounds
    print("First selection from bbox_quartier to optimize treatment")
    df_elem_inquartier = df_elem_copy[(df_elem_copy["lon"]>=bbox_quartier[0])&(df_elem_copy["lon"]<=bbox_quartier[2])]
    df_elem_inquartier = df_elem_inquartier[(df_elem_inquartier["lat"]>=bbox_quartier[1])&(df_elem_inquartier["lat"]<=bbox_quartier[3])]
    df_elem_inquartier["in_quartier"] = df_elem_inquartier["geom"].apply(lambda x:quartier_polygon.intersects(x) if x else None)
    #getting ids for elements in  quartier
    ids_in_quartier = df_elem_inquartier[df_elem_inquartier["in_quartier"]].index.to_list()
    #Assigning corresponding quartier and code postal
    df_parks.loc[ids_in_quartier,"quartier_id"] = rowquartier["quartier_id"]
    df_parks.loc[ids_in_quartier,"code_postal_id"] = rowquartier["code_postal_id"]
    #dropping matched elements for a faster analaysis in next loop
    df_elem_copy.drop(index=ids_in_quartier,inplace=True)
print(f"Time in seconds to assign Ids: {end_time-start_time}")
print(f"""Number of not assigned points = {df_parking["quartier_id"].isna().sum()}""")
df_parks = df_parks[~df_parks["quartier_id"].isna()]
df_parks.reset_index(inplace=True)
df_parks.to_csv(f"{path_output}/green_areas_quartier.csv",index=False)

df_feature = df_parks.groupby(["quartier_id"]).sum().loc[:,"Superficie totale réelle"]
df_feature.name = "green_space_surf"
df_feature = df_feature.reset_index()
#df_feature["quartier_id"]  = df_feature["quartier_id"].astype(int)
df_feature["green_space_surf_m2"]  = df_feature["green_space_surf"]

#col_feature = feature_to_join(df_parking,"Régime prioritaire","number_parkings")
features_per_quartier = pd.merge(left=features_per_quartier, right =df_feature, 
                                 how="left", left_on = "quartier_id" ,right_on="quartier_id")

#%%ZONES OF TOURISM
df_zti = pd.read_csv(f"data/zones-touristiques-internationales.csv",sep=";",encoding="utf-8-sig" )

df_zti["geom"] = df_zti["Geo Shape"].apply(lambda x: to_shapely(x))
df_zti["lat"] = df_zti["geom"].apply(lambda x: x.centroid.y if x else None)
df_zti["lon"] = df_zti["geom"].apply(lambda x: x.centroid.x if x else None)
df_zti["quartier_id"] = None
df_zti["code_postal_id"] = None
df_elem_copy= df_zti.copy()
print("For efficiency we select points around each neighbourhood (quartier)")
start_time = time.time()
for iquartier, rowquartier in df_quartier.iterrows():
    quartier_polygon = shape(json.loads(rowquartier["geometry"]))
    bbox_quartier = quartier_polygon.bounds
    print("First selection from bbox_quartier to optimize treatment")
    df_elem_inquartier = df_elem_copy[(df_elem_copy["lon"]>=bbox_quartier[0])&(df_elem_copy["lon"]<=bbox_quartier[2])]
    df_elem_inquartier = df_elem_inquartier[(df_elem_inquartier["lat"]>=bbox_quartier[1])&(df_elem_inquartier["lat"]<=bbox_quartier[3])]
    df_elem_inquartier["in_quartier"] = df_elem_inquartier["geom"].apply(lambda x:quartier_polygon.intersects(x) if x else None)
    #getting ids for elements in  quartier
    ids_in_quartier = df_elem_inquartier[df_elem_inquartier["in_quartier"]].index.to_list()
    #Assigning corresponding quartier and code postal
    df_zti.loc[ids_in_quartier,"quartier_id"] = rowquartier["quartier_id"]
    df_zti.loc[ids_in_quartier,"code_postal_id"] = rowquartier["code_postal_id"]
    #dropping matched elements for a faster analaysis in next loop
    df_elem_copy.drop(index=ids_in_quartier,inplace=True)
print(f"Time in seconds to assign Ids: {end_time-start_time}")
print(f"""Number of not assigned points = {df_parking["quartier_id"].isna().sum()}""")
df_zti = df_zti[~df_zti["quartier_id"].isna()]
df_zti.to_csv(f"{path_output}/zti_quartier.csv",index=False)

features_per_quartier["zti"] = 0
for iquartier in df_zti["quartier_id"].unique():
    features_per_quartier[features_per_quartier["quartier_id"]==iquartier]["zti"] =1
#%%ZONES DE LOYER
def feature_to_join(df_feature,col_name,feat_name):
    feature_to_join = pd.pivot_table(df_feature, index="quartier_id", columns = col_name, aggfunc="count",
                                       margins=True)
    count_pquartier =feature_to_join.iloc[:-1,-1]
    count_pquartier.name = feat_name
    return count_pquartier.T.to_frame().reset_index()
def Random_Points_in_Bounds(polygon, number):   
    minx, miny, maxx, maxy = polygon.bounds
    x = np.random.uniform( minx, maxx, number )
    y = np.random.uniform( miny, maxy, number )
    return x, y
def check_poly_n_poly(row, polygcol, polyg2):
    polyg1 = row[polygcol]
    #Creating 10 POunts
    x,y = Random_Points_in_Bounds(polyg1,20)
    pdf = pd.DataFrame()
    pdf['points'] = list(zip(x,y))
    pdf['points'] = pdf['points'].apply(Point)
    for p_i in pdf['points']:
        if polyg2.contains(p_i):
            return True
    return False
import json
from shapely.geometry import shape
from shapely.geometry import Point, MultiPoint
import time

df_loyer = pd.read_csv(f"data/logement-encadrement-des-loyers.csv",sep=";",encoding="utf-8-sig" )
df_loyer = df_loyer[df_loyer["Année"]==2021]
df_loyer = df_loyer[['Nom du quartier', 'Nombre de pièces principales',
                         'Type de location', 'Loyers de référence',
                         'geo_shape',"geo_point_2d"]]
df_loyer["geom"] = df_loyer["geo_shape"].apply(lambda x: shape(json.loads(x)))
df_loyer[["lat","lon"]] = df_loyer["geo_point_2d"].str.split(",", expand=True)
df_loyer[["lat","lon"]] = df_loyer[["lat","lon"]].astype(float)
df_loyer["quartier_id"] = None
df_loyer["code_postal_id"] = None

df_elem_copy= df_loyer.copy()
print("For efficiency we select points around each neighbourhood (quartier)")
start_time = time.time()
for iquartier, rowquartier in df_quartier.iterrows():
    print(iquartier)
    quartier_polygon = shape(json.loads(rowquartier["geometry"]))
    bbox_quartier = quartier_polygon.bounds
    print("First selection from bbox_quartier to optimize treatment")
    df_elem_inquartier = df_elem_copy.copy()
    # df_elem_inquartier = df_elem_copy[(df_elem_copy["lon"]>=bbox_quartier[0])&(df_elem_copy["lon"]<=bbox_quartier[2])]
    # df_elem_inquartier = df_elem_inquartier[(df_elem_inquartier["lat"]>=bbox_quartier[1])&(df_elem_inquartier["lat"]<=bbox_quartier[3])]
    
    df_elem_inquartier["in_quartier"] = df_elem_inquartier.apply(check_poly_n_poly, args=("geom", quartier_polygon,),axis=1)
    #getting ids for elements in  quartier
    ids_in_quartier = df_elem_inquartier[df_elem_inquartier["in_quartier"]].index.to_list()
    print(ids_in_quartier)

    #Assigning corresponding quartier and code postal
    df_loyer.loc[ids_in_quartier,"quartier_id"] = rowquartier["quartier_id"]
    df_loyer.loc[ids_in_quartier,"code_postal_id"] = rowquartier["code_postal_id"]
    #dropping matched elements for a faster analaysis in next loop
    #df_elem_copy.drop(index=ids_in_quartier,inplace=True)
end_time =  time.time()
print(f"Time in seconds to assign Ids: {end_time-start_time}")
print(f"""Number of not assigned points = {df_loyer["quartier_id"].isna().sum()}""")
df_loyer = df_loyer[~df_loyer["quartier_id"].isna()]
df_loyer.to_csv(f"{path_output}/loyer_p_quartier.csv",index=False)

df_feature = df_loyer.groupby("quartier_id").mean().loc[:, ["Nombre de pièces principales",
                                                          "Loyers de référence"]]

df_feature = df_feature.reset_index()


#col_feature = feature_to_join(df_parking,"Régime prioritaire","number_parkings")
features_per_quartier = pd.merge(left=features_per_quartier, right =df_feature, 
                                 how="left", left_on = "quartier_id" ,right_on="quartier_id")
#%%
#%%TRAFFIC
df_trafic_res = pd.read_csv(f"data/referentiel-comptages-routiers.csv",sep=";",encoding="utf-8-sig" )
from shapely.geometry import LineString, Point, Polygon
import pyproj
from functools import partial
from shapely.ops import transform


def reproj(geometry, epsg_from,epsg_to):
    # Geometry transform function based on pyproj.transform
    project = partial(
        pyproj.transform,
        pyproj.Proj(epsg_from),
        pyproj.Proj(epsg_to))
    
    line2 = transform(project, geometry)
    #print(str(line2.length) + " meters")
    return line2
def line_in_polygon(row, geo_col, quartier_polygon):
    x = row[geo_col]
    polygon_ext = LineString(list(quartier_polygon.exterior.coords))

    intersections = quartier_polygon.intersection(x)
    if intersections.is_empty:
        first, last = x.boundary
        if quartier_polygon.contains(first):
            print("The segment completely lies within the polygon.")
            return True, reproj(x, 'EPSG:4326', "EPSG:32631").length
        else:
            print("The segment does not lies within the polygon.")
            return False, 0
    else:
        return True, reproj(intersections, 'EPSG:4326', "EPSG:32631").length
        print("Segment-polygon intersections are found.")

df_trafic_res["geom"] = df_trafic_res["geo_shape"].apply(lambda x: to_shapely(x))
df_trafic_res["lat"] = df_trafic_res["geom"].apply(lambda x: x.centroid.y if x else None)
df_trafic_res["lon"] = df_trafic_res["geom"].apply(lambda x: x.centroid.x if x else None)
df_trafic_res["quartier_id"] = None
df_trafic_res["length_m_voie"] = None

df_elem_copy= df_trafic_res.copy()
print("For efficiency we select points around each neighbourhood (quartier)")
start_time = time.time()
for iquartier, rowquartier in df_quartier.iterrows():
    quartier_polygon = shape(json.loads(rowquartier["geometry"]))
    bbox_quartier = quartier_polygon.bounds
    print("First selection from bbox_quartier to optimize treatment")
    df_elem_inquartier = df_elem_copy[(df_elem_copy["lon"]>=bbox_quartier[0])&(df_elem_copy["lon"]<=bbox_quartier[2])]
    df_elem_inquartier = df_elem_inquartier[(df_elem_inquartier["lat"]>=bbox_quartier[1])&(df_elem_inquartier["lat"]<=bbox_quartier[3])]
    df_elem_inquartier[["in_quartier","length_m_voie"]] = df_elem_inquartier.apply(line_in_polygon,axis=1,
                                                                                   result_type="expand",                                                                                   
                                                                                           args=("geom",quartier_polygon,))
    
    #getting ids for elements in  quartier
    ids_in_quartier = df_elem_inquartier[df_elem_inquartier["in_quartier"]].index.to_list()
    #How much we inside each quartier

    #Assigning corresponding quartier and code postal
    df_trafic_res.loc[ids_in_quartier,"quartier_id"] = rowquartier["quartier_id"]
    df_trafic_res.loc[ids_in_quartier,"length_m_voie"] =  df_elem_inquartier[df_elem_inquartier["in_quartier"]].loc[:,"length_m_voie"]
    #dropping matched elements for a faster analaysis in next loop
    #Not valid for traffic data becaus eof lines
    #df_elem_copy.drop(index=ids_in_quartier,inplace=True)
end_time = time.time()
print(f"Time in seconds to assign Ids: {end_time-start_time}")
print(f"""Number of not assigned points = {df_parking["quartier_id"].isna().sum()}""")
df_trafic_res = df_trafic_res[~df_trafic_res["quartier_id"].isna()] #Only keep INTRAMUROSZ

df_trafic_res['Date fin dispo data'] = pd.to_datetime(df_trafic_res['Date fin dispo data'],format="%Y-%m-%dT%H:%M",utc=True)
#Keep ONLY THE ONE WITH newest data available
df_trafic_res = df_trafic_res[df_trafic_res['Date fin dispo data'] >= "2021-01-01"]


list_drop_ids = []
#Check  duplicates on Identifien Arv
double_ids = df_trafic_res[df_trafic_res[["Identifiant arc","quartier_id"]].duplicated(keep=False)].sort_values(by="Identifiant arc")
#Keep ONLY THE ONE WITH newest data available
if double_ids.shape[0]> 0 :
    for repeated_id, doublerow in double_ids.groupby("Identifiant arc"):
        #Check if data is available in period of study
        doublerow['Date fin dispo data'] = pd.to_datetime(doublerow['Date fin dispo data'],format="%Y-%m-%dT%H:%M",utc=True)
        doublerow.sort_values(by='Date fin dispo data', inplace=True)
        if doublerow[doublerow['Date fin dispo data'] < "2023-01-01"].shape[0] > 0:
            drop_ids = doublerow[doublerow['Date fin dispo data'] <"2023-01-01"].index.values
            list_drop_ids.append(drop_ids)
        elif doublerow[doublerow['Date fin dispo data']< "2023-01-01"].shape[0]==0:
            print(doublerow)
            doublerow['Date debut dispo data'] = pd.to_datetime(doublerow['Date debut dispo data'],format="%Y-%m-%dT%H:%M")
            doublerow.sort_values(by='Date debut dispo data', inplace=True)
            drop_ids = doublerow.iloc[1:,:].index.values
            list_drop_ids.append(drop_ids)
        else:
            print("No recent data drop id")
    df_trafic_res.loc[list_drop_ids]       
        
    
df_trafic_res.to_csv(f"{path_output}/reseau_traffic_quartier.csv",index=False)

#COMPTAGE TRAFFIC
import glob
path_files  ="data/traffic"
year=2022


treat_traffic = False
if treat_traffic:
    list_df_traffic = []
    
    for year in [2021,2022]:
        all_files = glob.glob(f"{path_files}/opendata_txt_{year}/*.txt")
        
        #f_ = all_files[1]
        for f_ in all_files:
            df_file = pd.read_csv(f_,sep=";",decimal=".",encoding="utf-8",usecols=['iu_ac',"t_1h",'etat_trafic'])
            list_df_traffic.append(df_file)
        
    df_tri = pd.concat(list_df_traffic)
    df_tri["t_1h"] = pd.to_datetime(df_tri["t_1h"],format="%Y-%m-%d %H:%M")
    df_tri["date"] = df_tri.t_1h.dt.date
    trafic_sum = pd.pivot_table(data=df_tri, index=["date","iu_ac" ],columns = "etat_trafic",
            aggfunc="count")
    trafic_sum.columns = trafic_sum.columns.droplevel(0) #remove amount
    trafic_sum.columns.name = None               #remove categories
    trafic_sum = trafic_sum.reset_index()
    trafic_sum = trafic_sum.fillna(0)
    
    for col_ in range(5):
        trafic_sum.rename(columns = {col_: f"etat_trafic_{col_}"},inplace=True)
    # per_date_traffic = df_tri.groupby([df_tri.t_1h.dt.date, "iu_ac", "etat_trafic"]).count()
    # per_date_traffic.name="number_hours"
    #Add quartier 
    trafic_per_quartier = pd.merge(left= trafic_sum, right =df_trafic_res[["Identifiant arc", 
                                                     'length_m_voie',
                                                     "quartier_id"]],
                                                     how="left",
                                                     left_on ="iu_ac",right_on="Identifiant arc")
    trafic_per_quartier["length_m_voie"] = trafic_per_quartier["length_m_voie"].astype(float)
    
    trafic_per_quartier.drop(columns = ["Identifiant arc","iu_ac"],inplace=True)
    
    df_traffic_year = trafic_per_quartier.groupby(["date","quartier_id"]).sum()
    df_traffic_year = df_traffic_year.reset_index()
            
    df_traffic_year["date"] = pd.to_datetime(df_traffic_year["date"], format="%Y-%m-%d")
    df_traffic_year =df_traffic_year[df_traffic_year["date"]<"2023-01-01"]
    #Check duplicates
    df_traffic_year[df_traffic_year[["date","quartier_id"]].duplicated()]
    
    df_traffic_year.to_csv(f"{path_output}/traffic_pquartier_2021_2022.csv",index=False)
else:
    df_traffic_year = pd.read_csv(f"{path_output}/traffic_pquartier_2021_2022.csv")

#%%READING POPULATION
df_pop = pd.read_csv(f"{path_output}/pop_density_parrond.csv",dtype = {'quartier_id': str}   )
df_pop = df_pop[['quartier_id', 'pop_2021', 'density_pop_2021', 'superficie_km2',
       'menages_km2']]

features_per_quartier = pd.merge(left=features_per_quartier, right =df_pop[["quartier_id","density_pop_2021","menages_km2"]], 
                                 how="left", left_on = "quartier_id" ,right_on="quartier_id")
features_per_quartier["pop_2021"] = features_per_quartier["density_pop_2021"]*features_per_quartier["surface_km2"]

features_per_quartier.to_csv(f"{path_output}/feature_set_quartier_2.csv",index=False)

#%%




#ind_to_fill = features_per_quartier[features_per_quartier["Loyers de référence"].isnull()].index

# features_per_quartier.loc[ind_to_fill,"Loyers de référence"] = val_to_fill

# val_to_fill = [27.253125,
# 25.5625,
# 27.459375,
# 31.209375,
# 28.71875,
# 28.353125,
# 25.834375,
# 24.371875,
# 25.5625,
# 22.925,
# 22.925,
# 23.728125,
# 25.209375,
# 25.209375,
# 25.209375,
# 28.353125,
# 24.371875,
# 23.728125,
# 23.728125,
# 21.096875,
# 21.096875,
# 21.096875,
# 25.9
# ]
