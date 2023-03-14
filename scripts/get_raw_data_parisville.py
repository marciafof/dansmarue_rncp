# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:16:42 2023
Script used to get raw databases from VilledeParis API
@author: marci
"""
import os
import pymysql.connections
import pymysql.cursors
from shapely.geometry import shape,Polygon, Point
import json
import contextlib
import requests
import time
from sqlalchemy import create_engine 
import pandas as pd

os.chdir("C:\\Users\\marci\\dev\\FinalProject_IronHack\\dansmarue")

#%%
@contextlib.contextmanager
def connect():
    pw = "getpassword" #getpass.getpass()
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
        
def parse_geojson(geoj):
    row_geojson = geoj["properties"].copy()
    #row_geojson["geometry"] = json.dumps(geoj["geometry"]) 
    #geometry = json.loads(row_geojson["geometry"])
    geometry = geoj["geometry"]
    shape_geom = shape(geometry)
    wkt_geo = shape_geom.wkt
    wkt_geo_centroid = shape_geom.centroid.wkt
    row_geojson["lat"] = row_geojson["geom_x_y"]["lat"]
    row_geojson["lon"] = row_geojson["geom_x_y"]["lon"]
    row_geojson["geometrywkt"] = wkt_geo
    return row_geojson
#%%  ARRONDISSEMENTS
def create_code_postal_long(code_short):
    paris_base_code = "75000"
    n = len(str(code_short))
    replacementStr = str(code_short)
    return replacementStr.join(paris_base_code.rsplit(paris_base_code[-n:], 1))

#%%
#BECAUSE OF CHANGES IN THE FUSION OF PARIS CENTRE WE NEED TO CREATE A NEW ARRONDISSEMENTS GEOGRAPHIC MAP
from shapely.ops import unary_union
import shapely.wkt
from shapely import to_geojson
dataset_id = "arrondissements"
format_ = "geojson"
query_url = f"https://opendata.paris.fr/api/v2/catalog/datasets/{dataset_id}/records"
query_url = f"https://opendata.paris.fr/api/v2/catalog/datasets/{dataset_id}/exports/{format_}"
drop_table = True
r = requests.get(query_url)
print(f"Number of rows in json downloaded {len(r.json())}")


with connect() as cnx:
    cur = cnx.cursor()
    if drop_table:
        cur.execute("""DROP TABLE IF EXISTS `code_postal_new`""")
    cur.execute("""CREATE TABLE IF NOT EXISTS `code_postal_new` 
                ( `code_postal_id` varchar(10),
                  `code_postal_long` varchar(20),
                  `numero_insee` int,
                  `name` varchar(128),
                  `lon` float,
                  `lat` float,
                  `geometry` json,
                  PRIMARY KEY (`code_postal_id`)
                 )""")
    cur.execute("DELETE FROM `code_postal_new`")    
    
    arrondisement_to_group = []
    geom_to_join = []
    for row in r.json()["features"]:        
        start_time = time.time()
        row_geojson = parse_geojson(row)
        row_geojson["code_postal_long"] = create_code_postal_long(row_geojson["c_ar"])
        
        if row_geojson["c_ar"] in [1,2,3,4]:
            arrondisement_to_group.append(row_geojson)
            geom_to_join.append(shapely.wkt.loads(row_geojson['geometrywkt']))
            row_geojson["c_ar"] = str(row_geojson["c_ar"])+"_OLD"
        query_to_sql = """INSERT INTO `code_postal_new` 
                      ( 
                        `code_postal_id`,
                        `code_postal_long`,
                        `numero_insee`,
                        `name`,
                        `lon`,
                         `lat`,
                        `geometry`
                     ) 
                    VALUES ( %s,
                             %s,
                             %s,
                             %s,
                             %s,
                             %s,
                             ST_AsGeoJSON(ST_GeomFromText(%s))
                             )"""
        
        print(query_to_sql)
        cur.execute( query_to_sql, (
                                    row_geojson["c_ar"],
                                    row_geojson["code_postal_long"],
                                    row_geojson["c_arinsee"],
                                    row_geojson["l_aroff"],
                                    row_geojson["lat"],
                                    row_geojson["lon"],
                                    row_geojson["geometrywkt"]))
        cnx.commit()
        end_time =  time.time()
        print(f"Time in seconds to send chunk: {end_time-start_time}")
    #Creating entity for Paris CENTRE
    start_time = time.time()
    paris_centre_geom = unary_union(geom_to_join)
    #TO GEOJSON
    new_map = r.json().copy()
    paris_centre_area = 0
    for row in new_map["features"]:
        if row["properties"]["c_ar"] in [1,2,3,4]:
            row["properties"]["c_ar"] = str(row["properties"]["c_ar"])+"_OLD"
            paris_centre_area+=row["properties"]["surface"]
    new_dic = dict()
    new_dic["type"] = "Feature"
    new_dic["geometry"] = shapely.geometry.mapping(paris_centre_geom)
    new_dic["properties"] = {'n_sq_ar': None,
                             'c_ar': 1,
                             'c_arinsee': None,
                             'l_ar': 'Paris Centre',
                             'l_aroff': 'Paris Centre',
                             'n_sq_co': None,
                             'surface': paris_centre_area,
                             'perimetre': None,
                             'geom_x_y': {'lon': paris_centre_geom.centroid.x, 'lat':paris_centre_geom.centroid.y}}
    new_map["features"].append(new_dic)
    
    with open("data/arrondissement_modify.geojson", "w") as f:
        json.dump(new_map, f)
    
    row_to_sql = {}
    row_to_sql["c_ar"] = 1
    row_to_sql["code_postal_long"] = "PARIS_CENTRE"
    row_to_sql["c_arinsee"] = None
    row_to_sql["l_aroff"] = "PARIS CENTRE"
    row_to_sql["lat"] = paris_centre_geom.centroid.y
    row_to_sql["lon"] = paris_centre_geom.centroid.x
    row_to_sql["geometrywkt"] = paris_centre_geom.wkt
    query_to_sql = """INSERT INTO `code_postal_new` 
                  ( 
                    `code_postal_id`,
                    `code_postal_long`,
                    `numero_insee`,
                    `name`,
                    `lon`,
                     `lat`,
                    `geometry`
                 ) 
                VALUES ( %s,
                         %s,
                         %s,
                         %s,
                         %s,
                         %s,
                         ST_AsGeoJSON(ST_GeomFromText(%s))
                         )"""
    
    print(query_to_sql)
    cur.execute( query_to_sql, (
                                row_to_sql["c_ar"],
                                row_to_sql["code_postal_long"],
                                row_to_sql["c_arinsee"],
                                row_to_sql["l_aroff"],
                                row_to_sql["lat"],
                                row_to_sql["lon"],
                                row_to_sql["geometrywkt"]))
    cnx.commit()
    end_time =  time.time()
    print(f"Time in seconds to send chunk: {end_time-start_time}")
#%%   
    
dataset_id = "arrondissements"
format_ = "geojson"
query_url = f"https://opendata.paris.fr/api/v2/catalog/datasets/{dataset_id}/records"
query_url = f"https://opendata.paris.fr/api/v2/catalog/datasets/{dataset_id}/exports/{format_}"
drop_table = True
r = requests.get(query_url)
print(f"Number of rows in json downloaded {len(r.json())}")


with connect() as cnx:
    cur = cnx.cursor()
    if drop_table:
        cur.execute("""DROP TABLE IF EXISTS `code_postal`""")
    cur.execute("""CREATE TABLE IF NOT EXISTS `code_postal` 
                ( `code_postal_id` varchar(10),
                  `code_postal_long` varchar(10),
                  `numero_insee` int,
                  `name` varchar(128),
                  `lon` float,
                  `lat` float,
                  `geometry` json,
                  PRIMARY KEY (`code_postal_id`)
                 )""")
    cur.execute("DELETE FROM `code_postal`")    
    for row in r.json()["features"]:        
        start_time = time.time()
        row_geojson = parse_geojson(row)
        row_geojson["code_postal_long"] = create_code_postal_long(row_geojson["c_ar"])
        query_to_sql = """INSERT INTO `code_postal` 
                      ( 
                        `code_postal_id`,
                        `code_postal_long`,
                        `numero_insee`,
                        `name`,
                        `lon`,
                         `lat`,
                        `geometry`
                     ) 
                    VALUES ( %s,
                             %s,
                             %s,
                             %s,
                             %s,
                             %s,
                             ST_AsGeoJSON(ST_GeomFromText(%s))
                             )"""
        
        print(query_to_sql)
        cur.execute( query_to_sql, (
                                    row_geojson["c_ar"],
                                    row_geojson["code_postal_long"],
                                    row_geojson["c_arinsee"],
                                    row_geojson["l_aroff"],
                                    row_geojson["lat"],
                                    row_geojson["lon"],
                                    row_geojson["geometrywkt"]))
        cnx.commit()
        end_time =  time.time()
        print(f"Time in seconds to send chunk: {end_time-start_time}")
        
        
        
#%%            
dataset_id = "conseils-quartiers"
format_ = "geojson"
query_url = f"https://opendata.paris.fr/api/v2/catalog/datasets/{dataset_id}/records"
query_url = f"https://opendata.paris.fr/api/v2/catalog/datasets/{dataset_id}/exports/{format_}"
drop_table = True
r = requests.get(query_url)
print(f"Number of rows in json downloaded {len(r.json())}")
 
   
with connect() as cnx:
    cur = cnx.cursor()
    if drop_table: #`geometry` geometry,
        cur.execute("""DROP TABLE IF EXISTS `quartier`""")
    cur.execute("""CREATE TABLE IF NOT EXISTS `quartier` 
                ( `quartier_id` varchar(10), 
                  `code_postal_id` varchar(10),
                  `name` varchar(128),
                  `lon` float,
                  `lat` float,
                  `geometry` json,
                  
                  PRIMARY KEY (quartier_id)
                 )""")
    cur.execute("DELETE FROM `quartier`")    
    for row in r.json()["features"]:        
        start_time = time.time()
        row_geojson = parse_geojson(row)
        query_to_sql = """INSERT INTO `quartier` 
                      ( `quartier_id`, 
                        `code_postal_id`,
                        `name`,
                        `lon`,
                         `lat`,
                        `geometry`
                     ) 
                    VALUES ( %s,
                             %s,
                             %s,
                             %s,
                             %s,
                             ST_AsGeoJSON(ST_GeomFromText(%s))
                             )"""
        
        print(query_to_sql)
        cur.execute( query_to_sql, (
                                    row_geojson["no_consqrt"],
                                    row_geojson["nar"],
                                    row_geojson["nom_quart"],
                                    row_geojson["lat"],
                                    row_geojson["lon"],
                                    row_geojson["geometrywkt"]))
        cnx.commit()
        end_time =  time.time()
        print(f"Time in seconds to send chunk: {end_time-start_time}")
        
#%%CREATE ETAT TABLE
with connect() as cnx:
    cur = cnx.cursor()
    if drop_table: #`geometry` geometry,
        cur.execute("""DROP TABLE IF EXISTS `etat`""")
        cur.execute(
                    """CREATE TABLE `etat` (
                      `etat_id` int unsigned NOT NULL AUTO_INCREMENT,
                      `description` varchar(256) DEFAULT NULL,
                      PRIMARY KEY (`etat_id`)
                    ) ;""")
        cur.execute("""
        INSERT INTO `etat` ( 
                          `description` 
                         ) 
                        VALUES ( "Service fait"),
                       ( "Rejeté"),
        			   ( "Nouveau"),
        			   ( "Transféré à un tiers"),
                       ("Service programmé"),
                       ("Sous surveillance"),
                       ("A traiter"),
                       ("Service programmé tiers"),
                       ("A requalifier"),
                       ("A faire terrain"),
                       ("A faire bureau"),
                       ("Echec d'envoi par WS");"""
                       )
    cnx.commit()

#%%Check OSM DATA
import shapely.wkt

def create_connection_sql(schema="dansmarue"):
    connection_string = 'mysql+pymysql://root:' + "getpassword" + f'@127.0.0.1:3306/{schema}'
    engine = create_engine(connection_string)
    return engine

conn = create_connection_sql(schema="dansmarue")
df_quartier=pd.read_sql('SELECT * FROM quartier', conn)
df_quartier.sort_values(by="quartier_id",inplace=True)
df_quartier["geom"] = df_quartier["geometry"].apply(lambda x: Polygon(shape(json.loads(x))))

#Which names of the tables
list_of_tables = ["shop", "amenity"]
for key_element in list_of_tables:
    df_elem=pd.read_sql(f"""SELECT * FROM {key_element}""", conn)
    df_elem.set_index("id_",inplace=True)
    df_elem_copy = df_elem.copy()
    df_elem_copy.dropna(subset=["lat","lon"],axis=0,inplace=True)
    df_elem_copy["geom"] = df_elem_copy["geometry"].apply(lambda x: shapely.wkt.loads(x))
    df_elem["quartier_id"] = None
    df_elem["code_postal_id"] = None
    
    print("For efficiency we select points around each neighbourhood (quartier)")
    start_time = time.time()
    for iquartier, rowquartier in df_quartier.iterrows():
        quartier_polygon = rowquartier["geom"]
        bbox_quartier = quartier_polygon.bounds
        print("First selection from bbox_quartier to optimize treatment")
        df_elem_inquartier = df_elem_copy[(df_elem_copy["lon"]>=bbox_quartier[0])&(df_elem_copy["lon"]<=bbox_quartier[2])]
        df_elem_inquartier = df_elem_inquartier[(df_elem_inquartier["lat"]>=bbox_quartier[1])&(df_elem_inquartier["lat"]<=bbox_quartier[3])]
        df_elem_inquartier["in_quartier"] = df_elem_inquartier["geom"].apply(lambda x:quartier_polygon.contains(x))
        #getting ids for elements in  quartier
        ids_in_quartier = df_elem_inquartier[df_elem_inquartier["in_quartier"]].index.to_list()
        #Assigning corresponding quartier and code postal
        df_elem.loc[ids_in_quartier,"quartier_id"] = rowquartier["quartier_id"]
        df_elem.loc[ids_in_quartier,"code_postal_id"] = rowquartier["code_postal_id"]
        #dropping matched elements for a faster analaysis in next loop
        df_elem_copy.drop(index=ids_in_quartier,inplace=True)
    end_time =  time.time()
    print(f"Time in seconds to assign Ids: {end_time-start_time}")
    print(f"""Number of not assigned points = {df_elem["quartier_id"].isna().sum()}""")
    df_elem = df_elem[~df_elem["quartier_id"].isna()]
    df_elem.reset_index(inplace=True)
    table_name = f"{key_element}_georef"
    df_elem.to_sql(table_name,conn,index=False)