# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:16:11 2023
Obtain OpenStreetMap Elements using the OverpassAPI wrapper : OSMPythonTools
@author: marci
"""
import os
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import overpassQueryBuilder, Overpass
import pymysql.connections
import pymysql.cursors
import contextlib
import numpy as np
import pandas as pd
#LIBRARIES FOR GIS MANIPULATION
from shapely.geometry import shape
from shapely.geometry import Point, MultiPoint
import time
#%%
@contextlib.contextmanager
def connect():
    pw = "getyourpassword" #getpass.getpass()
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
 
#%%

nominatim = Nominatim()
#Get AreaId for the city of Paris
areaId = nominatim.query('Paris, France').areaId() 
#Initialize connection
overpass = Overpass()
#%%Parse elements


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
def parse_result_osm(result_element, key_element = "shop", nan_value = np.nan):
    tag_info = result_element.tags()
    type_element = key_element
    if key_element in tag_info.keys():
        if len(tag_info[key_element].split(";"))>1:
            sub_element = tag_info[key_element].split(";")[0]
        else:
            sub_element = tag_info[key_element]
    else:
        sub_element = nan_value
    if "name" in tag_info.keys() :
        name_element = tag_info["name"]

        name_element = name_element.replace('’',"'")
    else:
        name_element = nan_value
    id_ = result_element.id()
    lat = result_element.lat()
    lon = result_element.lon()
    try: 
        if result_element.countMembers()>2:
            list_centroids= []
            for result_member in result_element.members():
                try:
                    geom = result_member.geometry()
                    shape_geom = shape(geom)
                    #print(shape_geom)
                    list_centroids.append(shape_geom.centroid)
                except:
                    print("Cant get geometry for this point we passe to another one")
                    
            multiP = MultiPoint(list_centroids)
            lat = multiP.centroid.y
            lon = multiP.centroid.x
            wkt_geo = multiP.wkt
        else:
            try:
                geom = result_element.geometry()
                shape_geom = shape(geom)
                if geom["type"] != "Point":
                    #If not POINT then we usally cannot get lat lon
                    lat = shape_geom.centroid.y
                    lon = shape_geom.centroid.x
                    wkt_geo = shape_geom.centroid.wkt
                else:
                    wkt_geo = shape_geom.wkt
            except:
                print('Could not retrieve geometry for element')
                wkt_geo = nan_value
    except:
        try:
            geom = result_element.geometry()
            shape_geom = shape(geom)
            if geom["type"] != "Point":
                #If not POINT then we usally cannot get lat lon
                lat = shape_geom.centroid.y
                lon = shape_geom.centroid.x
                wkt_geo = shape_geom.centroid.wkt
            else:
                wkt_geo = shape_geom.wkt
        except:
            print('Could not retrieve geometry for element')
            wkt_geo = nan_value
    return [id_, lat, lon, type_element, sub_element, name_element, wkt_geo]
    
#%% Build queries and save result in json

def overpass_to_csv(result,key_element, path_save):
    #Parsing results
    list_df_elements =[]
    for result_element in result.elements():
        list_df_elements.append( pd.DataFrame(parse_result_osm(result_element, key_element=key_element)).T)
    #list_df_elements = [ parse_result_osm(result_element) for result_element in result.elements()]
    df_key_elements = pd.concat(list_df_elements,axis=0)
    df_key_elements.columns = ["id_","lat","lon","category", "subcategory","name","geometry"]

    df_key_elements.to_csv(fr"{path_save}/{key_element}_features.csv",index=False)
    df_key_elements_nogeom = df_key_elements.drop(columns="geometry")
    df_key_elements_nogeom.to_csv(fr"{path_save}/{key_element}_features_nogeo.csv",index=False)

def overpass_to_sql(result):
    #Parsing results
    list_df_elements =[]
    for result_element in result.elements():
        row_results = parse_result_osm(result_element,key_element=key_element,nan_value=None)
        Dic_results= {}
        for i, key in enumerate(["id_","lat","lon","category", "subcategory","name","geometry"]):
            Dic_results[key] = row_results[i]
        list_df_elements.append(Dic_results)
    return list_df_elements


#%%

os.chdir("C:\\Users\\marci\\dev\\FinalProject_IronHack\\dansmarue")

path_save = "data/osm"
to_sql = True
to_csv = True

#Which features to obtain from the OverpassAPI based on MapElements
# https://wiki.openstreetmap.org/wiki/Map_features
list_of_elements = ["shop", "amenity"]
drop_table = True
for key_element in list_of_elements:
    query = overpassQueryBuilder(area=areaId, elementType=['way', 'relation',"node"],selector=f'{key_element}', includeGeometry=True)
    result = overpass.query(query)
    result.countElements()
    if to_csv:
        overpass_to_csv(result,key_element, path_save)
    if to_sql:
        all_records = overpass_to_sql(result)
        
        with connect() as cnx:
            cur = cnx.cursor()
            if drop_table:
                cur.execute(f"""DROP TABLE IF EXISTS `{key_element}`""")
            cur.execute(f"""CREATE TABLE IF NOT EXISTS `{key_element}` 
                        ( `id_` varchar(128), 
                          `lat` float,
                          `lon` float,
                          `category` varchar(50),
                          `subcategory` varchar(128),
                          `name` varchar(128),
                          `geometry` varchar(2048) 
                         )""")
            cur.execute(f"""DELETE FROM `{key_element}`""")
            print("Sending to sql")
            #print(len(records_format))
            ichunk = 1
            start_time = time.time()
            for records in chunks(all_records, 100):
                cur.executemany(f"""INSERT INTO `{key_element}` ( `id_`, 
                  `lat`,
                  `lon`,
                  `category`,
                  `subcategory`,
                  `name`,
                  `geometry` 
                 ) 
                VALUES ( %(id_)s,
                         %(lat)s,
                         %(lon)s,
                         %(category)s,
                         %(subcategory)s,
                         %(name)s,
                         %(geometry)s)""", records)
            cnx.commit()
            end_time =  time.time()
            print(f"Time in seconds to send chunk: {end_time-start_time}")
            print(f"{ichunk}")
            ichunk += 1

#%% EXAMPLE USING REQUEST
# overpass_url = "http://overpass-api.de/api/interpreter"
# overpass_query = """
# [out:json];
# area["ISO3166-1"="DE"][admin_level=2];
# (node["amenity"="biergarten"](area);
#  way["amenity"="biergarten"](area);
#  rel["amenity"="biergarten"](area);
# );
# out center;
# """
# response = requests.get(overpass_url, 
#                         params={'data': overpass_query})
# data = response.json()