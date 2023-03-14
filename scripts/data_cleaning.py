# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:28:38 2023

This script is made to explore the dataset of 2022

@author: marci
"""
#%matplotlib
import os
#import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine 
import shapely.wkt
from shapely.geometry import shape,Polygon, Point
import json
import time
#%%
os.chdir("C:\\Users\\marci\\dev\\FinalProject_IronHack\\dansmarue")
data_path = "data/dmr_historique"

#%% FUNCTIONS
def create_connection_sql(schema="dansmarue", password = "addpassword"):
    connection_string = 'mysql+pymysql://root:' + {password} + f'@127.0.0.1:3306/{schema}'
    engine = create_engine(connection_string)
    return engine
def convert_date_format(df_col):
    for format_str in ["%Y-%m-%d %H:%M:%S","%d/%m/%Y %H:%M:%S","%d/%m/%Y %H:%M"]: 
        try:
            return pd.to_datetime(df_col,format=format_str)
        except ValueError:
            pass
    return None

def parse_soustype(row, column_name="soustype"):
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

def translate_category(category_FR):
    category_FR = str(category_FR).strip()
    categories_eng = {"Activités commerciales et professionnelles":"commercial_activities",
                      'Autos, motos, vélos...':"vehicles",
                      'Graffitis, tags, affiches et autocollants' :"graffities", 
                      'Mobiliers urbains':"street_furniture",
                      'Objets abandonnés':"abandon_objects", 
                      'Propreté':"cleanliness", 
                      'Voirie et espace public':"public_space",
                      'Éclairage / Électricité': "public_lightning", 
                      'Arbres, végétaux et animaux' : "vegetation_animals", 'Eau':"water_service"}
    if category_FR in categories_eng.keys():
        return categories_eng[category_FR]
    else:
        category_FR

def load_data_dmr_historique(fname : str, sep=";"):
    """
    Load data from historical data obtained from Open Data Paris 
    containing the points from the application Dans Ma Rue Paris.
    Some files have different encoding so we try with utf8 and latin.

    Parameters
    ----------
    fname : str
        DESCRIPTION.

    Returns
    -------
    df : pandas.DataFrame
        DESCRIPTION.

    """
    try:
        df = pd.read_csv(fname,sep=sep, decimal=",")

    except UnicodeDecodeError:
        df = pd.read_csv(fname,sep=sep, encoding="latin",decimal=",")
    return df

def outliers_removal(data, columns = False):
    stats = data.describe().transpose()
    stats['IQR'] = stats['75%'] - stats['25%']
    if columns:
        outliers = pd.DataFrame(columns=columns)
        features = columns
    else:
        outliers = pd.DataFrame(columns=data.columns)
        features = data.columns
    for feature in features:
        # Identify 25th & 75th quartiles
        q25, q75 = stats.at[feature,'25%'] , stats.at[feature,'75%']
        #print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
        feat_iqr = q75 - q25
        #print('iqr: {}'.format(feat_iqr))
        
        feat_cut_off = feat_iqr * 1.5
        feat_lower, feat_upper = q25 - feat_cut_off, q75 + feat_cut_off
        #print('Cut Off: {}'.format(feat_cut_off))
        #print(feature_name +' Lower: {}'.format(feat_lower))
        #print(feature_name +' Upper: {}'.format(feat_upper))
        
        
        results = data[(data[feature] < feat_lower) | (data[feature] > feat_upper)].copy()
        results['Outlier'] = feature
        #outliers = outliers.append(results)
        outliers = pd.concat([outliers, results],axis=0)
        print(feature +' Count of Outliers: {}'.format(results.shape[0]))
    return outliers

paris_bbox_ll_ur= (2.220268,48.812799,2.473640,48.903715) #(2.219067,48.812629,2.421112,48.905464)
def check_lat_lon_brut(row, bbox_ll_ur=(2.219067,48.812629,2.421112,48.905464)):
    if (row["lat"] < bbox_ll_ur[1]) or (row["lat"] > bbox_ll_ur[3]):
        return False
    elif (row["lon"] < bbox_ll_ur[0]) or (row["lon"] > bbox_ll_ur[2]):
        return False
    else:
        return True

#%%
#pd.set_option('display.max_columns', df_elem.columns.size)

years_analysis = [2016,2017,2018,2019,2020,2021,2022]

#year_analysis = "2020"


#%%GET EXTRA DATABASE

#df["conseilquartier"] = df["conseilquartier"].apply(lambda x: x.strip())



conn = create_connection_sql(schema="dansmarue")
df_quartier=pd.read_sql('SELECT * FROM quartier', conn)
df_quartier["geom"] = df_quartier["geometry"].apply(lambda x: Polygon(shape(json.loads(x))))

# df_to_mer = df_quartier.loc[:,["quartier_id","name"]]

# df_clean = pd.merge(df,df_to_mer , left_on = "conseilquartier", right_on = "name",how="left")
# df_clean.drop(columns = ["conseilquartier","name"],inplace=True)

# print("Check if there are points with no QUartier _id")
# df_clean[df_clean["quartier_id"].isna()]
#%% READING THE DATA

for year_analysis in years_analysis:
    fname_format = fr"DMR_{year_analysis}.csv"

    fname = fr"{data_path}/{fname_format}"
    df = load_data_dmr_historique(fname)
    
    #%% GET BASIC INFORMATION OF INFO
    df.info()
    #%%
    df.head()
    #%%
    df.describe()
    #%%GETTING ORIGINAL COLUMNS NAME
    df.columns
    #%%REFORMATTING COLUMNS
    col_names_std = [ column.lower() for column in df.columns]
    df.columns = col_names_std
    df.rename(columns = {"x":"lon","y":"lat"},inplace=True)
    df.rename(columns = {"arrondissement":"code_postal_id"},inplace=True)
    df.rename(columns = {"type":"category_FR"},inplace=True)
    
    #%% CHECKING IF THERE ARE NaNs
    for column_ in df.columns:
        print(f"NaN values for columns {column_} = {df.loc[:,column_].isnull().sum()}")
        print(f"Percentage of NaN: {df.loc[:,column_].isnull().sum()/df.shape[0] * 100}")
    print("""
          The column with highest number of NaN values is `intervenant`
          However this is not necessary for further analysis so we do not need to drop it
          Same applies for other columns with NaN values like: `conseilquartier`, `adresse`
          """)
    #%%FORMATE DATE COLUMNS
    df["date_input"] = convert_date_format(df["datedecl"])#dfi["datedecl"].apply(convert_date_format)
    if int(year_analysis) >2020:
        df["date_etat"] = convert_date_format(df["dateetat"])#dfi["datedecl"].apply(convert_date_format)
        df["deltat_signal_etat"] = df["date_etat"] - df['date_input']
        df["deltadays_signal_etat"] =(df['deltat_signal_etat'].astype('timedelta64[s]'))/86400
        print(""" Before 2021 No registry of date of changement d'etat was included in data""")
        
    #%% DROPPING UNNECSSARY COLUMNS
    if int(year_analysis) >2020:
        df.drop(columns=["moisdecl","anneedecl","datedecl","dateetat"],inplace=True)
        df.drop(columns = "deltat_signal_etat",inplace=True)
    else:
        df.drop(columns=["moisdecl","anneedecl","datedecl"],inplace=True)
    df.drop(columns="ville",inplace=True)
    if "oid_" in df.columns:
        df.drop(columns = "oid_",inplace=True)
    #df.drop(columns="arrondissement",inplace=True)
    
    #%% CHECK FOR DUPLICATES
    for column_ in ["id_dmr",'numero']:
        print(f"Duplicated values for columns {column_}")
        print(f"Number of duplicates: {df[df[column_].duplicated()].shape[0]}")
    print("Dropping duplicated of id_dmr")
    df.drop_duplicates(subset="id_dmr",keep="last",inplace=True)
    
    #%% CHECK FOR INVALID DATA
    
    #For example invalid lat lon coordinates
    check_lat_lon = df.apply(check_lat_lon_brut, axis=1, args=(paris_bbox_ll_ur,))
    print(f"Check size of points outside bbox: {df[~check_lat_lon].shape[0]}")
    if df[~check_lat_lon].shape[0] == 0:
        print("""
              All points within Paris in theory
              """)
    else:
        print("""
              Dropping points outside of Paris
              """)
        df = df[check_lat_lon]
    #%% REPROJECTION FOR QUARTIER
    
    
    df.set_index("id_dmr",inplace=True)
    df_elem_copy = df.copy()
    df["quartier_id"] = None
    df_elem_copy["geom"] = df_elem_copy.apply(lambda x: Point(x.lon,x.lat),axis=1)
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
        df.loc[ids_in_quartier,"quartier_id"] = rowquartier["quartier_id"]
        df.loc[ids_in_quartier,"code_postal_id"] = rowquartier["code_postal_id"]
        #dropping matched elements for a faster analaysis in next loop
        df_elem_copy.drop(index=ids_in_quartier,inplace=True)
        end_time =  time.time()
    print(f"Time in seconds to assign Ids: {end_time-start_time}")
    
    print(f"""Number of not assigned points = {df["quartier_id"].isna().sum()}""")
    #df_elem = df_elem[~df_elem["quartier_id"].isna()]
    print(f"""Dropping points not assigned to code postal = {df["code_postal_id"].isna().sum()}""")
    
    df_clean = df[~df["code_postal_id"].isna()]
    df_clean.drop(columns = ["conseilquartier"],inplace=True)

    df_clean.reset_index(inplace=True)
    #%% EXTRACT EXTRA INFORMATION FROM SUBCATEGORY
    
    df_clean["category_FR"].value_counts()
    df_clean[["subcategory_FR","extrainfo"]] =  df_clean.apply(parse_soustype,axis=1, args=("soustype",), result_type="expand")
    df_clean.drop(columns = "soustype",inplace=True)
    
    #%% CHECK IF TYPE OF ACTIVITY DECLARED HAS ISSUES
    for category_ in df_clean["category_FR"].unique():
        df_sub = df_clean[df_clean["category_FR"]==category_]
        print(df_sub["subcategory_FR"].value_counts())
    #%%TRANSLATE CATEGORY to ENGLISH AND STANDARIZED VERSION
    df_clean["category_EN"] = df_clean["category_FR"].apply(lambda x: translate_category(x))
    #Verifying all have the same columns
    if year_analysis <2021:
        df_clean["etat"] = None
        df_clean["date_etat"] = None
        df_clean["deltadays_signal_etat"] = None
    df_clean = df_clean[['id_dmr', 'category_FR', 'adresse', 'code_postal',
           'code_postal_id', "etat", "date_etat",'numero', 'prefixe', 'intervenant', 'lon', 'lat',
           'date_input', 'quartier_id', 'subcategory_FR', 'extrainfo',
           "deltadays_signal_etat",
           'category_EN']]
    
    
    
    #%%
    from sqlalchemy.types import Integer,VARCHAR, FLOAT,DATE,TEXT
    dictionary_types = dict.fromkeys(['id_dmr', 'category_FR', 'adresse', 'code_postal', 'code_postal_id',
           'etat', 'numero', 'prefixe', 'intervenant', 'lon', 'lat', 'date_input',
           'date_etat', 'deltadays_signal_etat', 'quartier_id', 'subcategory_FR',
           'extrainfo', 'category_EN'])
    dictionary_types["id_dmr"] = VARCHAR(20)
    dictionary_types["category_FR"] = VARCHAR(50)
    dictionary_types["adresse"] = TEXT
    dictionary_types["code_postal"] = VARCHAR(10)
    dictionary_types["code_postal_id"] = VARCHAR(10)
    dictionary_types["etat"] = VARCHAR(30)
    dictionary_types["numero"] = Integer
    dictionary_types["prefixe"] = VARCHAR(5)
    dictionary_types["intervenant"] = VARCHAR(128)
    dictionary_types["lon"] = FLOAT
    dictionary_types["lat"] = FLOAT
    dictionary_types["date_input"] = DATE
    dictionary_types["date_etat"] = DATE
    dictionary_types["deltadays_signal_etat"] = FLOAT
    dictionary_types["quartier_id"] = VARCHAR(20)
    dictionary_types["subcategory_FR"] = TEXT
    dictionary_types["extrainfo"] = TEXT
    dictionary_types["category_EN"] = VARCHAR(128)
    
    
    #`id_dmr` varchar(20), 
    #               `type` varchar(50),
    #               `soustype` varchar(50),
    #               `adresse` varchar(512),
    #               `code_postal` varchar(10),
    #               `etat` varchar(20),
    #               `dateetat` varchar(50),
    #               `numero` varchar(20),
    #               `prefixe` varchar(5),
    #               `intervenant` varchar(50),
    #               `conseilquartier` varchar(128),
    #               `lon` float,
    #               `lat` float,
    #               `date_input` varchar(50),
    #               `subcategory` varchar(50),
    #               `extrainfo` varchar(512),
    #               `category` varchar(30),
    #%%
    to_sql = True
    to_csv = False
    if to_csv:
        df_clean.to_csv(fr"data/DMR_{year_analysis}_clean.csv",index=False)
    
    if to_sql:
        
        
        schema = "dansmarue"
        table_name = f"dmr_{year_analysis}_clean"
        password = "getpassword"
        connection_string = 'mysql+pymysql://root:{password}@127.0.0.1:3306/'
        engine = create_engine(connection_string)
        df_clean.to_sql(table_name,engine,schema,index=False,
                        dtype=dictionary_types,if_exists='replace')
    print(f"Finished with year :{year_analysis} ")
#%%Check OSM DATA
check_osm_data = False
if check_osm_data :
    from shapely.geometry import shape
    

    path_file= "data/osm"
    overwrite_csv = False
    
    #Which features to obtain from the OverpassAPI based on MapElements
    # https://wiki.openstreetmap.org/wiki/Map_features
    list_of_elements = ["shop", "amenity"]
    for key_element in list_of_elements:
        df_elem = pd.read_csv(fr"{path_file}/{key_element}_features.csv")
        print(f"Columns in dataframe {df_elem.columns}")
        print("Check NaNs")
        print(df_elem[["id_","lat","lon","geometry"]].isnull().sum())
        #We can drop NaN because we have no information on the position lat lon
        df_elem.dropna(subset=["lat","lon"],axis=0,inplace=True)
        
        #For example invalid lat lon coordinates
        check_lat_lon = df_elem.apply(check_lat_lon_brut, axis=1, args=(paris_bbox_ll_ur,))
        print(f"Check size of points outside bbox: {df_elem[~check_lat_lon].shape[0]}")
        print("""
              All points within Paris in theory
              """)
        
        if overwrite_csv:
            df_elem.to_csv(fr"{path_file}/{key_element}_features.csv",index=False)
        
    
    
