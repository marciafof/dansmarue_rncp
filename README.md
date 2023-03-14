<img src="img/dansmarue_logo.jpg"  width="10%" height="10%">

# DansMaRue - a crowdsourcing service for anomaly detection in the city of Paris

Welcome to the introduction of my Data Analysis project "DansMaRue"

⚠️ **This is a work in progress**

DansMaRue is a well-established crowdsourcing service with over 1000 records of anomalies a day. The collected data, which includes geographical position and imagery (photos), is currently only used as a collective monitoring tool. However, given the volume of anomalies registered each year, could we be missing important insights?

## Objective

This project has the objective of exploring additional ways to utilize the collected data and prevent the loss of any valuable information. 
By doing so, the goal is to develop a framework to identify key parameters in the data that could improve the efficiency of services dedicated to resolving anomalies. 

Furthermore, it provides a large playground field for any Data Analyst interested in testing its skills in data cleaning and visualization.

## Structure of the project

This project is divided following the classical path in Data Analytics: Prepare, Process, Analyse and Share.
- Data collecting
- Data cleaning and processing
- EDA

### Data collecting

Links to the main collected datasets:
 - [Open Data Ville de Paris](https://opendata.paris.fr/pages/home/)
 - [Historical data DansMaRue](https://parisdata.opendatasoft.com/explore/dataset/dans-ma-rue-historique-anomalies-signalees/information/)
 - [INSEE Comparateur des territoires Paris](https://www.insee.fr/fr/statistiques/1405599?geo=DEP-75)
 - [OpenStreetMap Python API](https://wiki.openstreetmap.org/wiki/OSMPythonTools)

The covered period of study is from 2016 to 2022.

### Data cleaning

- [Data cleaning script](scripts/data_cleaning.py)
- [SQL Queries](scripts/sql_insights.sql)
- [Python script for aggregation and formatting](scripts/data_to_dataviz.py)

### EDA

The current analysis was done using the software Tableau Desktop.

## Some insights

... TBD..

