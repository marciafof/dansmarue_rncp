create database if not exists dansmarue;
USE dansmarue;

 -- JOIN TABLES FROM EACH YEAR
 CREATE TABLE dmr_all
	SELECT * FROM dmr_2016_clean
	UNION ALL
	SELECT * FROM dmr_2017_clean
	UNION ALL
	SELECT * FROM dmr_2018_clean
	UNION ALL
	SELECT * FROM dmr_2019_clean
	UNION ALL
	SELECT * FROM dmr_2020_clean
	UNION ALL
	SELECT * FROM dmr_2021_clean
	UNION ALL
	SELECT * FROM dmr_2022_clean
    ;
 
 /* ---------------------- OVERALL VIEW *------------------ */  
    -- GET THE TOTAL COUNT OF REPORTS
SELECT count(*) as count
from dmr_all
ORDER by count DESC ;

    -- GET THE AVERAGE OF TOTAL COUNT OF REPORTS PER WEEK
SELECT count_p_week.year, AVG(count_p_week.count)
FROM
	(SELECT extract(year from date_input) as year, extract(week from date_input) as week, count(*) as count
	from dmr_all
	group by year, week
	ORDER by count DESC) as count_p_week
group by year
;

   -- GET THE COUNT PER YEAR  COMPARE TO PREVIOUS YEAR IN PERC
WITH yearly_count as (
 SELECT 
   EXTRACT(year from date_input) as year,
   COUNT(*) as count
 FROM dmr_all	 
 GROUP BY year
)
	select
	  *,
	  (yearly_lag.count - yearly_lag.count_previous_year) / yearly_lag.count_previous_year * 100 as perc_diff
	from (
			SELECT yearly_count.year, yearly_count.count,
					LAG(count) OVER ( ORDER BY year ) AS count_previous_year
			from yearly_count
            ) as yearly_lag
	GROUP BY year;
			
    
-- GET THE COUNT PER YEAR PER MONTH AND COMPARE TO PREVIOUS MONTH
WITH yearly_monthly_count as (
 SELECT 
   extract(year from date_input) as year,
   extract(month from date_input) as month,
   COUNT(*) as count
 FROM dmr_all	 
 GROUP BY year, month 
)
	SELECT yearly_monthly_count.year, yearly_monthly_count.month, yearly_monthly_count.count,
			LAG(count) OVER ( ORDER BY year , month) AS count_previous_month
	from yearly_monthly_count;
  
-- GET THE COUNT PER MONTH AND COMPARE TO PREVIOUS MONTH
WITH monthly_count as (
 SELECT 
   extract(month from date_input) as month,
   COUNT(*) as count
 FROM dmr_all	 
 GROUP BY  month 
)
	SELECT monthly_count.month, monthly_count.count,
			LAG(count) OVER ( ORDER BY month) AS count_previous_month
	from monthly_count;
    
 -- GET THE COUNT PER ARRONDISSEMENT
SELECT code_postal_id, COUNT(*) as count
FROM dmr_all
GROUP BY code_postal_id
ORDER BY count DESC;

 -- GET THE COUNT OF CATEGORY 
SELECT category_FR, count(*) as count
from dmr_all
group by category_FR
ORDER by count DESC ;

 -- GET THE COUNT OF CATEGORY PER YEAR
SELECT category_FR, EXTRACT(YEAR FROM date_input) AS year, count(*) as count
from dmr_all
group by category_FR, year
ORDER by count DESC ;

 /* ---------------------- UNDERSTANDING ETAT FIELD *------------------ */  

-- GET UNIQUE ETAT CATEGORIES
  select count(distinct etat) from dmr_all;
 
-- GET THE COUNT PER ETAT
SELECT etat, COUNT(`id_dmr`) as count_events 
 FROM dmr_all
 GROUP BY etat
 ORDER by count_events DESC;
 
  /* ---------------------- JOINING WITH OTHER FIELDS *------------------ */  

-- GET COLUMNS FROM shop_georef
  select * from shop_georef;
 /* id_, lat, lon, category, subcategory, name, geometry, quartier_id, code_postal_id */
 -- SHOPS GROUP BY quartier_id
 SELECT category, quartier_id, code_postal_id, COUNT(id_) as count_shops
 FROM shop_georef
 GROUP BY category, quartier_id, code_postal_id;
 
WITH count_by_quartier AS (
	SELECT code_postal_id, quartier_id, COUNT(*) as count
	FROM dmr_all
	GROUP BY code_postal_id, quartier_id)
    
	SELECT *
	FROM count_by_quartier
	LEFT JOIN 
			(SELECT category, quartier_id, code_postal_id, COUNT(id_) as count_shops
			 FROM shop_georef
			 GROUP BY category, quartier_id, code_postal_id) shops_by_quartier
	ON count_by_quartier.quartier_id = shops_by_quartier.quartier_id
         
;
 


