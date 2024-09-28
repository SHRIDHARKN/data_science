# BigQuery
## Google Analytics Sample Data
**basic query**<br>
```SQL
SELECT date,channelGrouping as channel,totals.visits,totals.transactionRevenue 
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170801` 
WHERE channelGrouping IN ('Organic Search','Direct')
ORDER BY totals.transactionRevenue DESC
LIMIT 5
```
**conversion rate and average order value**<br>
```SQL
SELECT channelGrouping as channel,
SUM(totals.transactions)/SUM(totals.visits) as conversion_rate,
SUM(totals.transactionRevenue)/SUM(totals.transactions) as aov
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170801` 
WHERE channelGrouping IN ('Organic Search','Direct')
GROUP BY channelGrouping
```
**using case statements**<br>
```SQL
SELECT channelGrouping as channel,
CASE WHEN SUM(totals.visits)>0 THEN SUM(totals.transactions)/SUM(totals.visits)
ELSE 0 END as conversion_rate,
SUM(totals.transactionRevenue)/SUM(totals.transactions) as aov
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170801` 
WHERE channelGrouping IN ('Organic Search','Direct')
GROUP BY channelGrouping
```
**working with date time**<br>
```SQL
SELECT 
date_formatted,
EXTRACT(YEAR FROM date_formatted) as year,EXTRACT(MONTH FROM date_formatted) as month,
EXTRACT(DAY FROM date_formatted) as day
FROM (
SELECT PARSE_DATE('%Y%m%d',date) as date_formatted
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`)
```
**working with records**<br>
```SQL
SELECT page.pagePath,isEntrance
FROM 
`bigquery-public-data.google_analytics_sample.ga_sessions_20170801`
CROSS JOIN UNNEST(hits)
```
**working with json**<br>
*Example - ![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/0aca945e-f9f2-405c-a5cb-c4c2e250f97d)<br>
'work' column has data in json format. Unnest the column.*<br>
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/f672eb3f-2fa6-4949-956d-e4b53cfb5416)

```SQL
with data_formatted as (
SELECT name,place,userId,work from `<TABLE>`)
SELECT
  userId,
  STRING_AGG(JSON_EXTRACT_SCALAR(day, '$')) AS days_worked
FROM
  data_formatted,
  UNNEST(JSON_EXTRACT_ARRAY(work, '$.details.days_worked')) AS day
GROUP BY
  userId;
```
**mode -- most frequent**
```
SELECT MODE() WITHIN GROUP(ORDER BY order_occurrences)
FROM items_per_order
```


