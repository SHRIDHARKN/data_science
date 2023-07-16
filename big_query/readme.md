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
