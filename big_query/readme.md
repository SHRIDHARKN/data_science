# BigQuery
*basic query*<br>
```
SELECT date,channelGrouping as channel,totals.visits,totals.transactionRevenue 
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170801` 
WHERE channelGrouping IN ('Organic Search','Direct')
ORDER BY totals.transactionRevenue DESC
LIMIT 5
```
