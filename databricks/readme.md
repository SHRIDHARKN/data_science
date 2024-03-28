![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/aa710bbe-b961-4d7c-9476-24097ea91aec)

### randomly sample data

```
select * from eventsRaw TABLESAMPLE (1 PERCENT)
```
```
select * from eventsRaw TABLESAMPLE (7 ROWS)
```
## set operators
### union, union all, intersect, intersect all, minus or except, except all
same structure - use set operators<br>
`union removes duplicate records`
```
select * from person1
union 
select * from person2
```
`union all provides all the records from A and B`
```
select * from person1
union all
select * from person2
```
`intersect - record common to both A and B, but without duplicates`
```
select * from person1
intersect
select * from person2
```
`intersect all - record common to both A and B, but with duplicates`
<br>
`except - records in A but not in B and without duplicates`
`except and minus are same`
```
select * from person1
except
select * from person2
```
`except all- records in A but not in B and with duplicates from A`
```
select * from person1
except all
select * from person2
```









### convert date col in string to required format
```
data = [("15-01-2023",), ("20-02-2023",), ("25-03-2023",)]
df = spark.createDataFrame(data, ["date_str"])
df = df.withColumn("date", to_date(df["date_str"], "dd-MM-yyyy"))
```
