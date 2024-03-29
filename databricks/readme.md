![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/aa710bbe-b961-4d7c-9476-24097ea91aec)

### managed table and external table
  - first create an external table then create a managed table
```sql
-- Step 1: Create External Table
CREATE TABLE outdoorProductsRaw_external USING csv OPTIONS (
  path "/mnt/training/online_retail/data-001/data.csv",
  header "true"
);

-- Step 2: Create Managed Table using External Table
CREATE TABLE outdoorProductsRaw_managed
USING DELTA
LOCATION '/mnt/training/online_retail/data-001/managed_table/'
AS SELECT * FROM outdoorProductsRaw_external;

```
### temporary view vs view
  - view persists in the database whereas temporary view is available onyl for the duration of spark session
  - restarting session gives error

### pad to left upto n chars
```sql
lpad(<col name>,<number of chars>,<char to replace> )
```
```sql
select month,lpad(month,2,0 ) as month_padded from outdoorProducts order by month asc limit 5
```
### randomly sample data

```sql
select * from eventsRaw TABLESAMPLE (1 PERCENT)
```
```sql
select * from eventsRaw TABLESAMPLE (7 ROWS)
```
### working with dates
```sql
SELECT date_format(concat_ws("-", 2020, 10, 5), "yyyy-MM-dd")
```
```sql
SELECT to_date(concat_ws("/", "10", "05", "20"),"MM/dd/yy")
```
```sql
SELECT to_date(concat_ws("/", "10", "05", "2020"),"MM/dd/yyyy")
```
## set operators
### union, union all, intersect, intersect all, minus or except, except all
same structure - use set operators<br>
`union removes duplicate records`
```sql
select * from person1
union 
select * from person2
```
`union all provides all the records from A and B`
```sql
select * from person1
union all
select * from person2
```
`intersect - record common to both A and B, but without duplicates`
```sql
select * from person1
intersect
select * from person2
```sql
`intersect all - record common to both A and B, but with duplicates`
<br>
`except - records in A but not in B and without duplicates`
`except and minus are same`
```sql
select * from person1
except
select * from person2
```sql
`except all- records in A but not in B and with duplicates from A`
```sql
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
