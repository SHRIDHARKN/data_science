![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/aa710bbe-b961-4d7c-9476-24097ea91aec)

## create a table and insert/ update values
```python
data = [("John", 30), ("Alice", 35), ("Bob", 40)]
df1 = spark.createDataFrame(data, ["name", "age"])
df1.write.saveAsTable("default.my_table", mode="overwrite")
```
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/3dff3a9c-b24b-456c-a992-5c82432183c2)

## overwrite the values in table created above
```python
data2 = [("Alice", 45), ("Bob", 40), ("Eve", 45)]
df2 = spark.createDataFrame(data2, ["name", "age"])
df2.write.saveAsTable("default.my_table", mode="overwrite")
```
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/f3b21a0d-21e3-4d4e-8778-f275ff4c93b1)

## insert values or update with latest values
```python
data = [("John", 30), ("Alice", 35), ("Bob", 40)]
df1 = spark.createDataFrame(data, ["name", "age"])
df1.write.saveAsTable("default.my_table", mode="overwrite")
data2 = [("Alice", 45), ("Bob", 40), ("Eve", 45)]
df2 = spark.createDataFrame(data2, ["name", "age"])
df2.createOrReplaceTempView("df2_temp_view")
```
```python
spark.sql("""
    MERGE INTO default.my_table AS target
    USING df2_temp_view AS source
    ON target.name = source.name
    WHEN MATCHED THEN UPDATE SET target.age = source.age
    WHEN NOT MATCHED THEN INSERT (name, age) VALUES (source.name, source.age)
""")
```
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/638f1c4c-6e56-449f-9ab6-19fb020311c7)



### make a directory
```
%fs 
mkdirs /dbfs/my_Files
```
### list the contents
```
%fs
ls /dbfs/my_Files/
```
### remove the file stored as csv/excel
```
%fs
rm -r /dbfs/my_Files/example.csv
```
### count nan and nulls
```python
def count_nan_and_nulls(df):
    nan_null_counts = df.select([
        count(when(isnan(c) | col(c).isNull(), c)).alias(c) 
        for c in df.columns
    ])
    display(nan_null_counts)
```
### null value counts in all cols
```python
def get_null_counts(df):
    null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    display(null_counts)
```
### item value counts in a col
```python
def get_value_counts(df,col_name,asc=False):
  value_counts_df = df.groupBy(col_name).count().orderBy("count", ascending=asc)
  display(value_counts_df)
```
### crosstab
```python
def generate_crosstab(df,col_axis,row_axis):
    crosstab_df = df.groupBy(col_axis).pivot(row_axis).count()
    display(crosstab_df)
```
### save csv
```python
def save_df_as_csv(df,file_name,folder_name="my_files",header="true"):
    
    save_path = f"/dbfs/{folder_name}/{file_name}"
    df.coalesce(1).write.option("header", header).csv(save_path)
```
### load csv
```python
def load_csv(file_name,folder_name="my_files",header="true"):
    
    save_path = f"/dbfs/{folder_name}/{file_name}"
    df = spark.read.option("header", header).csv(save_path)
    return df
```
### join dfs
```python
def join_dfs(df_left,df_right,join_columns,how="inner"):

    joined_df = df_left.join(df_right, on=join_columns,how=how)
    display(joined_df)
    return joined_df
```

#### split string
```python
df.select("name","age","geo_code",split(split(df["geo_code"],"-")[1],"_")[0].alias("location"))
```
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
### data types
#### SQL
`Integer Data Types`:
  - TINYINT: 1 byte (8 bits)
  - SMALLINT: 2 bytes (16 bits)
  - MEDIUMINT: 3 bytes (24 bits)
  - INT/INTEGER: 4 bytes (32 bits)
  - BIGINT: 8 bytes (64 bits)

`Floating-Point Data Types:`:
  - FLOAT: 4 bytes (single precision)
  - DOUBLE: 8 bytes (double precision)

#### SPARK
  - ByteType: 1-byte signed integer.
  - ShortType: 2-byte signed integer.
  - IntegerType: 4-byte signed integer.
  - LongType: 8-byte signed integer.
  - FloatType: 4-byte single-precision floating-point number.
  - DoubleType: 8-byte double-precision floating-point number.
  - DecimalType: Arbitrary-precision signed decimal.

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
### get week day from date
```sql
select *,date_format(date,"E" ) as weekday from salesDateFormatted
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
```
`intersect all - record common to both A and B, but with duplicates`
<br>
`except - records in A but not in B and without duplicates`
`except and minus are same`
```sql
select * from person1
except
select * from person2
```
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
