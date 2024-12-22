# Pyspark
### create tmp storage and mount it
- this is useful in case there is disk spill by executors. Executors spill the data to /tmp/spark_temp temporarily 
```
mkdir /tmp/spark_temp
sudo mount -o size=2G -t tmpfs tmpfs /tmp/spark_temp
```
---
### creating a spark session
```
from pyspark.sql import SparkSession
def create_spark_session(app_name="app",num_cores=2,exec_memory="1g",driver_memory="1g",local_dir="/tmp/spark_temp"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(f"local[{num_cores}]") \
        .config("spark.executor.memory", exec_memory) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.local.dir", local_dir) \
        .getOrCreate()
    spark.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") # to avoid storing success files
    return spark   

```




### import libs
```
from pyspark.sql.types import *
```
### creating dataframe
#### method 1:
```
data = [(1,"chandler"),(2,"ross"),(3,"joey")]
schema = ["id","name"]
df = spark.createDataFrame(data=data,schema=schema)
df.show()
```
#### method 2:
```
data = [{"id":1,"name":"chandler"},{"id":2,"name":"ross"},{"id":3,"name":"joey"}]
schema = StructType([StructField(name="id",dataType=IntegerType()),StructField(name="name",dataType=StringType())])
df = spark.createDataFrame(data=data,schema=schema)
df.show()
```
### read csv file
```
schema = StructType().add(field='Date',data_type=StringType())\
                     .add(field='Total',data_type=IntegerType())

df = spark.read.csv(path=r'D:\data\tabular\time-series\sales\cat-fish-sales.csv',schema=schema,header=True)
df.show(5)
```
OR
```
df = spark.read.format('csv').option(key='header',value=True).load(\
                        path=r'D:\data\tabular\time-series\sales\cat-fish-sales.csv')

```
### array type
```
schema = StructType().add(field='id',data_type=StringType())\
                     .add(field='nums',data_type=ArrayType(IntegerType()))

data  = [('alice',[1,2]),('bob',[4,5]),('candice',[5,6])]
df = spark.createDataFrame(data=data,schema=schema)
df.show()
```
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/cb17a9c6-41aa-4f99-89ba-ce8b8763427d)
### second element of array
```
df.withColumn('second-element',col('nums')[1]).show()
```
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/989cc181-71ff-4621-a0ab-b94e4dc2e883)
### combine two columns into one as array
```
data = [("chandler",1,2),("ross",3,4),("joey",5,6)]
schema = ["name","num1","num2"]
df = spark.createDataFrame(data=data,schema=schema)
df.withColumn("nums",array(col("num1"),col("num2"))).show()
```
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/bea2e614-b16b-4e53-a379-73b60fcf276a)
### explode array columns
```
data = [(1,"chandler",["python","sql"]),(2,"ross",["machine-learning","deep-learninig"]),(3,"joey",["mongodb","aws"])]
schema = ["id","name","skill-set"]
df = spark.createDataFrame(data=data,schema=schema)
df.withColumn("skill",explode(col("skill-set"))).show()
```
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/f84dfea9-a1a6-48bc-9aa6-95ad28553bbf)
### string split to array
```
data = [(1,"chandler","python,sql"),(2,"ross","machine-learning,deep-learninig"),(3,"joey","mongodb,aws")]
schema = ["id","name","skill-set"]
df = spark.createDataFrame(data=data,schema=schema)
df.withColumn("skill",split(col("skill-set"),",")).show()
```
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/ffcfb6f1-df35-4278-8718-73ee06578a5b)
### check if array contains specific value
```
df.withColumn("has-python-skill",array_contains(col("skill-set"),'python')).show()
```
![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/bf7beda8-f046-4afe-b0db-199fcf32302a)
### working with dictionaries 
```
data = [(1,"chandler",{"hairs":"white","eyes":"blue"}),
        (2,"ross",{"hairs":"brown","eyes":"black"}),\
        (3,"joey",{"hairs":"black","eyes":"brown"})]

schema = StructType([StructField('id',IntegerType()),
                     StructField('name',StringType()),
                     StructField('appearance',MapType(StringType(),StringType()))])

df = spark.createDataFrame(data=data,schema=schema)
df.withColumn("hair",col("appearance")["hairs"]).show(truncate=False),\
df.withColumn("hair",col("appearance").getItem("hairs")).show(truncate=False)
```
