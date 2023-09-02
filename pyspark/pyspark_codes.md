# Pyspark codes
---
### starting the pyspark session
```
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("pyspark").getOrCreate()
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
