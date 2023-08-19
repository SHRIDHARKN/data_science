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
