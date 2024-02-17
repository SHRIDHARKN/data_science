# PySpark Install
### download Java
* jdk-21_windows-x64_bin
* JAVA_HOME -
  ```
  C:\Program Files\Java\jdk-21
  ```
### download Spark 3.5
* Spark 3.5.0 [link](https://www.apache.org/dyn/closer.lua/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz)
### create a folder "pyspark" and add subfolder hadoop:
### clone winutils in pyspark folder
*
  ```
  git clone https://github.com/cdarlint/winutils  
  ```
* copy hadoop3.3.5/bin contents and paste in a folder hadoop created above
* delete rest of the winutil files
### set paths in user environment variables
```%SPARK_HOME% - C:\pyspark\spark-3.5.0
```
```%HADOOP_HOME% - C:\pyspark\hadoop
```
```%JAVA_HOME% - C:\Program Files\Java\jdk-21
```
```%PYSPARK_HOME% - path to python executable
```
#### CLICK ON PATH AND ADD BIN PATHS
``%SPARK_HOME%\bin
```
```%HADOOP_HOME%\bin
```
```%JAVA_HOME%\bin
```
```%PYSPARK_HOME%
```
