import os
import multiprocessing
from functools import partial
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import tensorflow as tf
import warnings
warnings.simplefilter("ignore")

# ____________________________________________________________________________________________________________________
# Get the number of CPU cores
print("==============================================================================================================")
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")
# Get the number of threads
num_threads = multiprocessing.cpu_count()
print(f"Number of threads: {num_threads}")
print("==============================================================================================================")

def show_msg(msg):
    print("="*50)
    print(msg)
    print("="*50)
    
def start_spark_session(app_name="app",num_cores=2,exec_memory="1g",driver_memory="1g",local_dir="/tmp/spark_temp"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(f"local[{num_cores}]") \
        .config("spark.executor.memory", exec_memory) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.local.dir", local_dir) \
        .getOrCreate()
    #spark.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") # to avoid storing success files
    return spark   

# _________________________PARQUET FILES____________________________________________________________________
def save_as_parquet(df, output_path, max_records_per_file=5000):
    show_msg(f"Saving DataFrame as Parquet files with maxRecordsPerFile = {max_records_per_file}")
    df.write.option("maxRecordsPerFile", max_records_per_file).mode("overwrite").parquet(output_path)
    show_msg(f"DataFrame saved as Parquet files in: {output_path}")

# _____________________________TF RECORDS____________________________________________________________________

def save_as_tf_records(df,output_dir, text_col, label_col, token_col, label_encoded_col):
    def serialize_example(row):
        feature = {
            "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[row[text_col].encode()])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[row[label_col].encode()])),  # Ensure label is string
            "tokens": tf.train.Feature(int64_list=tf.train.Int64List(value=row[token_col])) , # Use int64_list instead of bytes_list
            "label_encoded": tf.train.Feature(int64_list=tf.train.Int64List(value=row[token_col]))  # Use int64_list instead of bytes_list
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    # Function to write each partition to TFRecord
    def write_partition(index, iterator):
        file_path = os.path.join(output_dir, f"part-{index:05d}.tfrecord")
        with tf.io.TFRecordWriter(file_path) as writer:
            for row in iterator:
                writer.write(serialize_example(row))
        return iter([])
    df.rdd.mapPartitionsWithIndex(write_partition).collect()
    show_msg(f"TFRecords saved @ {output_dir}")
