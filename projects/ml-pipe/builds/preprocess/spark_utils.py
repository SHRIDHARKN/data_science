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
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")
# Get the number of threads
num_threads = multiprocessing.cpu_count()
print(f"Number of threads: {num_threads}")

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

# =========================SAMPLING=======================================================================

# ____________________________STRATIFIED_SAMPLE______________________________________________________________

def get_stratified_sample(df,group_col, sample_frac, seed=42):
    
    unique_labels = df.select(group_col).distinct().rdd.map(lambda row: row[group_col]).collect()
    fractions = {label: sample_frac for label in unique_labels}
    sampled_df = df.sampleBy(group_col, fractions, seed=seed)
    print("""
          Note:
          1. No. of records depends on seed. Different seed results in different number of records.
          2. sample_frac - probability of record being selected. sample_frac * no. records = no. records considered.
             0.3*5 = 1.5 records ~ 1 record
          3. Category A - 5 records, Category B - 2 record. Cateogry A might get excluded but Cateogry B might be included, given random nature and probable selection. 
             Category A - 0.3*5=round(1.5)=1 (randomly rejected)
             Category B - 0.3*2=round(0.6)=1 (randomly selected)
             
          """)
    return sampled_df 


# =========================SAVE_UTILS=======================================================================

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
