from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, BinaryType
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,udf
import numpy as np
import tensorflow as tf
import os

spark = SparkSession.builder.appName("app").getOrCreate()
fp = "/mnt/e/data/text/news_headlines/stock_news.csv"
df = spark.read.format("csv").option("header", True).load(fp) 
df = df.select(["Date","Label","Top1"]).limit(100)

@udf(ArrayType(FloatType()))
def get_tokens():
    return [np.random.rand() for i in range(10)]

df = df.withColumn("tokens", get_tokens())
df = df.na.drop()

output_path = "/mnt/e/data/spark/tf_records_new"  # Adjust the path as needed

# ✅ Ensure the directory exists before writing
os.makedirs(output_path, exist_ok=True)

RECORDS_PER_FILE = 20  # ✅ Set the number of records per file

def write_tfrecords(partition_index, iterator):
    """Writes a partition of TFRecords to multiple files with a fixed batch size."""
    file_count = 0
    record_count = 0
    writer = None
    
    for row in iterator:
        if record_count % RECORDS_PER_FILE == 0:  
            # ✅ Close old writer & open a new file every RECORDS_PER_FILE records
            if writer:
                writer.close()
            file_path = f"{output_path}/part-{partition_index}-{file_count}.tfrecord"
            writer = tf.io.TFRecordWriter(file_path)
            file_count += 1
        
        if row and "tf_record" in row and row["tf_record"] is not None:
            writer.write(row["tf_record"])
            record_count += 1

    if writer:
        writer.close()  # ✅ Close last writer
    
    return iter([f"{file_count} files written for partition {partition_index}"])

def serialize_row(l,t):
    feature = {
        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[l.encode()])),
        "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[t.encode()])),
         
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

serialize_udf = udf(lambda label, text: serialize_row(label, text), BinaryType())
df_serialized = df.withColumn("tf_record", serialize_udf(col("Label"), col("Top1")))

# ✅ Use mapPartitionsWithIndex correctly
df_serialized.rdd.mapPartitionsWithIndex(write_tfrecords).collect()

print(f"✅ TFRecords saved successfully at: {output_path}")

raw_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(f"{output_path}/*.tfrecord"))

feature_description = {
    "label": tf.io.FixedLenFeature([], tf.string),
    "text": tf.io.FixedLenFeature([], tf.string)
}

def _parse_function(proto):
    return tf.io.parse_single_example(proto, feature_description)

# Apply Parsing
parsed_dataset = raw_dataset.map(_parse_function)

def extract_text_label(parsed_record):
    return parsed_record["label"], parsed_record["text"]

parsed_dataset = parsed_dataset.map(extract_text_label)

for label, text in parsed_dataset.take(5):  # View first 5 records
    print(f"Label: {label.numpy().decode('utf-8')}")
    print(f"Text: {text.numpy().decode('utf-8')}\n")

