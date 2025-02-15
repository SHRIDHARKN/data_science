import tensorflow as tf
from functools import partial

# ____________________________________________________________________________________________________________________
# code to convert spark dataframe to tf records

def write_tfrecords(partition_index, iterator, output_path,records_per_file):
    """Writes a partition of TFRecords to multiple files with a fixed batch size."""
    file_count = 0
    record_count = 0
    writer = None
    
    for row in iterator:
        if record_count % records_per_file == 0:  
            # ✅ Close old writer & open a new file every records_per_file records
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

"""
def serialize_row(l, t):
    feature = {
        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[l.encode()])),
        "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[t.encode()])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# ✅ Register UDF for serialization
serialize_udf = udf(lambda label, text: serialize_row(label, text), BinaryType())

# ✅ Add serialized TFRecords column
df_serialized = df.withColumn("tf_record", serialize_udf(col("Label"), col("Top1")))


write_tfrecords_with_args = partial(write_tfrecords, output_path=output_path, records_per_file=records_per_file)

# ✅ Apply mapPartitionsWithIndex correctly
df_serialized.rdd.mapPartitionsWithIndex(write_tfrecords_with_args).collect()

"""

# ____________________________________________________________________________________________________________________
# code to read tf records

def _parse_function(proto,feature_description):
    return tf.io.parse_single_example(proto,feature_description)

def extract_text_label(parsed_record):
    return parsed_record["label"], parsed_record["text"]

def load_tf_records(output_path,feature_description):
    raw_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(f"{output_path}/*.tfrecord"))
    parsed_dataset = raw_dataset.map(lambda proto: _parse_function(proto, feature_description))
    return parsed_dataset

"""
feature_description = {
    "label": tf.io.FixedLenFeature([], tf.string),
    "text": tf.io.FixedLenFeature([], tf.string)
}
raw_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(f"{output_path}/*.tfrecord"))
parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset = parsed_dataset.map(extract_text_label)
"""
# ____________________________________________________________________________________________________________________
