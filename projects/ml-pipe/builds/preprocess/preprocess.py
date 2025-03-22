# load  libs
from spark_utils import start_spark_session
from pyspark.sql.functions import concat, lit, col
from pyspark.sql.functions import udf
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, IntegerType
from transformers import AutoTokenizer
from pyspark.sql.functions import monotonically_increasing_id
from spark_utils import save_as_tf_records, save_as_parquet
from spark_utils import show_msg
import os
import pickle
import click

# Correcting the tokenizer_udf function
@udf(ArrayType(IntegerType()))
def tokenizer_udf(text):
    tokenizer = tokenizer_broadcast.value
    tokens = tokenizer.encode(text, truncation=True, padding="max_length", max_length=512)
    return tokens

def create_label_dictionary(spark_df, label_col):
    # Get unique labels from the DataFrame
    unique_labels = spark_df.select(label_col).distinct().rdd.flatMap(lambda x: x).collect()
    # Create a dictionary mapping label names to IDs
    label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    return label_to_id

def save_dictionary_as_pickle(dictionary, pickle_file_path):
    # Save the dictionary as a pickle file
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(dictionary, pickle_file)
    show_msg(f"Dictionary saved as pickle @ {pickle_file_path}")

def add_label_id_column(df, label_col, label_map_dict, label_id_col):
    # Define a UDF to map labels to IDs
    map_label_to_id_udf = F.udf(lambda label: label_map_dict.get(label, -1), IntegerType())
    # Add the new column using the UDF
    df_with_label_id = df.withColumn(label_id_col, map_label_to_id_udf(F.col(label_col)))
    return df_with_label_id

# Main function using Click
@click.command()
@click.option("--project_name")
@click.option("--label_col")
@click.option("--text_col")
@click.option("--tokenizer_model", default="bert-base-uncased", help="The tokenizer model to use (default: 'bert')")
@click.option("--token_col", default="tokenized_text", help="Column contains tokenized text data")
@click.option("--save_format", default="tfrecords", help="Format to save spark dataframes")
@click.option("--required_cols", default=None, help="Columns to keep while saving spark dataframe")
@click.option("--records_per_file", default=5000, help="Number of records per file")
def main(project_name, tokenizer_model, text_col, label_col, token_col, save_format, required_cols, records_per_file):
    
    file_path = f"/mnt/d/data/{project_name}/raw/{project_name}.csv"
    output_dir = f"/mnt/d/data/{project_name}/preprocessed"
    label_map_file_path = f"/mnt/d/data/{project_name}/constants"
    label_encoded_col = "label_encoded"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_map_file_path, exist_ok=True)
    
    spark = start_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    df = spark.read.csv(file_path, header=True, multiLine=True, escape="\"")
    print(df.show(5))
    # Load a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    # Broadcast the tokenizer
    global tokenizer_broadcast
    tokenizer_broadcast = spark.sparkContext.broadcast(tokenizer)
    show_msg(f"Loaded {tokenizer_model} model tokenizer")
    
    # Apply the tokenizer_udf to tokenize text
    df = df.withColumn(token_col, tokenizer_udf(df[text_col]))
    
    # get unique labels for encoding
    label_map_dict = create_label_dictionary(spark_df=df, label_col=label_col)
    save_dictionary_as_pickle(dictionary=label_map_dict, pickle_file_path=os.path.join(label_map_file_path,"label_map.pkl"))
    df = add_label_id_column(df=df, label_col=label_col, label_map_dict=label_map_dict, label_id_col=label_encoded_col)
    
    if required_cols is None:
        df = df.select([text_col,label_col,token_col,label_encoded_col])
    if save_format=="parquet":
        save_as_parquet(df, output_dir, max_records_per_file=records_per_file)
    else:
        df = df.withColumn("id", monotonically_increasing_id())
        df = df.withColumn("partition", (df["id"] / records_per_file).cast("int"))
        df = df.repartition("partition")
        save_as_tf_records(df=df, output_dir=output_dir, text_col=text_col, label_col=label_col, label_encoded_col=label_encoded_col, token_col=token_col)
    
    spark.stop()
    
# Entry point
if __name__ == "__main__":
    main()
