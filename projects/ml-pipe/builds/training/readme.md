import tensorflow as tf
import torch
import tensorflow as tf
import glob
from torch.utils.data import DataLoader
import os
import click


class TFRecordDataset(torch.utils.data.Dataset):
    def __init__(self, tfrecord_dir,feature_description):
        self.filenames = glob.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
        self.dataset = tf.data.TFRecordDataset(self.filenames)  # Load all TFRecord files
        self.feature_description = feature_description

    def _parse_function(self, example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, self.feature_description)

        # Decode text and label (from bytes to string)
        text = parsed_example["text"].numpy().decode("utf-8")
        label = parsed_example["label"].numpy().decode("utf-8")

        # Convert tokens and label_encoded to PyTorch tensors
        tokens = torch.tensor(parsed_example["tokens"].numpy(), dtype=torch.long)
        label_encoded = torch.tensor(parsed_example["label_encoded"].numpy(), dtype=torch.long)

        return text, label, tokens, label_encoded

    def __len__(self):
        return sum(1 for _ in self.dataset)

    def __getitem__(self, index):
        iterator = iter(self.dataset.skip(index).take(1))
        example = next(iterator)  # Get single record
        return self._parse_function(example)

@click.command()
@click.option("--project_name")
@click.option("--batch_size",default=8)
@click.option("--token_max_length",default=512)
def main(project_name,batch_size,token_max_length):
    # Path where TFRecords are stored
    tfrecord_path = f"/mnt/d/data/{project_name}/preprocessed"
    feature_description = {
        "text": tf.io.FixedLenFeature([], tf.string),  # Variable-length string
        "label": tf.io.FixedLenFeature([], tf.string),  # Variable-length string
        "tokens": tf.io.FixedLenFeature([token_max_length], tf.int64),  # Fixed-length sequence
        "label_encoded": tf.io.FixedLenFeature([], tf.int64)  # Single integer
    }

    # Create dataset
    dataset = TFRecordDataset(tfrecord_dir=tfrecord_path,feature_description=feature_description)
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Example: Iterate through the DataLoader
    for batch in dataloader:
        text, label, tokens, label_encoded = batch
        print(f"Text: {text}")
        print(f"Label: {label}")
        print(f"Tokens: {tokens.shape}")  # Should be (batch_size, FIXED_TOKENS_LEN)
        print(f"Label Encoded: {label_encoded.shape}")
        break 
    
# Entry point
if __name__ == "__main__":
    main()
