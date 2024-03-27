# Codes
## load images from folder
```python
def load_data_from_fold(fold_path,image_size=(64,64),batch_size=32,label_mode=None):

    dataset = tf.keras.preprocessing.image_dataset_from_directory( 
        fold_path, 
        label_mode=label_mode, 
        image_size=image_size, 
        batch_size=batch_size
    )    
    return dataset
```
