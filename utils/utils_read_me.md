## code for data validation - pandas
```python
import pandas as pd
import numpy as np

def data_validation(reference_data: pd.DataFrame, target_data: pd.DataFrame, id_column: str):
    # Set index to the identifier column for easier comparison (assuming `id_column` uniquely identifies rows)
    reference_data = reference_data.set_index(id_column)
    target_data = target_data.set_index(id_column)
    
    # Ensure columns in both DataFrames match
    common_columns = reference_data.columns.intersection(target_data.columns)
    reference_data = reference_data[common_columns]
    target_data = target_data[common_columns]

    # Align data types in each column
    for col in common_columns:
        if reference_data[col].dtype != target_data[col].dtype:
            if pd.api.types.is_numeric_dtype(reference_data[col]):
                reference_data[col] = pd.to_numeric(reference_data[col], errors='coerce')
                target_data[col] = pd.to_numeric(target_data[col], errors='coerce')
            elif pd.api.types.is_datetime64_any_dtype(reference_data[col]):
                reference_data[col] = pd.to_datetime(reference_data[col], errors='coerce')
                target_data[col] = pd.to_datetime(target_data[col], errors='coerce')
            else:
                reference_data[col] = reference_data[col].astype(str)
                target_data[col] = target_data[col].astype(str)

    # Fill NaN values with a placeholder to ensure consistent comparison
    reference_data = reference_data.fillna("NaN_placeholder")
    target_data = target_data.fillna("NaN_placeholder")
    
    # Align target rows with reference data by index (id_column)
    target_data = target_data.reindex(reference_data.index)

    # Dictionary to store mismatches by column
    mismatches = {col: [] for col in common_columns}

    # Check for matches and mismatches
    for col in common_columns:
        mismatched_indices = reference_data.index[reference_data[col] != target_data[col]].tolist()
        mismatches[col] = mismatched_indices

    # Count total matches and mismatches
    num_matches = len(reference_data) - sum(len(indices) for indices in mismatches.values())
    num_mismatches = len(reference_data) - num_matches

    # Filter out columns with no mismatches
    mismatches = {col: indices for col, indices in mismatches.items() if indices}

    # Result summary
    result = {
        "Number of Records Matched": num_matches,
        "Number of Records Mismatched": num_mismatches,
        "Mismatches by Column": mismatches
    }
    return result

```
