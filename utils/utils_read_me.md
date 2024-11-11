# setup project folders
```python
import os
# create project structure
def setup_project_structure(root_dir="."):
    
    print("NOTE: Folders would be created @ current directory. If you wish to create the folders else where, then use root_dir=<YOUR PATH>")
    # Define the folder structure
    folders = [
        "data/raw",
        "data/interim",
        "data/processed",
        "notebooks",
        "src/data",
        "src/features",
        "src/models",
        "src/utils",
        "models",
        "reports/figures",
        "reports/metrics",
        "tests",
        "configs"
    ]

    # Create the folders
    for folder in folders:
        os.makedirs(os.path.join(root_dir, folder), exist_ok=True)

    # Define essential files in the project root
    files = [
        ".gitignore",
        "README.md",
        "requirements.txt",
        "environment.yml",
        "setup.py",
        "main.py",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py"
    ]

    for file in files:
        with open(os.path.join(root_dir, file), "w") as f:
            pass  # Creates an empty file

    # Add example content to README.md
    with open(os.path.join(root_dir, "README.md"), "w") as f:
        f.write("# Project Name\n\nThis project is a data science project.\n\n## Folder Structure\n\n")

    print(f"Folder structure created successfully in {os.path.abspath(root_dir)}!")

```
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
