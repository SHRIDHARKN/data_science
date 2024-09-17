## xgboost classifier
* **colsample_bytree** <br>
  -  Example: Suppose you have features [A, B, C, D, E, F, G, H, I, J]. If colsample_bytree=0.5, XGBoost might randomly pick [A, C, E, G, H] to build one tree and [B, D, F, I, J] to build another.<br>
* **colsample_bylevel** <br>
  - Example: If your features are [A, B, C, D, E, F, G, H, I, J] for a **tree**, and colsample_bylevel=0.7, XGBoost might choose [A, C, F, G, H, I, J] for one level and [B, D, E, F, G, H, I] for the next.<br>
* **colsample_bynode** <br>
  - Example: For a node deciding on a split, if there are 10 features for a **level** and colsample_bynode=0.5, XGBoost might choose to consider only 5 features like [A, D, F, G, J] for that particular split. The next node might use a different set of features like [B, C, E, H, I].<br>

## data transformation
* **Right-Skewed Data:** **Logarithmic** and **square root** transformations are often appropriate.
* **Left-Skewed Data:** : mean<median<mode **Exponential** and **reciprocal** transformations may be useful.
## entropy
$$Entropy = - \sum_{i=1}^{n} **p** \ log(**p**)$$
## logistic regression derivation
![log_reg_notes](https://github.com/SHRIDHARKN/data_science/assets/74343939/81cbc9ae-95c8-456f-8762-3a1453d8577d)
## support vector classifier
![support vector classifier](https://github.com/SHRIDHARKN/data_science/assets/74343939/c5f79abe-81cc-4605-a923-a5a80b6b9f3c)
## gradient boost classifier
![gradient boost classifier](https://github.com/SHRIDHARKN/data_science/assets/74343939/967c09a0-13ac-424b-bb62-b2e9e9c38164)

