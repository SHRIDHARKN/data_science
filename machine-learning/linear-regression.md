# Linear regression

## check for multicollinearity - VIF
```python
def get_vif_info(df,features,target):
    
    X = df[features]
    y = df[target]

    vif_data = pd.DataFrame()
    vif_data['features'] = features
    # calculate VIF for each feature
    vif_scores = []
    for feature in features:
        all_feats_except_x, x = X.drop(columns=[feature]), X[feature]
        model = LinearRegression().fit(all_feats_except_x, x)
        r2 = model.score(all_feats_except_x, x)
        vif_i = 1/(1-r2)
        vif_scores.append(vif_i)
        
    vif_data["VIF"] = vif_scores
    
    return vif_data
```
