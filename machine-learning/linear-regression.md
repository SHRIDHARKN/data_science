# Linear regression

### get scaled data
```python
def get_scaled_data(df,features,target):
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features)
    scaled_df[target] = df[target].values

    return scaled_df,scaler

```
### get feature coeffs
```python
def get_feature_imp(model):
    df_feat_imp = pd.DataFrame()
    df_feat_imp["features"] = model.feature_names_in_
    df_feat_imp["feature_coeffs"] = model.coef_
    return df_feat_imp
```
### check for multicollinearity - VIF
```python
def get_vif_info(df,features,target,vif_thr=5):
    
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
    vif_data["multicollinearity_exists"] = np.where(vif_data["VIF"]>vif_thr,True,False)
    feats_2_drop = vif_data.features.loc[vif_data.multicollinearity_exists].values.tolist()
    return vif_data,feats_2_drop
```
**Positive Autocorrelation:** model is consistently overshooting the target **(making predictions that are consistently too high)** or undershooting **(making predictions that are consistently too low)**. Basically **model isn't learning from its mistakes**.

**Negative Autocorrelation:** Model will **overshoot, then undershoot, then overshoot again**. This can also be a problem because your model is **overcompensating**, and it's not making very accurate predictions.

In a good model, the **errors** should be **random** and **not show a consistent pattern like overshooting or undershooting**. Autocorrelation tells us that our model isn't learning from its mistakes and may need some adjustments to make more accurate predictions.
### check for autocorrelation of residuals
```python
def perf_durbin_watson_test(residuals):

    # Perform the Durbin-Watson test
    dw_statistic = durbin_watson(residuals)
    # Check for autocorrelation
    if dw_statistic < 1.5:
        result = "positive autocorrelation of residuals"
    elif dw_statistic > 2.5:
        result = "negative autocorrelation of residuals"
    else:
        result = "no significant autocorrelation of residuals"

    return {"Durbin-Watson statistic":round(dw_statistic,2),
            "result":result}
```
If a model is doing well for predicting prices of smaller houses, but it's doing poorly for larger houses, then predictions for the larger houses have a wider range of potential errors.
Inefficient Parameter Estimates: Using more data for some parts of your model and less for others.
