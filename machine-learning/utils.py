# load libs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rich.progress import track
from rich.progress_bar import ProgressBar

import pandas as pd
import numpy as np

# Sklearn Libs
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold,mutual_info_classif
from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import chi2,SequentialFeatureSelector

# Visualization Libs
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
tqdm.pandas()

# Statisitcal Test Libs
from scipy.stats import kendalltau as Kendal_Tau_Correlation
from scipy.stats import pointbiserialr as Point_Biserial_Correlation
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_goldfeldquandt
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import wilcoxon
from scipy.stats import f_oneway
from scipy.stats import ttest_ind

# Sampling Libs
# from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import ADASYN

import joblib
import pickle
import ast
import os
import datetime
import calendar
import re
import copy
import pytz
import time
tz = pytz.timezone("Asia/Calcutta")
import warnings
warnings.simplefilter("ignore")

###################### TIMESERIES #############################
# libs
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox


def build_ar_model(df,lags):
    ar_model = AutoReg(df, lags=lags).fit()
    print("\nAR Model Summary:")
    print(ar_model.summary())
    return ar_model

def perform_adf_test(df):
    adf_test = adfuller(df)
    print('ADF Statistic: %f' % adf_test[0])
    print('Critical Values @ 0.05: %.2f' % adf_test[4]['5%'])
    print('p-value: %f' % adf_test[1])
    stat_cond = True if adf_test[1]<0.05 else False
    print("Stationary time series",stat_cond)
    return {
        "p-value":adf_test[1],
        "Critical Values @ 0.05":adf_test[4]['5%'],
        "ADF Statistic":adf_test[0],
        "Stationary":stat_cond
    }      
    

def check_diff_for_stationarity(df,diff_start,diff_end,col_name=None):
    if col_name:
        df = df[col_name]
    best_diff = diff_end
    for i in tqdm(range(diff_start,diff_end)):
        print("For differencing of :",i)
        adf_test = perform_adf_test(df=df.diff(i).dropna())
        print("#########################")
        if adf_test["Stationary"]:
            best_diff = i
            print(f"Differencing of {best_diff} will make time series stationary")
            break

def get_boxcox_transf_data(df, col_name=None):
    if col_name:
        res = boxcox(df[col_name])
    else:
        res = boxcox(df)
    trf_vals, lam = res
    
    # Reshape the arrays to 2D
    trf_vals = trf_vals.reshape(-1, 1)
    lam_vals = np.array([lam] * len(df)).reshape(-1, 1)

    # Create a DataFrame
    df_transformed = pd.DataFrame({'box-cox': trf_vals.flatten(), 'box-cox-lambda': lam_vals.flatten()})
    return df_transformed.values    

def inverse_boxcox(y, lambda_):
  if lambda_ == 0:
    return np.exp(y)
  else:
    return np.exp(np.log(lambda_ * y + 1) / lambda_)

def load_csv(fp,header_=True):
    return pd.read_csv(fp,header=header_)

tabular_data_folder = r"D:\data\tabular"
time_series_data_folder = r"D:\data\tabular\time-series"

def what_2_do(task):
    task_dict = {
        "feature_selection":["sequential_feature_selection"]    
    }
    return what_2_do.get(task,"no idea")
    

##################### text data processing #####################
def remove_extra_spaces(text):
    return re.sub("\s+"," ",text)

def preprocess_data_col_names(col_names):
    show_message("""col names changed to lower and removed extra space""")
    show_message("""replaced col names with space by underscore""")
    return [remove_extra_spaces(x.lower()).replace(" ","_") for x in col_names]

# display msgs
def format_pandas_output():
    pd.set_option('display.float_format', '{:.2f}'.format)
    
def show_message(msg,msg_speed=0.005):
    msg = msg+"\n"
    for i in msg:
        print(i,end='', flush=True)
        time.sleep(msg_speed)

##################### dataframe #####################
#___________________________________________________________________________#
def get_balanced_sample(df,group_col,num_samples,random_state_=42):
    
    show_message(f"getting data with equal number of samples from each of groups in {group_col}")
    categories = np.unique(df[group_col])
    df_bal = pd.DataFrame()
    for cat in categories:
        dfi = df.loc[df[group_col]==cat].sample(num_samples,random_state=random_state_)
        df_bal = pd.concat([df_bal,dfi])
    df_bal = df_bal.reset_index(drop=True)
    show_message(f"returning concatenated df =  dfg1 + dfg2 + .......+dfgn")
    return df_bal

def get_columns_with_nan(df):
    return df.columns[df.isna().any()].tolist()


# backward or forward feature selection method
def sequential_feature_selection(X,y,model,n_feats_2_select=3,direction="forward"):
    
    show_message(f"executing {direction} feature selection method")
    show_message(f"num feats to select = {n_feats_2_select}")
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_feats_2_select,
                                    direction=direction)
    sfs.fit(X, y)
    selected_indices = sfs.get_support(indices=True)
    show_message(f"total num feats = {X.shape[1]}, num feats selected = {len(selected_indices)}")
    return  X.iloc[:, selected_indices]



def get_samples_to_draw(df,col):
    res = dict(df[col].value_counts())
    min_so_far = np.inf
    for k,v in res.items():
        if v<min_so_far:
            min_so_far=v
    return min_so_far



# pandas data check

def get_strat_batch(file_name_list,label_list,major_class,minor_class):
    
    skip_fact = label_list.count(major_class)//label_list.count(minor_class)
    final_ls = []
    for file_name,label_i in zip(file_name_list,label_list):
        if label_i==major_class:
            final_ls.append(file_name)
    
    j=0
    for file_name,label_i in zip(file_name_list,label_list):
        if label_i==minor_class:
            final_ls.insert(j,file_name)
            j+=skip_fact
    
    return final_ls        

# model development
def get_train_test_split(X,y,test_size=0.25,stratify=None):
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=test_size,stratify=stratify)
    return x_train,x_test,y_train,y_test

def get_clf_predictions(clf,X,x_train,x_test):
    
    y_pred_test = clf.predict(x_test)
    y_pred_proba_test = clf.predict_proba(x_test)
    
    y_pred_train = clf.predict(x_train)
    y_pred_proba_train = clf.predict_proba(x_train)
    
    y_pred_all = clf.predict(X)
    y_pred_proba_all = clf.predict_proba(X)
    
    results = {}
    results["y_pred_test"] = y_pred_test
    results["y_pred_proba_test"] = y_pred_proba_test
    results["y_pred_train"] = y_pred_train
    results["y_pred_proba_train"] = y_pred_proba_train
    results["y_pred_all"] = y_pred_all
    results["y_pred_proba_all"] = y_pred_proba_all
    return results

def get_optimal_roc_auc_thr(fpr,tpr,thr,round_off=2):
    opt_idx = np.argmax(tpr-fpr)
    optimal_thr = round(thr[opt_idx],round_off)
    return opt_idx,optimal_thr

def get_clf_performance_details(clf,X,y):
    
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)
    fpr_,tpr_,thr_ = get_roc_curve_details(y,y_proba)
    tnr_ = 1-fpr_
    fnr_ = 1-tpr_
    return y_pred,y_proba,fpr_,tpr_,tnr_,fnr_,thr_

def plot_roc_curve(fpr,tpr,thr,plot_color="blue",optimal_thr_color="green",
                font_size_=8,title_="",x_ticks_range = None,y_ticks_range = None,
                optimal_thr_line_width=1,grid_line_width=0.3,round_off=2):
    
    if x_ticks_range is None:
        x_ticks_range = np.arange(0,1,0.15)
    if y_ticks_range is None:
        y_ticks_range = np.arange(0,1,0.15)    

    plt.plot(fpr,tpr,color=plot_color)
    #opt_idx = np.argmax(tpr-fpr)
    #optimal_thr = round(thr[opt_idx],2)
    opt_idx,optimal_thr = get_optimal_roc_auc_thr(fpr=fpr,tpr=tpr,thr=thr,round_off=round_off)
    fpr_opt = fpr[opt_idx]
    tpr_opt = tpr[opt_idx]
    plt.xticks(x_ticks_range,fontsize=font_size_)
    plt.yticks(y_ticks_range,fontsize=font_size_)
    #plt.axvline(fpr_opt,linestyle='--',color=optimal_thr_color,linewidth=optimal_thr_line_width)
    #plt.axhline(tpr_opt,linestyle='--',color=optimal_thr_color,linewidth=optimal_thr_line_width)
    plt.scatter(fpr_opt,tpr_opt,color="red")
    plt.xlabel('False Positive Rate',fontsize=font_size_)
    plt.ylabel('True Positive Rate',fontsize=font_size_)
    plt.title(f"""{title_} optimal thr = {optimal_thr}""",fontsize=font_size_)
    plt.grid(True,color="grey",linewidth=grid_line_width);


def save_df_as_csv(df,save_path):
    df.to_csv(save_path)
    print(f"file saved @ {save_path}")
    
def get_roc_curve_details(y,y_pred_proba):
    
    y_pred_proba = y_pred_proba if y_pred_proba.ndim==1 else y_pred_proba[:,1]
    fpr_, tpr_, thr_ = roc_curve(y, y_pred_proba)
    return fpr_, tpr_, thr_   

def get_auc_roc_score(y,y_pred_proba):
    y_pred_proba = y_pred_proba if y_pred_proba.ndim==1 else y_pred_proba[:,1]
    return roc_auc_score(y,y_pred_proba)

# perf metrics
def get_perfomance_metrics(df,actual_clas_col,pred_class_col,
                            true_class=True,false_class=False):
    
    res = df.groupby([pred_class_col,actual_clas_col])[actual_clas_col].agg('count')
    
    tp = res.get((true_class, true_class),0)
    tn = res.get((false_class, false_class),0)
    fp = res.get((true_class, false_class),0)
    fn = res.get((false_class, true_class),0)
    tpr = tp/(tp+fn) if tp+fn!=0 else 0
    fpr = fp/(fp+tn) if fp+tn!=0 else 0
    fnr = fn/(fn+tp) if fn+tp!=0 else 0
    tnr = tn/(tn+fp) if tn+fp!=0 else 0
    
    return {
        "tp":round(tp,2),
        "fp":round(fp,2),
        "fn":round(fn,2),
        "tn":round(tn,2),
        "tpr":round(tpr,2),
        "tnr":round(tnr,2),
        "fpr":round(fpr,2),
        "fnr":round(fnr,2)}

def get_clf_feature_importance(clf):
    if not hasattr(clf,"feature_importances_"):
        raise Exception("Model doesn't contain feature importance")
    df_feat_imp = pd.DataFrame()
    df_feat_imp["feature_name"] = clf.feature_names_in_
    df_feat_imp["feature_importance"] = clf.feature_importances_
    df_feat_imp.sort_values(by="feature_importance",ascending=False,inplace=True)
    return df_feat_imp

def get_file_extension(file_name):
    return file_name.split(".")[-1]

def collect_files_from_folder(folder_path):
    df = pd.DataFrame()
    print("Reading files from folder :")
    for file_name in track(os.listdir(folder_path)):
        file_ext = get_file_extension(file_name)
        if file_ext=="csv":
            print(f"collecting data from {file_ext} file : {file_name}")
            dfi = pd.read_csv(os.path.join(folder_path,file_name)) 
        elif file_ext=="json":
            print(f"collecting data from {file_ext} file : {file_name}")
            dfi = pd.read_json(os.path.join(folder_path,file_name))
        else:
            pass
        df = pd.concat([df,dfi])
    return df

def save_as_pickle_file(data,file_save_path):
    pickle.dump(data, open(file_save_path, 'wb'))
    print(f"Data saved @ {file_save_path}")
    
def load_pickle_file(file_path):
    pickle_file = pickle.load(open(file_path, 'rb'))
    return pickle_file

def getTimestampFromEpoch(epoch):
    return datetime.datetime.fromtimestamp(epoch/1000,tz)

def extractDayMonthYear(date):
    return pd.Series([date.day,date.month,date.year])

def get_quarter_from_timestamp(timestamp):
    return f"Q-{timestamp.quarter}"

def get_epoch_in_milli_sec(date_time_data):
    
    if pd.isna(date_time_data):
        return np.nan

    elif hasattr(date_time_data, "timestamp"):
        if date_time_data is not pd.NaT:
            return date_time_data.timestamp()*1000
        else:
            return np.nan

    elif isinstance(date_time_data,str):
        date_time_data = pd.to_datetime(str(date_time_data),infer_datetime_format=True,format='%d-%m-%Y',errors='coerce')
        if date_time_data is not pd.NaT:
            return date_time_data.timestamp()*1000
        else:
            date_time_data = pd.to_datetime(str(date_time_data).replace(" PM","").repalce(" AM",""),
                                            format='%b %d, %Y %H:%M:%S',errors='coerce')
            
            return date_time_data.timestamp()*1000 if not pd.isna(date_time_data) else np.nan
            
    else:
        return float(date_time_data)

    
# split criterias
def calculate_gini(value_counts_dict):
    
    gini=0
    total_num_samples = sum(value_counts_dict.values())
    for group in value_counts_dict.keys():
        prob = (value_counts_dict.get(group,0)/total_num_samples)
        gini+= prob*(1-prob)
    return gini

def get_gini_split_thr(df,feature_name,target,n_bins=20,top_n=1):
    
    split_crit_idx = np.argwhere(df[target].values[:1]!=df[target].values[1:]).flatten()
    split_crit = df[feature_name].values[split_crit_idx]
    split_crit = np.unique(split_crit)
    gini_scores=[]
    
    for crit in split_crit:
        
        left = df[[target]].loc[df[feature_name]<crit]
        right = df[[target]].loc[df[feature_name]>=crit]
        left_counter = dict(left[target].value_counts())
        right_counter = dict(right[target].value_counts())
        left_gini = calculate_gini(left_counter)
        right_gini = calculate_gini(right_counter)
        w_l = sum(left_counter.values())
        w_r = sum(right_counter.values())
        w_left = w_l/(w_l+w_r)
        w_right = w_r/(w_l+w_r)
        gini_scores.append(left_gini*w_left+right_gini*w_right)
    
    gini_scores = np.array(gini_scores)
    sorted_idxs = np.flipud(np.argsort(gini_scores))
    split_crit = split_crit[sorted_idxs]
    return split_crit[:top_n]
    
    
def get_scaled_data(df,feature_columns):
    df_scaled = pd.DataFrame()
    scaler = MinMaxScaler()
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df_scaled,scaler

def getStratifiedSamples(df,stratify_by,frac_data=None,num_samples=None,random_state_=None):
    
    show_message(f"stratify data on {stratify_by}\n")
    if frac_data is not None:
        show_message(f"returning {frac_data*100}% of stratified samples from data")
        strat_df = df.groupby(stratify_by,group_keys=False).apply(lambda x: \
                                                            x.sample(frac=frac_data,\
                                                            random_state=random_state_))
        return strat_df
    else:
        show_message(f"returning {num_samples} samples from data\n")
        strat_df = df.groupby(stratify_by,group_keys=False).apply(lambda x: \
                                                                x.sample(n=num_samples,\
                                                                random_state=random_state_))
        return strat_df

def Perform_Data_Imputation(df,feature_columns,method='simple',strategy_='median',n_neighbors_=5,weights_='uniform'):
    
    nan_cols = [x for x in feature_columns if df[x].isnull().any()]
    print("NaN feature columns found: ",len(nan_cols))
    print("Total feature columns: ",len(feature_columns))
            
    if method=='simple':
        print(f"Performing data imputation with strategy : {strategy_}")
        if len(nan_cols)>0:
            imputer = SimpleImputer(missing_values = np.nan,strategy=strategy_)
            imputer = imputer.fit(df[nan_cols])
        return imputer.transform(df[nan_cols])
    
    else:
        print(f"Performing data imputation using KNN")
        if len(nan_cols)>0:
            imputer = KNNImputer(missing_values = np.nan,n_neighbors=n_neighbors_,weights=weights_)
            df_scaled = Get_Scaled_Data(df,feature_columns)
            imputer = imputer.fit(df_scaled[nan_cols])
        return imputer.transform(df_scaled[nan_cols])

    
def Perform_SMOTE_Sampling(X_train,y_train,sampling_strategy_='auto',k_neighbors_=5,random_state_=None):
    """
    Based on KNN. Random number is generated from (0,1] (Random oversampling). This is multiplied to the gap of a positive class point and
    its nearest neighbor (positive class) and a new data point is generated. 
    
    """
    print("Generating SMOTE Samples")
    X_train_sm,y_train_sm = SMOTE(sampling_strategy=sampling_strategy_,\
                                  k_neighbors=k_neighbors_,\
                                  random_state=random_state_).fit_resample(X_train,y_train)
    
    return X_train_sm,y_train_sm 


def Perform_ADASYN_Sampling(X_train,y_train,sampling_strategy_='auto',n_neighbors_=5,random_state_=None):
    """
    KNN and Density distribution based oversampling( samples that are difficult to learn).
    Total no. of synthetic samples to be generated, G = (N– – N+) x β. Here, β = (N+/ N–).
    This is done wrt all neighbours.
    """
    print("Generating ADASYN Samples")
    X_train_sm,y_train_sm = ADASYN(sampling_strategy=sampling_strategy_,\
                                  n_neighbors=n_neighbors_,\
                                  random_state=random_state_).fit_resample(X_train,y_train)
    
    return X_train_sm,y_train_sm 


def get_stratified_sample(df,stratify_by,frac_data=None,num_samples=None,random_state_=None):
    
    if frac_data is not None:
        print(f"returning {frac_data*100}% of stratified data on {stratify_by}")
        strat_df = df.groupby(stratify_by,group_keys=False).apply(lambda x: \
                                                            x.sample(frac=frac_data,\
                                                            random_state=random_state_))

        return strat_df
    else:
        strat_df = df.groupby(stratify_by,group_keys=False).apply(lambda x: \
                                                                x.sample(n=num_samples,\
                                                                random_state=random_state_))
        return strat_df
    
def Filter_Columns_Threshold_NaN(df,feature_cols,threshold_nan=0.1):
    
    print(f"Filtering columns that have atleast {round(threshold_nan*100)}% non NA values")
    if feature_cols is None:
        return df.dropna(thresh=round(threshold_nan*len(df)),axis=1).columns
    else:
        return df[feature_cols].dropna(thresh=round(threshold_nan*len(df)),axis=1).columns.tolist()

    
def Filter_Columns_Threshold_Variance(df,feature_cols,cut_off_variance=0.005):
    
    print(f"Filtering columns having variance>= {cut_off_variance}")
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[feature_cols])
    df_scaled = pd.DataFrame(df_scaled,columns=feature_cols)
    
    variance_threshold = VarianceThreshold(threshold=cut_off_variance)  
    variance_threshold.fit(df_scaled[feature_cols])
    return df_scaled[feature_cols].columns[variance_threshold.get_support()].tolist()


def Filter_Columns_Mutual_Info(X,Y,feature_cols,mutual_info_threshold=0):
    
    print(f"Filtering features with mutual info > {mutual_info_threshold}")
    mutual_info = mutual_info_classif(X,Y)
    return np.array(feature_cols)[np.where(mutual_info>mutual_info_threshold)].tolist()


def get_confusion_matrix(y_test,y_pred,cmap_="Pastel2",title_=None,
                         label_font_size=14,title_font_size=10):
    
    # Pastel2,Pastel1,gnuplot2,Set3_r -- best
    sns.heatmap(pd.crosstab(y_test,y_pred),annot=True, cmap=cmap_ ,\
            fmt='d',linecolor='white',\
            cbar=False,linewidths=3)
    plt.xlabel("Predicted",fontsize=label_font_size)
    plt.ylabel("Actual",fontsize=label_font_size)
    plt.xticks(rotation=45,fontsize=label_font_size)
    plt.yticks(rotation=45,fontsize=label_font_size)
    title_ = "Confusion Matrix" if title_ is None else title_
    plt.title(title_,fontsize=title_font_size);


def get_confusion_matrix(y_test,y_pred,cmap_="Pastel2",title_=None,
                         label_font_size=14,title_font_size=10,x_label_="predicted",y_label_="actual"):
    
    # Pastel2,Pastel1,gnuplot2,Set3_r -- best
    sns.heatmap(pd.crosstab(y_test,y_pred),annot=True, cmap=cmap_ ,\
            fmt='d',linecolor='white',\
            cbar=False,linewidths=3)
    plt.xlabel(x_label_,fontsize=label_font_size)
    plt.ylabel(y_label_,fontsize=label_font_size)
    plt.xticks(rotation=45,fontsize=label_font_size)
    plt.yticks(rotation=45,fontsize=label_font_size)
    title_ = "Confusion Matrix" if title_ is None else title_
    plt.title(title_,fontsize=title_font_size);
    
    
def Get_TP_FP_FN_TN_details(df,actual_clas_col,pred_class_col):
    """
    Return TP,TN,FP,FN,TPR,FPR
    """
    
    res = df.groupby([pred_class_col,actual_clas_col])[actual_clas_col].agg('count')
    tp = res.get((True, True),0)
    tn = res.get((False, False),0)
    fp = res.get((True, False),0)
    fn = res.get((False, True),0)
    tpr = tp/(tp+fn) if tp+fn!=0 else 0
    fpr = fp/(fp+tn) if fp+tn!=0 else 0
    
    return round(tp,2),round(tn,2),round(fp,2),round(fn,2),round(tpr,2),round(fpr,2)    


def perform_hyperparameter_tuning(model,model_params,cv_type,scoring_type,X_train,y_train):
    
    model_cv = RandomizedSearchCV(model, param_distributions=model_params, \
                                  cv=cv_type, scoring=scoring_type,random_state=42,n_iter=25)
    model_cv.fit(X_train,y_train)
    return model_cv.best_params_,model_cv.best_score_


# def calculateGini(value_counts_dict):
    
#     gini=0
#     total_num_samples = sum(value_counts_dict.values())
#     for group in value_counts_dict.keys():
#         prob = (value_counts_dict.get(group,0)/total_num_samples)
#         gini+= prob*(1-prob)
#     return gini


# def Get_Split_Criteria_Gini(df,feature_name,target,n_bins=20):
    
#     n_uniq = len(df[feature_name].unique())
#     n_bins = min([n_bins,n_uniq])
#     min_val = df[feature_name].min()
#     max_val = df[feature_name].max()
#     step_size = round((max_val-min_val)/n_bins)
    
#     split_crit = np.arange(min_val,max_val,step_size)
#     gini_scores=[]
    
#     for crit in split_crit:
        
#         left = df[[target]].loc[df[feature_name]<crit]
#         right = df[[target]].loc[df[feature_name]>=crit]
#         left_counter = dict(left[target].value_counts())
#         right_counter = dict(right[target].value_counts())
#         left_gini = Calculate_Gini(left_counter)
#         right_gini = Calculate_Gini(right_counter)
#         w_l = sum(left_counter.values())
#         w_r = sum(right_counter.values())
#         w_left = w_l/(w_l+w_r)
#         w_right = w_r/(w_l+w_r)
#         gini_scores.append(left_gini*w_left+right_gini*w_right)
    
#     return split_crit[np.argmin(np.array(gini_scores))]


# --- STATISTICAL TESTS ------- #

def shapiroWilkNormalityTest(df,feature_name):
    """checks if the data is normally distributed 

    Args:
        df (dataframe): data
        feature_name (string): name of the feature/column

    Returns:
        float: stat and p_value
    """
    stat,p_value = shapiro(df[feature_name])
    return stat,p_value


def perform_Mann_Whitney_Utest(df,feature_name,target,label_major,label_minor,num_samples=50):
    
    """
    Checks if there is a significant difference in the feature median values of the two INDEPENDENT groups.
    """
    p_value_list=[]
    stat_val_list = []
    for i in range(num_samples):
        df_minor = df[[feature_name]].loc[df[target]==label_minor]
        df_major = df[[feature_name]].loc[df[target]==label_major].sample(len(df_minor))
        stat_, p_value = mannwhitneyu(df_major[feature_name].tolist(),df_minor[feature_name].tolist())
        p_value_list.append(p_value)
        stat_val_list.append(stat_)
    
    return np.mean(stat_val_list),np.mean(p_value_list)


def perform_Wilcoxon_Test(df,feature_name,target,label_major,label_minor,num_samples=50):
    
    """
    Checks if there is a significant difference in the feature mean ranks values of the two DEPENDENT groups.
    """
    
    p_value_list=[]
    for i in range(num_samples):
        df_minor = df[[feature_name]].loc[df[target]==label_minor]
        df_major = df[[feature_name]].loc[df[target]==label_major].sample(len(df_minor))
        stat, p_value = wilcoxon(df_major[feature_name].tolist(),df_minor[feature_name].tolist())
        if not np.isnan(p_value):
            p_value_list.append(p_value)
    
    return np.mean(p_value_list)


def perform_One_Way_ANOVA(df,group_name,feature_name,random_state_=None):
    show_message("""
    One-way ANOVA -whether there exists a statistically significant difference between the 
    mean values of more than one group.
    Dependent variable should be continuous.
    """)
    res = dict(df[group_name].value_counts())
    dfs = get_stratified_sample(df,stratify_by=group_name,frac_data=None,\
                                num_samples=min(res.values()),\
                                random_state_=random_state_)
    sample_groups = []
    for group in res:
        sample_groups.append(dfs[feature_name].loc[dfs[group_name]==group].tolist())
    
    stat_value,p_value = f_oneway(*sample_groups)
    return p_value


def perform_Independent_Ttest(df,binary_group,feature_name,random_state_=None,
                            equal_var_=True,stratified=False,sig=0.05):
    """
    data_group1: First data group
    data_group2: Second data group
    equal_var = “True”: The standard independent two sample t-test will be
    conducted by taking into consideration the equal population variances.
    equal_var = “False”: The Welch’s t-test will be conducted by not taking 
    into consideration the equal population variances.
    If the ratio of the larger data groups to the small data group is less than 4:1 
    then we can consider that the given data groups have equal variance.
    
    """
    print("data groups",binary_group)
    print("analyzing distributions of ",feature_name)
    res = dict(df[binary_group].value_counts())
    if stratified:
        show_message(f"perf t-test on stratified samples")
        df_sampled = get_stratified_sample(df,stratify_by=binary_group,frac_data=None,\
                                num_samples=min(res.values()),\
                                random_state_=random_state_)
    else:
        show_message("perf t-test using entire data")
        df_sampled = df
    sample_groups = []
    for group in res:
        sample_groups.append(df_sampled[feature_name].loc[df_sampled[binary_group]==group].tolist())
    
    stat_value,p_value = ttest_ind(a=sample_groups[0],b=sample_groups[1],equal_var=equal_var_)
    if p_value<=sig:
        show_message(f"""there is significant difference in the two distributions, 
                    reject null hyp (H0)""")
        
    else:
        show_message(f"""there is NO significant difference in the two distributions, 
                    FAILED to reject null hyp (H0)""")
            
    t_test_res = {"stat":stat_value,"p_value":p_value}
    show_message("__________________________________________________")
    return t_test_res


def get_Kendal_Tau_Correlation(df,ordinal_feature,target,label_major,label_minor,num_samples=50):
    
    KendalTau=[]
    
    for i in range(num_samples):
        
        df_minor = df[[ordinal_feature,target]].loc[df[target]==label_minor]
        df_major = df[[ordinal_feature,target]].loc[df[target]==label_major].sample(len(df_minor))
        df_temp = pd.concat([df_major,df_minor])
        kendal_tau_i, p = Kendal_Tau_Correlation(df_temp[ordinal_feature].values, df_temp[target].values)
        if not np.isnan(kendal_tau_i):
            KendalTau.append(kendal_tau_i)

    return np.mean(KendalTau)


def get_Point_Biserial_Correlation(df,continuous_feature,target,label_major,label_minor,num_samples=50):
    
    point_biserial_ls = []
    
    for i in range(num_samples):
        
        df_minor = df[[continuous_feature,target]].loc[df[target]==label_minor]
        df_major = df[[continuous_feature,target]].loc[df[target]==label_major].sample(len(df_minor))
        df_temp = pd.concat([df_major,df_minor])
        pb_corr = Point_Biserial_Correlation(df_temp[continuous_feature],df_temp[target])[0]
        if not np.isnan(pb_corr):
            point_biserial_ls.append(pb_corr)
        
    return np.mean(point_biserial_ls)


def perform_ChiSquare_Test(df,feature_name,target,label_major,label_minor,num_samples=50):
    
    p_temp=[]
    for i in range(num_samples):
        
        df_minor = df[[feature_name,target]].loc[df[target]==label_minor]
        df_major = df[[feature_name,target]].loc[df[target]==label_major].sample(len(df_minor))
        df_temp = pd.concat([df_major,df_minor])
        
        p_value = chi2(df_temp[[feature_name]],df_temp[target])[1][0]
        if not np.isnan(p_value):
            p_temp.append(p_value)

    return np.mean(p_temp)


def cramers_V_Correlation(x, y):
    
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    if phi2corr==0:
        cramers_v_corr=0
    else:
        cramers_v_corr = np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
    return cramers_v_corr


def get_Cramers_Correlation(df,categorical_feature,target,label_major,label_minor,num_samples=50):
    
    Cramers=[]

    for i in range(num_samples):
        
        df_minor = df[[categorical_feature,target]].loc[df[target]==label_minor]
        df_major = df[[categorical_feature,target]].loc[df[target]==label_major].sample(len(df_minor))
        df_temp = pd.concat([df_major,df_minor])
        
        Cramers_i = Cramers_V_Correlation( df_temp[categorical_feature],df[target])
        if not np.isnan(Cramers_i):
            Cramers.append(Cramers_i)

    return np.mean(Cramers)

def Goldfeld_Quandt_heterosced_test(resids,X,conf=0.05):

    gq_stat,p_val,f_ind = het_goldfeldquandt(resids,X)
    results = {"gq_stat":gq_stat,"p_val":p_val,
               "f_ind":f_ind, "heteroscedascity_exists":True if p_val<conf else False}
    return results

def Box_Tidwell_test(data,feature_cols,target_col,model_params=None):
    
    X = data[feature_cols]
    y = data[target_col]
    if model_params is None:
        model = LogisticRegression(class_weight="balanced",solver="liblinear",random_state=42)
    else:
        model = LogisticRegression(**model_params,random_state=42)
    model.fit(X,y)
    coef = model.coef_[0]

    x_transf = np.log(X)
    x_transf = x_transf.replace([-np.inf,np.inf,np.nan],0)
    for i in range(x_transf.shape[1]):
        x_transf.iloc[:,i] *= X.values[:,i]

    if model_params is None:
        model_trnsf = LogisticRegression(class_weight="balanced",solver="liblinear",random_state=42)
    else:
        model_trnsf = LogisticRegression(**model_params,random_state=42)
    model_trnsf.fit(x_transf,y)    

    coef = model.coef_[0]
    coef_trnf = model_trnsf.coef_[0]
    bt_stats = coef_trnf/coef

    df_bt_stat = pd.DataFrame()
    df_bt_stat["feature_name"] = X.columns
    df_bt_stat["Box_Tidwell"] = bt_stats
    return df_bt_stat       

# misc

def generate_requirements_file():
    
    import session_info
    res = session_info.show()
    res = res.data
    res = res.split("-----")[1]
    res_final = ""
    for i,j in zip(res.split()[1::2],res.split()[0::2]):
        if not pd.isna(j):
            req_str_i = j+"=="+i
            res_final+=req_str_i
            res_final+="\n"

    print(res_final) 

# Quick data read


class readFilesFromDirectory:
    
    def __init__(self,filePath=None,fileType=None):
        self.filePath = filePath
        self.fileType = fileType
        if (self.filePath is not None) and (self.fileType is not None):
            self.fileNames = os.listdir(filePath)
        else:
            self.fileNames = "No Files Found"
            
    def collectData(self):          
        
        df = pd.DataFrame()
        if self.fileNames != "No Files Found":
            for fileName in self.fileNames:
                if "."+self.fileType in fileName:
                    print(f"Reading --> {fileName}")
                    df_temp = pd.read_csv(self.filePath+fileName)
                    df = pd.concat([df,df_temp])
                    print("Completed")
        return df            
    
class quick_data_check:
    
    def __init__(self,data,return_col_names=False,return_trf_col_names=False):
        self.data = data
        self.return_col_names = return_col_names
        self.return_trf_col_names = return_trf_col_names
        
    
    def execute(self):
        
        show_message(f"number of rows and columns {self.data.shape}")
        
        
        if isinstance(self.data,pd.DataFrame):
            self.nan_cols = [x for x in self.data.columns if self.data[x].isnull().any()]
            self.all_nan_cols = [x for x in self.nan_cols if self.data[x].isnull().all()]
            if len(self.nan_cols)>0:
                show_message(f"WARNING - Found {len(self.nan_cols)} column(s) with null values")
            else:
                show_message(f"Found {len(self.nan_cols)} column(s) with null values")
            if len(self.all_nan_cols)>0:
                show_message(f"WARNING - Found {len(self.all_nan_cols)} column(s) with all null values")
            else:
                show_message(f"Found {len(self.all_nan_cols)} column(s) with all null values")
            self.check_str_2_numeric_transf()
            show_message("===================================================")
            if self.return_col_names:
                return self.nan_cols,self.all_nan_cols    
            
    def check_str_2_numeric_transf(self):
        
        if isinstance(self.data,pd.DataFrame):
        
            self.cols = [x for x in self.data.columns if self.data[x].dtype=='O']
            self.col_transform_str_float,self.col_all_str = [],[]

            for col in self.cols:
                try:
                    self.data[col].astype(float)
                    self.col_transform_str_float.append(col)
                except:
                    self.col_all_str.append(col)
            if len(self.col_transform_str_float)>0:    
                show_message(f"""WARNING - Found {len(self.col_transform_str_float)} column(s) whose data type is object but can be converted to float""")
            else:
                show_message(f"""Found {len(self.col_transform_str_float)} column(s) whose data type is object but can be converted to float""")
            show_message(f"""Found {len(self.col_all_str)} column(s) which are all string type""")
            
            if self.return_trf_col_names:
                    return self.col_transform_str_float,self.col_all_str   

show_message("😃😃😃😃😃😃😃😃😃😃😃")                
show_message("datascience libraries loaded")
show_message("current working directory set to : "+os.getcwd())
show_message("🤖🐼🐍🐉🐧🐞")