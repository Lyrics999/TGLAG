import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

def CumMSE(df_model:pd.DataFrame):
    cum_mse = []
    for i in range(len(df_model)):
        if i < 1:
            cum_mse.append(mean_squared_error(df_model.iloc[:i+1,0], df_model.iloc[:i+1,1]))
        else:
            cum_mse.append(mean_squared_error(df_model.iloc[i-1:i+1,0], df_model.iloc[i-1:i+1,1]))
    return cum_mse

def CombineWeight(cum_mse1:list,cum_mse2:list):
    weightList = []
    for i in range(len(cum_mse1)):
        weightOfModel1 = cum_mse2[i]/sum([cum_mse1[i],cum_mse2[i]])
        weightList.append(weightOfModel1)
    return weightList

def CombineModel(df_Model1:pd.DataFrame,df_Model2:pd.DataFrame,combineWeightOfModel1:list):
    y_pre = []
    for idx,weight in enumerate(combineWeightOfModel1):
        prediction = df_Model1['pre'][idx] *weight + df_Model2['pre'][idx]*(1-weight)
        y_pre.append(prediction)
    return y_pre

y_predGCN=pd.read_excel('.luong.xlsx')
y_predGRU=pd.read_excel('.gru.xlsx')
y_test=pd.read_excel('.true.xlsx')
df_GCN = pd.DataFrame(y_predGCN)
df_GRU = pd.DataFrame(y_predGRU)
df_true= pd.DataFrame(y_test)
df_combined1 = pd.DataFrame()
df_combined2 = pd.DataFrame()
mae_list = []
rmse_list = []
weighted_predictions_list = []

for i in range(15):
    df_combined1['pre'] = df_GCN[i]  
    df_combined2['pre'] = df_GRU[i] 
    df_combined2['true'] = df_true[i]  
    df_combined1['true'] = df_true[i]
    
    cum_mse_GCN = CumMSE(df_combined1)
    cum_mse_GRU = CumMSE(df_combined2)
    weight_GCN = CombineWeight(cum_mse_GCN,cum_mse_GRU)
    y_pre = CombineModel(df_combined1,df_combined2,weight_GCN)
    y = df_true[i]
    mae = mean_absolute_error(y_true=y, y_pred=y_pre)
    rmse = np.sqrt(mean_squared_error(y_true=y, y_pred=y_pre))
    
    mae_list.append(mae)
    rmse_list.append(rmse)
    
for i in range(15):
    print(f"Iteration {i + 1} - MAE: {mae_list[i]}, RMSE: {rmse_list[i]}")
