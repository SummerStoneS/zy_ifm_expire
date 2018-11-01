"""
@time: 9/13/2018 11:53 AM

@author: 柚子
"""
import os
import numpy as np
import pandas as pd
from modeling import get_train_data, split_x_y, generate_performance_resultset
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from preprocess import folder


train_periods = ["20180630", "20180331", "20171231", "20170930"]
predict_period = "20180930"
model_folder = "train" + "_".join(train_periods) + "predict_" + predict_period
save_folder = os.path.join(folder, "model_result", model_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

data_train = get_train_data(train_periods)
X_train, y_train = split_x_y(data_train, y_col="is_churn")
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=34)
gbr = GradientBoostingRegressor(loss='ls', max_depth=8, learning_rate=0.04, n_estimators=150, min_samples_split=10,
                                min_samples_leaf=10)
gbr.fit(np.array(X_train), np.array(y_train))
joblib.dump(gbr, os.path.join(save_folder, "gbdt_model.pkl"))

feature_columns = X_train.columns
X_predict = pd.read_hdf(os.path.join(folder, "data", predict_period, "new_data.h5"), key="preRF")[feature_columns]
X_predict.fillna(-2, inplace=True)
y_pred = gbr.predict(np.array(X_predict))
y_pred = pd.Series(y_pred, index=X_predict.index, name='pred_prob')
y_pred_group = pd.Series(pd.qcut(y_pred, 10, labels=range(10)), name='pred_group')
y_pred_result = pd.concat([y_pred, y_pred_group], axis=1)
# y_pred_result.to_excel(os.path.join(save_folder, "prediction_of_total.xlsx"))
dataX_y = pd.merge(y_pred_result, X_predict, how="left", left_index=True, right_index=True)
dataX_y.to_csv(os.path.join(save_folder, "prediction_result_data.csv"))


test_pred = gbr.predict(np.array(X_test))
print("roc_auc_score:")
auc = roc_auc_score(y_test, test_pred)
print(auc)
generate_performance_resultset(y_test, test_pred, save_folder, "test")

feature_importance = pd.Series(gbr.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importance.to_excel(os.path.join(save_folder, "feature_importance.xlsx"))



