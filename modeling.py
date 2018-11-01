"""
@time: 8/16/2018 3:27 PM

@author: 柚子
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor
from variables import params
from preprocess import folder


def split_x_y(initial_data, y_col="is_churn"):
    y = initial_data[y_col]
    X = initial_data.copy()
    X.fillna(-2, inplace=True)
    del X[y_col]
    return X, y


def plot_roc(y_test, y_pred, model_name):
    fpr, tpr, alpha = roc_curve(y_test, y_pred, pos_label=1)
    img_name = "ROC of " + model_name
    plt.plot(fpr, tpr, linewidth=2, label=img_name)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.legend(loc=4)
    plt.savefig(model_name)
    plt.show()
    return fpr, tpr, alpha


def train_model_use_decisiontreeregressor(data_x_y, save_path):
    X, y = split_x_y(data_x_y, y_col="is_churn")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    model2 = DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=5, min_samples_leaf=5, presort=True)
    model2.fit(np.array(X_train), np.array(y_train))
    feature_importance = pd.Series(model2.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    save_feature_name = "dt_feature_importance" + datetime.now().strftime("%m%d%H%M") + ".xlsx"
    feature_importance.to_excel(os.path.join(save_path, save_feature_name))
    save_model_name = 'DecisionTree' + datetime.now().strftime("%m%d%H%M") + '.pkl'
    joblib.dump(model2, os.path.join(save_path, save_model_name))
    y_pred = model2.predict(np.array(X_test))
    generate_performance_resultset(y_test, y_pred, save_path, "traindatatest")
    return model2


def generate_performance_resultset(y_test, y_pred, save_path, savename):
    """
    :param y_test:
    :param y_true:
    :param alpha: threshold for classifying as churn 1
    :return: confusion matrix，ROC， AUC
    """
    fpr, tpr, alpha = plot_roc(y_test, y_pred, os.path.join(save_path, savename + "ROC.png"))
    #TODO(decision tree 用下面一行)
    # threshold = alpha[-2:-1][0]
    threshold = alpha[tpr > 0.87][0]
    y_pred_01 = pd.Series(y_pred >= threshold, index=y_test.index, name="y_pred").astype(int)
    confusion_mat = confusion_matrix(y_test, y_pred_01)
    confusionMatrix = pd.DataFrame(confusion_mat, columns=[0, 1], index=[0, 1]). \
        sort_index(axis=0, ascending=False).sort_index(axis=1, ascending=False)
    confusionMatrix.to_excel(os.path.join(save_path, savename + "confusionMatrix.xlsx"))
    return confusionMatrix


def run_gbdt(X_train, X_test, y_train, y_test, save_folder):
    gbr = GradientBoostingRegressor(loss='ls', max_depth=10, learning_rate=0.04, n_estimators=150, min_samples_split=10,
                                    min_samples_leaf=10)
    gbr.fit(np.array(X_train), np.array(y_train))
    feature_importance = pd.Series(gbr.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    save_feature_name = "feature_importance" + datetime.now().strftime("%m%d%H%M") + ".xlsx"
    feature_importance.to_excel(os.path.join(save_folder, save_feature_name))
    y_pred = gbr.predict(np.array(X_test))
    print("roc_auc_score:")
    auc = roc_auc_score(y_test, y_pred)
    print(auc)
    joblib.dump(gbr, os.path.join(save_folder, "gbdt" + datetime.now().strftime("%m%d%H%M") + '.pkl'))
    return gbr


def get_train_data(time_periods: list):
    """
    :param time_periods: list combine这些period数据，用于训练
    :return: 训练数据
    """
    columns = params["downsale_all_columns"]
    combine_data = pd.DataFrame()
    for period in time_periods:
        period_data = pd.read_hdf(os.path.join(folder, "data", period, "new_data.h5"), key="preRF")
        period_data = period_data[columns]
        combine_data = pd.concat([combine_data, period_data])
    return combine_data


if __name__ == '__main__':

    train_periods = ["20180331", "20171231", "20170930"]
    test_period = ["20180630"]

    model_folder = "train" + "_".join(train_periods) + "oot" + "_".join(test_period)
    save_folder = os.path.join(folder, "model_result", model_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    data_train = get_train_data(train_periods)
    data_test = get_train_data(test_period)
    X_train, y_train = split_x_y(data_train, y_col="is_churn")
    X_test, y_test = split_x_y(data_test, y_col="is_churn")

    gbr = run_gbdt(X_train, X_test, y_train, y_test, save_folder)
    y_pred = gbr.predict(np.array(X_test))
    # 生成结果报告
    generate_performance_resultset(y_test, y_pred, save_folder, "oot_test")
    # 保存预测结果
    y_pred = pd.Series(y_pred, index=X_test.index, name='pred_prob')
    y_pred_group = pd.Series(pd.qcut(y_pred, 10, labels=range(10)), name='pred_group')
    y_pred_result = pd.concat([y_pred, y_pred_group], axis=1)
    dataX_y = pd.merge(y_pred_result, X_test, how="left", left_index=True, right_index=True)
    dataX_y.to_csv(os.path.join(save_folder, "prediction_result_data.csv"))





