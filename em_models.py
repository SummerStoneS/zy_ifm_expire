"""
@time: 10/25/2018 2:56 PM

@author: 柚子
"""
import os
import pickle
import pandas as pd
import lightgbm as lgb
from variables import params, alpha_dict
from preprocess import folder
from y_define import tag_y
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from collections import defaultdict


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


def merge_data(time_periods: list):
    """
    :param time_periods: list combine这些period数据，用于训练
    :return: 训练数据
    """
    columns = params["features_before_y_define"]
    combine_data = pd.DataFrame()
    for period in time_periods:
        period_data = pd.read_hdf(os.path.join(folder, "data", period, "new_data.h5"), key="preRF")
        period_data = period_data[columns]
        combine_data = pd.concat([combine_data, period_data])
    return combine_data


def create_folder(train_periods, oot_period):
    if oot_period:
        model_folder = "train" + "_".join(train_periods) + "oot" + "_".join(oot_period)
    else:
        model_folder = "train" + "_".join(train_periods) + "test"
    save_folder = os.path.join(folder, "model_result", model_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    return save_folder


def get_train_and_y(train_periods, save_folder, key):
    """
    :param train_periods:
    :param save_folder:
    :param key:
    :return: 合并数据，与预测变量有关的列：y1, y2, 和六个one_hot_column e.g. y1_up_y2_1   y1: up,even,down, y2: 1,0
    """
    train_no_y = merge_data(train_periods)
    train_with_ys = tag_y(train_no_y, products_contbuy="all")
    train_with_ys.to_hdf(os.path.join(save_folder, key + "_data.h5"), key=key + "_all_products")
    return train_with_ys


def lightgbm(y_col, tpr_thres):
    """
    :param y_col: 6个预测的y类别，分别建模
    :param tpr_thres: 预先设定的切分阈值
    :return:
    """
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test = lgb.Dataset(X_test, label=y_test)
    # # class_weight = {0: 1, 1: 0, 2: 1, 3: 1, 4: 1}
    lgb_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': 'true',
        'metric': 'auc',
        'learning_rate': 0.008,
        "n_estimators": 800,
        'max_depth': 4,
        "num_leaves": 10
    }
    clf = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_test])
    joblib.dump(clf, os.path.join(saveFolder, "{}_30lgb.pkl".format(y_col)))
    y_pred = clf.predict(X_test)
    model_result[y_col]["roc_auc_score"] = roc_auc_score(y_test, y_pred)
    fpr, tpr, alpha = roc_curve(y_test, y_pred, pos_label=1)
    threshold = alpha[tpr > tpr_thres][0]
    y_pred_01 = pd.Series(y_pred >= threshold, index=y_test.index, name="pred_01_{}".format(y_col)).astype(int)
    confusion_mat = confusion_matrix(y_test, y_pred_01)
    confusionMatrix = pd.DataFrame(confusion_mat, columns=[0, 1], index=[0, 1]). \
        sort_index(axis=0, ascending=False).sort_index(axis=1, ascending=False)
    confusionMatrix["tpr"] = confusionMatrix[1]/confusionMatrix.sum(axis=1)
    confusionMatrix["fpr"] = confusionMatrix[0]/confusionMatrix.sum(axis=1)
    model_result[y_col]["confusion_matrix"] = confusionMatrix
    print(confusionMatrix)
    print("precision:{}".format(sum(y_pred_01 == y_test)/len(y_test)))
    print("threshold:{}".format(threshold))
    model_result[y_col]["precision"] = sum(y_pred_01 == y_test)/len(y_test)
    model_result[y_col]["threshold"] = threshold
    feature_importance = clf.feature_importance()
    model_result[y_col]["importance"] = pd.Series(feature_importance, index=X_train.columns, name="feature_importance")
    return y_pred_01, pd.Series(y_pred, index=y_test.index, name="pred_{}".format(y_col))


if __name__ == '__main__':
    train_periods = ["20180907", "20180914", "20180815", "20180719", "20180608", "20180507", "20180405"]
    saveFolder = create_folder(train_periods, oot_period=None)

    train_data = get_train_and_y(train_periods, saveFolder, "train")
    X_train_all = train_data[params["all_features"]]
    X_train_all.fillna(-2, inplace=True)
    result_data = pd.DataFrame()                        # 每个类别预测的概率结果，和0-1结果
    model_result = defaultdict(dict)                    # record 6 independent models' results
    for aum_change in ["up", "even", "down"]:
        for cont_buy in [1, 0]:
            predict_column = "y1_{}_y2_{}".format(aum_change, cont_buy)
            y = train_data[predict_column]
            X_train, X_test, y_train, y_test = train_test_split(X_train_all, y, test_size=0.3, random_state=34)
            y_pred01, y_pred_prob = lightgbm(predict_column, tpr_thres=alpha_dict[predict_column])
            result_data = pd.concat([result_data, y_pred01, y_pred_prob], axis=1)
    result_data.to_hdf(os.path.join(saveFolder, "result_data.h5"), key="merge_30lgb")

    with open(os.path.join(saveFolder, "model6_result_report.pk"), "wb") as f:
        pickle.dump(model_result, f)

    # with open(os.path.join(saveFolder, "model6_result_report.pk"), "rb") as f:
    #     model_result = pickle.load(f)
    writer = pd.ExcelWriter(os.path.join(saveFolder, "model6_result_report.xlsx"))
    for key in model_result.keys():
        single_model = model_result[key]
        score_list = []
        for key2 in ["roc_auc_score", "threshold", "precision"]:
            score_list.append(single_model[key2])
        score_list = pd.DataFrame([score_list], columns=["roc_auc_score", "threshold", "precision"])
        score_list.to_excel(writer, sheet_name=key, startrow=0)
        confu_ma = pd.DataFrame(single_model["confusion_matrix"])
        confu_ma.to_excel(writer, sheet_name=key, startrow=3)
        top_features = single_model["importance"].sort_values(ascending=False)[:10]
        importance = single_model["importance"].sort_values(ascending=False)             # 变量重要性
        importance.to_excel(writer, sheet_name=key, startcol=6, startrow=0)
        correlation = train_data[[key] + list(top_features.index)].corr().iloc[:, 0]      # top10重要变量相关性矩阵
        correlation.to_excel(writer, sheet_name=key, startcol=9, startrow=0)
    writer.close()






