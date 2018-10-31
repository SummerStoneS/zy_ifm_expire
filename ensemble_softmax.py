"""
@time: 10/29/2018 10:46 PM

@author: 柚子
"""
import os
import pandas as pd
import lightgbm as lgb
from variables import params
from preprocess import folder
from y_define import tag_multiclass, tag_y
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


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
    :return: 合并数据，y: 0,1,2,3,4,5六类； 六个one_hot_column y1_up_y2_1   y1: up,even,down, y2: 1,0
    """
    train_no_y = merge_data(train_periods)
    train_with_ys = tag_y(train_no_y)
    train_with_ys = tag_multiclass(train_with_ys)
    train_with_ys.to_hdf(os.path.join(save_folder, key + "_data.h5"), key=key + "_all_products_y")
    return train_with_ys


def gbm_predict_prob(y_col, x_train, x_valida, x_test=""):
    """
    :param y_col: 一次传入一个类别的名字
    :param x_train: 训练数据X
    :param x_valida: 验证数据X
    :param x_test: oot测试数据X
    :return: 训练数据和验证数据的某一列的预测概率
    """
    clf = joblib.load(os.path.join(saveFolder, "{}_30lgb.pkl".format(y_col)))
    y_train_prob = clf.predict(x_train)
    y_valida_prob = clf.predict(x_valida)

    train_cate_prob = pd.Series(y_train_prob, name="prob_" + y_col, index=x_train.index)
    validation_cate_prob = pd.Series(y_valida_prob, name="prob_" + y_col, index=x_valida.index)
    if isinstance(x_test, pd.DataFrame):
        y_test_prob = clf.predict(x_test)
        test_cate_prob = pd.Series(y_test_prob, name="prob_" + y_col, index=x_test.index)
    else:
        test_cate_prob = None
    return train_cate_prob, validation_cate_prob, test_cate_prob


def lightgbm_multi(X_trainn, X_testt, y_trainn, y_testt):
    """
    :return: 对6类人的模型预测概率建模，预测最终分类
    """
    class_weight = {0: 1.3, 1: 1.2, 2: 1.1, 3: 1, 4: 1, 5: 0.8}
    lgb_train = lgb.Dataset(X_trainn, label=y_trainn)
    lgb_test = lgb.Dataset(X_testt, label=y_testt)
    # # class_weight = {0: 1, 1: 0, 2: 1, 3: 1, 4: 1}
    lgb_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 6,
        'metric': 'multi_logloss',
        'learning_rate': 0.02,
        "n_estimators": 800,
        "max_depth": 8,
        "num_leaves": 30,
        "class_weight": class_weight
    }
    # clf = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_test])
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='multiclass',
        num_class=6,
        metric='multi_logloss',
        learning_rate=0.02,
        n_estimators=800,
        max_depth=10,
        num_leaves=45,
        class_weight=class_weight)
    clf.fit(X_trainn, y_trainn)
    joblib.dump(clf, os.path.join(saveFolder, "probability_softmax.pkl"))
    y_pred = clf.predict(X_testt)
    # y_pred = [list(x).index(max(x)) for x in y_pred]
    # print("validation:")
    print(classification_report(y_testt, y_pred))
    return clf


if __name__ == '__main__':
    # step 1: 读入训练数据
    train_periods = ["20180907", "20180914", "20180815", "20180719", "20180608", "20180507", "20180405"]
    saveFolder = create_folder(train_periods, oot_period=None)
    train_data = get_train_and_y(train_periods, saveFolder, "train")
    X_train_all = train_data[params["all_features"]]
    X_train_all.fillna(-2, inplace=True)

    # step 2: 区分30%验证集， 对6类的每一个类别先用模型预测出6个概率
    train_prob_matrix = pd.DataFrame()
    valida_prob_matrix = pd.DataFrame()
    for aum_change in ["up", "even", "down"]:
        for cont_buy in [1, 0]:
            predict_column = "y1_{}_y2_{}".format(aum_change, cont_buy)
            y = train_data[predict_column]
            X_train, X_test, y_train, y_test = train_test_split(X_train_all, y, test_size=0.3, random_state=34)
            train_prob, valid_prob, _ = gbm_predict_prob(predict_column, X_train, X_test, "")       # 训练集和验证集6类人的概率预测
            print("{} test auc: {}".format(predict_column, roc_auc_score(y_test, valid_prob)))
            train_prob_matrix = pd.concat([train_prob_matrix, train_prob], axis=1)      # 6列概率
            valida_prob_matrix = pd.concat([valida_prob_matrix, valid_prob], axis=1)

    # step 3: 训练集的6个概率作为x，softmax预测六类， 验证集输出report
    X_cols = train_prob_matrix.columns
    train_prob_matrix_y = pd.merge(train_prob_matrix, train_data[["y"]], how="left", right_index=True, left_index=True)
    valida_prob_matrix_y = pd.merge(train_prob_matrix, train_data[["y"]], how="left", right_index=True, left_index=True)
    prob_softmax_model = lightgbm_multi(train_prob_matrix_y[X_cols], valida_prob_matrix_y[X_cols],
                                        train_prob_matrix_y["y"], valida_prob_matrix_y["y"])








