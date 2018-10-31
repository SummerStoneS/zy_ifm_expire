"""
@time: 9/13/2018 11:53 AM

@author: 柚子
"""
import os
import pandas as pd
from sklearn.externals import joblib
from preprocess import folder
from variables import params


def get_data(period):
    """
    :param period: 预测的时间
    :return: 预测数据X
    """
    columns = params["all_features"]
    period_data = pd.read_hdf(os.path.join(folder, "data", period, "new_data.h5"), key="preRF")
    period_data = period_data[columns]
    period_data.fillna(-2, inplace=True)
    return period_data


def predict_six_probability(x_predict):
    """
    :param x_predict: 预测数据的x
    :return: 6列六个模型的预测概率
    """
    predict_prob_matrix = pd.DataFrame()
    for aum_change in ["up", "even", "down"]:
        for cont_buy in [1, 0]:
            predict_column = "y1_{}_y2_{}".format(aum_change, cont_buy)
            clf = joblib.load(os.path.join(model_folder, "{}_30lgb.pkl".format(predict_column)))
            y_single_prob = clf.predict(x_predict)
            y_single_prob = pd.Series(y_single_prob, name="prob_" + predict_column, index=x_predict.index)
            predict_prob_matrix = pd.concat([predict_prob_matrix, y_single_prob], axis=1)
    return predict_prob_matrix


def predict_multiclass(six_probs_matrix):
    """
    :param six_probs_matrix: 预测人群的六类人预测概率
    :return: 最终分类结果
    """
    clf = joblib.load(os.path.join(model_folder, "probability_softmax.pkl"))
    y_pred = clf.predict(six_probs_matrix)
    # y_pred = [list(x).index(max(x)) for x in y_pred]
    return pd.Series(y_pred, index=six_probs_matrix.index, name="pred")

if __name__ == '__main__':
    train_periods = ["20180907", "20180914", "20180815", "20180719", "20180608", "20180507", "20180405"]
    predict_period = "20181029"
    model_folder = os.path.join(folder, "model_result", "train" + "_".join(train_periods) + "test")
    predict_folder = os.path.join(model_folder, "predict_" + predict_period)  # 预测结果保存地址
    if not os.path.exists(predict_folder):
        os.makedirs(predict_folder)
    predict_data = get_data(predict_period)
    predict_multi_probs = predict_six_probability(predict_data)
    prediction = predict_multiclass(predict_multi_probs)
    result_data = pd.concat([predict_data, prediction], axis=1)
    result_data.to_excel(os.path.join(predict_folder, "predict_result_{}.xlsx".format(predict_period)))









