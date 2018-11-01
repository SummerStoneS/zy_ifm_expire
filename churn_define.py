"""
@time: 8/22/2018 7:44 PM

@author: 柚子
"""
import numpy as np


def tag_churn(aum_data, old_month_num=3):
    """
    :return: 一个月相对之前n=3个月的平均AUM下降20%
    """
    past_aum_avg = ["aum_" + str(i) for i in range(old_month_num+1, 7)]
    aum_data["past_months_aum_avg"] = aum_data[past_aum_avg].sum(axis=1) / len(past_aum_avg)
    aum_data["month7_over_past_aum_change"] = aum_data["aum_7"] / (aum_data["past_months_aum_avg"]+0.1) - 1

    def markAsChurn(x, threshold=-0.2):
        if x <= threshold:
            return 1
        else:
            return 0
    aum_data["is_churn"] = aum_data["month7_over_past_aum_change"].apply(markAsChurn)


def tag_is_churn(data_y, old_month_num=3):
    """
    :param data_y: 7 month aum data 07.csv
    :return: 生成is_churn, smooth3个月，然后sharp drop为流失1；smooth6个月和一直grow的是0
    """
    churn_list = get_churn_list(data_y,continuous_flat_months=old_month_num)
    data_y["is_churn"] = np.nan
    data_y.loc[data_y[data_y.index.isin(churn_list)].index, "is_churn"] = 1
    smooth_list = set(get_smooth_list(data_y))
    grow_list = set(get_grow_list(data_y))
    data_y.loc[data_y[data_y.index.isin(smooth_list.union(grow_list))].index, "is_churn"] = 0


def get_churn_list(aum_data, continuous_flat_months=3):
    """
    :param aum_churn_merged:
    :param continuous_flat_months: 连续平稳n个月
    :return: 连续平稳n个月 变化不超过10%，第n+1个月drop20%, 目标是预测这类流失人群
    """
    aum_month9_with_churn = aum_data.copy()
    for i in range(7 - continuous_flat_months, 7):
        next_over_before = "month" + str(i) + "_over_month_" + str(i - 1)
        aum_month9_with_churn[next_over_before] = aum_month9_with_churn["aum_" + str(i)] / aum_month9_with_churn["aum_" + str(i - 1)] - 1
        aum_month9_with_churn = aum_month9_with_churn[abs(aum_month9_with_churn[next_over_before]) <= 0.1]
    aum_month9_with_churn["past6_avg"] = aum_month9_with_churn[["aum_" + str(x) for x in range(1, 7)]].sum(axis=1) / 6
    aum_month9_with_churn["month7_over_past6"] = aum_month9_with_churn["aum_7"] / (aum_month9_with_churn["past6_avg"] + 0.1) - 1
    churn_list = aum_month9_with_churn[aum_month9_with_churn["month7_over_past6"] <= -0.2].index
    # aum_month9_with_churn.sample(n=100)[["aum_"+str(i) for i in range(1, 8)]].T.plot()
    # plt.show()
    return churn_list


def get_smooth_list(aum_data):
    """
    :param aum_data:
    :return: aum一直波动不大的人的index list
    """
    smooth = aum_data.copy()
    for i in range(1, 8):
        next_over_before = "month" + str(i) + "_over_month_" + str(i - 1)
        smooth[next_over_before] = smooth["aum_" + str(i)] / smooth["aum_" + str(i - 1)] - 1
        smooth = smooth[abs(smooth[next_over_before]) <= 0.1]
    return smooth.index


def get_grow_list(aum_data):
    """
    :param aum_data: 7个月aum
    :return: aum一直是缓慢增长的人的index list
    """
    grow = aum_data.copy()
    for i in range(1, 8):
        next_over_before = "month" + str(i) + "_over_month_" + str(i - 1)
        grow[next_over_before] = grow["aum_" + str(i)] / grow["aum_" + str(i - 1)] - 1
        grow = grow[grow[next_over_before] >= 0.1]
    return grow.index

