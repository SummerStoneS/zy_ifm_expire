"""
@time: 8/22/2018 7:44 PM

@author: 柚子
"""
import pandas as pd
"""
    定义到期6类人
    0：AUM流失不续买；
    1：AUM流失续买；
    2：AUM不变不续买；
    3：AUM不变续买；
    4：AUM增加不续买；
    5：AUM增加续买；
"""


def tag_y(ifm_0csv, products_contbuy="all"):
    """
    :param ifm_0csv:
    :param products_contbuy: all->续买任何种类产品， ifm->续买非保本理财, 或者是list
    :return: 预测AUM上升下降y1和续买非保本理财y2, one_hot y1
    """
    ifm_0csv.loc[:, "总AUM变化"] = ifm_0csv["next_seven_day_aum_bal"] - ifm_0csv["last_one_day_aum_bal"]
    ifm_0csv.loc[:, "y1"] = (ifm_0csv["总AUM变化"]).apply(
        lambda x: "down" if x <= -2000 else "up" if x >= 2000 else "even")  # 0.77
    if products_contbuy == "ifm":
        ifm_0csv.loc[:, "y2"] = ifm_0csv["no_guaranteed_ifm_amt"].apply(lambda x: 1 if x > 0 else 0)
    elif products_contbuy == "all":
        all_products = ["buy_ifm_amt", "buy_fix_amt", "buy_jj_amt", "buy_cld_amt"]
        ifm_0csv.loc[:, "y2"] = ifm_0csv[all_products].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    elif isinstance(products_contbuy, list):
        ifm_0csv.loc[:, "y2"] = ifm_0csv[products_contbuy].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    else:
        raise ValueError("wrong product name")
    print(ifm_0csv["y1"].value_counts()/len(ifm_0csv))
    print(ifm_0csv["y2"].value_counts()/len(ifm_0csv))
    y1_onehot = pd.get_dummies(ifm_0csv["y1"], prefix='y1')
    features_and_ys = pd.concat([ifm_0csv, y1_onehot], axis=1)
    features_and_ys = features_and_ys.reset_index()
    # 输出y和x的correlation
    for y1_value in features_and_ys["y1"].unique():
        for y2_value in features_and_ys["y2"].unique():
            new_name = "y1_{}_y2_{}".format(y1_value, y2_value)
            is_yes_index = features_and_ys.query("(y1==@y1_value)&(y2==@y2_value)").index
            features_and_ys.loc[is_yes_index, new_name] = 1
            features_and_ys[new_name].fillna(0, inplace=True)
    features_and_ys.corr().to_excel("check_correlation_x_y.xlsx")
    # features_and_ys = features_and_ys[params["all_features"] + y1_onehot.columns.tolist() + ["y2"]]
    return features_and_ys


def tag_multiclass(ifm_0csv):
    ifm_0csv.loc[:, "总AUM变化"] = ifm_0csv["next_seven_day_aum_bal"] - ifm_0csv["last_one_day_aum_bal"]
    all_products = ["buy_ifm_amt", "buy_fix_amt", "buy_jj_amt", "buy_cld_amt"]
    ifm_0csv.loc[:, "y2"] = ifm_0csv[all_products].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    ifm_0csv.loc[:, "AUM变化程度"] = (ifm_0csv["总AUM变化"]).apply(
        lambda x: "down" if x <= -2000 else "up" if x >= 2000 else "even")
    ifm_0csv.loc[ifm_0csv[(ifm_0csv["AUM变化程度"] == "down") & (ifm_0csv["y2"] == 0)].index, "y"] = 0
    ifm_0csv.loc[ifm_0csv[(ifm_0csv["AUM变化程度"] == "down") & (ifm_0csv["y2"] == 1)].index, "y"] = 1
    ifm_0csv.loc[ifm_0csv[(ifm_0csv["AUM变化程度"] == "even") & (ifm_0csv["y2"] == 0)].index, "y"] = 2
    ifm_0csv.loc[ifm_0csv[(ifm_0csv["AUM变化程度"] == "even") & (ifm_0csv["y2"] == 1)].index, "y"] = 3
    ifm_0csv.loc[ifm_0csv[(ifm_0csv["AUM变化程度"] == "up") & (ifm_0csv["y2"] == 0)].index, "y"] = 4
    ifm_0csv.loc[ifm_0csv[(ifm_0csv["AUM变化程度"] == "up") & (ifm_0csv["y2"] == 1)].index, "y"] = 5
    print(ifm_0csv["y"].value_counts())
    return ifm_0csv






