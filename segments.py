"""
@time: 10/17/2018 2:44 PM

@author: 柚子
"""
import os
import pandas as pd
from preprocess import folder


def get_handout_data(top_n=2):
    """
    :param top_n: select top n group as handout list(which has the highest prob of churn)
    :return:
    """
    data = pd.read_csv(os.path.join(save_folder, "prediction_result_data.csv"), index_col="client_no")
    # cluster
    reachOut_leads = data[data["pred_group"] > (9-top_n)]         # 3279条
    reachOut_leads.to_csv(os.path.join(leads_folder, "线索所有数据.csv"))
    return reachOut_leads


def split_seller(cluster_data):
    """
    :param cluster_data:
    :return: 收单客户（交易top60%）和交易极度频繁的人（top 5%的人）当做小微企业户
    """
    seller = cluster_data[cluster_data["is_seller"] == 1]
    nonseller = cluster_data[~cluster_data.index.isin(seller.index)]
    return seller, nonseller


def split_by_rules(data, rule_col, rule_value):
    a = data[data[rule_col] > rule_value]
    b = data[data[rule_col] <= rule_value]
    return a, b


def splitSpecialSegements(leads):
    seller_leads, nonseller_leads = split_seller(leads)  # 区分小微企业主和其他人

    autoexpired = nonseller_leads[(nonseller_leads["ifm_recent2m_expire_amt_over_aum6"] > 0) |
                                  (nonseller_leads["fix_recent2m_expire_amt_over_aum6"] > 0) |
                                  (nonseller_leads["fix_ifm_future2m_expire_amt"] > 0)]   # 最近两个月自动到期或未来2个月有到期

    nonexpired = nonseller_leads[~nonseller_leads.index.isin(autoexpired.index)]
    withdraw = nonexpired[(nonexpired["ifm_recent2m_withdraw_amt_over_aum6"] > 0) |
                          (nonexpired["fix_recent2m_withdraw_amt_over_aum6"] > 0)]       # 最近两个月提前支取
    nonwithdraw = nonexpired[~nonexpired.index.isin(withdraw.index)]
    fixuser, nonfix = split_by_rules(nonwithdraw, "fix_history_amt", 0)  # 定期户,历史上有过定期
    ifmuser, nonifm = split_by_rules(nonfix, "ifm_days_avg", 0)             # 理财户
    highcurrent = nonifm[(nonifm["acct_cur_ledger_bal"] > 0) & (nonifm["cur_1m_compare_aum_rate"] > 0.7)]
    lowcurrent = nonifm[~nonifm.index.isin(highcurrent.index)]                           # 活期占比高

    cunlide = lowcurrent[lowcurrent["if_buy_cld"] == "Y"]                      # 是否存利得
    others = lowcurrent[lowcurrent["if_buy_cld"] == "N"]

    seller_leads.loc[:, "cluster"] = "小微企业主"
    autoexpired.loc[:, "cluster"] = "协议到期客户"
    withdraw.loc[:, "cluster"] = "提前支取客户"
    fixuser.loc[:, "cluster"] = "定期非到期户"
    ifmuser.loc[:, "cluster"] = "理财非到期户"
    highcurrent.loc[:, "cluster"] = "活期占比高客户"
    cunlide.loc[:, "cluster"] = "其他-存利得客户"
    others.loc[:, "cluster"] = "其他"
    clusters = pd.concat([seller_leads, autoexpired, fixuser, ifmuser, highcurrent, withdraw, cunlide, others])
    return clusters


if __name__ == '__main__':
    leads_folder = os.path.join(folder, "handout_list")
    if not os.path.exists(leads_folder):
        os.makedirs(leads_folder)

    train_periods = ["20180630", "20180331", "20171231", "20170930"]
    predict_period = "20180930"
    model_folder = "train" + "_".join(train_periods) + "predict_" + predict_period
    save_folder = os.path.join(folder, "model_result", model_folder)

    leads_data = get_handout_data(top_n=2)                      # 所有线索中得分最高的20%人群
    leads_features = pd.read_csv(open(os.path.join(folder, "data", predict_period, predict_period + "011.csv")), index_col="client_no") # 为了细分客群单独拉的
    leads_data = pd.merge(leads_data, leads_features, how="left", left_index=True, right_index=True)
    leads_data.to_csv(os.path.join(leads_folder, "leads_segments_full_features.csv"))
    # leads_data = pd.read_csv(os.path.join(leads_folder, "leads_segments.csv"), index_col="client_no")
    segments = splitSpecialSegements(leads_data)
    segments.to_excel(os.path.join(leads_folder, "leads_segments.xlsx"))
    print(segments["cluster"].value_counts())
    print(len(segments), len(leads_data))



