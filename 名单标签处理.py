import pandas as pd
import json
import os


def tag_cluster(x):
    if x in [0, 1]:
        return "高流失倾向客户"
    elif x in [2, 4]:
        return "资金闲置倾向客户"
    else:
        return "续持倾向客户"


def add_tag_remark(data):
    data["cluster"] = data["pred"].apply(tag_cluster)
    for key, value in params["features"].items():
        data[key] = pd.cut(data[value["column"]], value["cut"], labels=value["labels"])

    for row, values in data.iterrows():
        data.loc[row, "infos"] =\
            "到期日：{}".format(round(values["end_date"])) + \
            "；第一笔到期金额：{}".format(round(values["ifm_expire_amt"])) + \
            "；未来一个月到期笔数：{}".format(round(values["ifm_future30d_expire_cnt"])) + \
            "；未来一个月到期金额：{}".format(round(values["ifm_future30d_expire_amt"])) + \
            "|年龄：{}".format(values["age_category"]) + \
            "| 行龄：{}".format(values["bank_year_level"]) + \
            "| 客户等级：{}".format(values["aum_level"]) + "({}W)".format((round(values["month6_aum_avg"]/10000))) + \
            "；半年最高AUM：{}".format(round(values["max_history_aum_m_avg"]/10000)) + "W " + \
            "| 定期：利率—{}".format(values["fix_rate_level"]) + "%"\
            "；期限—{}".format(values["fix_term_level"]) + "天"\
            "；笔均金额—{}".format(values["fix_buy_amt"]) + \
            "| 理财：期限—{}".format(values["ifm_term_level"]) + \
            "；金额—{}".format(str(values["ifm_buy_amt"])) + \
            "；客户风险等级—{}".format(values["risk_level_desc"]) + \
            "| 近期活期占比：{}".format(str(values["cur_pert"]))

        data.loc[row, "remark"] = params["reco_reason"][values["cluster"]]
        data.loc[row, "telling"] = params["telling"][values["cluster"]]
    return data


def optu_priority(tag):
    if tag == "高流失倾向客户":
        return 3
    elif tag == "资金闲置倾向客户":
        return 2
    else:
        return 1


def crm_template(threads_data):
    threads = pd.DataFrame()
    for row, values in threads_data.iterrows():
        threads.loc[row, "cust_no"] = row
        threads.loc[row, "cust_tab"] = values["cluster"]        # 客群分类
        threads.loc[row, "optu_priority"] = optu_priority(values["cluster"])
        threads.loc[row, "track_user"] = values["client_manager"]
        threads.loc[row, "optu_type"] = "10"
        threads.loc[row, "optu_case_id"] = "CASE0002-LCDQ001"
        threads.loc[row, "optu_id"] = "GDLKLS" + values["manager_branch"] + "20181031"+"{:0>5}".format(row)
        threads.loc[row, "optu_name"] = "理财到期精准线索"
        threads.loc[row, "optu_gener_date"] = "20181031"
        threads.loc[row, "optu_end_date"] = "20181130"
        threads.loc[row, "reco_prds"] = "01B-%||||,02B-%||||,03B-%||||,04B-%||||"       # 涉及推荐产品编号
        threads.loc[row, "reco_reason"] = values["remark"]          # 推荐理由 *****
        threads.loc[row, "reco_example"] = values["telling"]        # 话术   *****
        threads.loc[row, "reco_advantage"] = ""                     # 产品优势
        threads.loc[row, "reco_remark"] = values["infos"]           # 客户信息    *******
        threads.loc[row, "verify_tot_amt_min"] = ""                 # 起买金额
        threads.loc[row, "verify_tot_amt_max"] = ""                 # 购买上限金额
        threads.loc[row, "到期日"] = round(values["end_date"])         # 购买上限金额
        threads.loc[row, "month6_aum_avg"] = values["month6_aum_avg"]  # 购买上限金额
    return threads


# 生成下发线索
def get_opt_threads(features):
    handouts_crm_features = add_tag_remark(features)
    result = crm_template(handouts_crm_features)
    return result


# 按order_by排序，去掉最小的5%去掉最大的5%，抽取5.5%抽取
def split_five_perct_control_group(data, order_by):
    data_order = data.sort_values(by=order_by)
    data_rows_cnt = data.shape[0]
    data_ninety_percent = data_order.iloc[
                                 round(data_rows_cnt * 0.05):round(data_rows_cnt * 0.95), :]
    no_crm_leads = data_ninety_percent.sample(frac=0.055)       # 非下发线索
    crm_show_leads = data[~data.index.isin(no_crm_leads.index)]
    return no_crm_leads, crm_show_leads


def read_config():
    with open("config_old.json", encoding="utf-8") as f:
        param = json.load(f)
    clusters = ['高流失倾向客户', '资金闲置倾向客户', '续持倾向客户']
    tell_cluster = {}
    for cluster_name in clusters:
        tell = pd.read_excel("理财到期话术.xlsx", sheet_name=cluster_name)
        telling_text = ''
        for row, values in tell.iterrows():
            telling_text += "#{}*{}".format(values["if"], values["then"])
        tell_cluster[cluster_name] = telling_text
    param["telling"] = tell_cluster
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(param, f)
    return param


if __name__ == '__main__':
    folder = "zhumadian"

    train_periods = ["20180907", "20180914", "20180815", "20180719", "20180608", "20180507", "20180405"]
    predict_period = "20181029"
    model_folder = os.path.join(folder, "model_result", "train" + "_".join(train_periods) + "test")
    predict_folder = os.path.join(model_folder, "predict_" + predict_period)  # 预测结果保存地址
    result_data = pd.read_excel(os.path.join(predict_folder, "predict_result_{}.xlsx".format(predict_period)), index_col="client_no")
    # 补上导入CRM系统需要的标签
    crm_features = pd.read_csv(open(os.path.join(folder, "data", predict_period, predict_period+"011.csv")), index_col="client_no")
    expire_features = pd.read_csv(open(os.path.join(folder, "data", predict_period, "new", predict_period + "00.csv")),index_col="client_no")
    full_features = pd.concat([result_data, crm_features, expire_features[["ifm_expire_amt", "end_date"]]], axis=1)
    full_features = full_features[full_features["month6_aum_avg"].notnull()]
    # 转换成CRM系统字段要求
    params = read_config()
    crm_leads = get_opt_threads(full_features)
    # 抽对照组
    no_crm_leads, crm_show_leads = split_five_perct_control_group(crm_leads, "month6_aum_avg")
    writer = pd.ExcelWriter("驻马店分行理财到期CRM线索下发with对照组1030.xlsx")
    crm_show_leads.to_excel(writer, sheet_name="下发组", index=None)
    no_crm_leads.to_excel(writer, sheet_name="对照组", index=None)
    writer.close()
















