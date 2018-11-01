import pandas as pd

import json
import os
import re


def add_tag_remark(data):
    for key, value in params["features"].items():
        data[key] = pd.cut(data[value["column"]], value["cut"], labels=value["labels"])
    data.loc[:, "is_2y_buy_ifm"] = data["ifm_days_avg"].apply(lambda x: "是" if x > 0 else "否")
    data.loc[:, "recent_aum_drop"] = pd.qcut(data["aum_ledger_bal"]/data["month6_aum_avg"], q=[0, 0.25, 0.5, 0.75, 1],
                                             labels=["重度流失", "中重度流失", "中度流失", "轻度流失"])
    for row, values in data.iterrows():
        if data.loc[row, "cluster"] == "其他-存利得客户":
            data.loc[row, "tag"] = "其他-存利得客户"
        else:
            data.loc[row, "tag"] = "高净值" + data.loc[row, "cluster"] if data.loc[row, "month6_aum_avg"] >= 800000 \
                else "普通" + data.loc[row, "cluster"]
        data.loc[row, "infos"] = "年龄：" + str(values["age_category"]) + \
                                 "| 行龄：" + str(values["bank_year_level"]) + \
                                 "| 客户等级：" + str(values["aum_level"]) + "({}W)".format((round(values["month6_aum_avg"]/10000)))+ \
                                 "；半年最高AUM：" + str(round(values["max_history_aum_m_avg"]/10000)) + "W " + \
                                 "；近期流失度：{}".format(values["recent_aum_drop"]) + \
                                 "| 定期：利率—" + str(values["fix_rate_level"]) + "%"\
                                 "；期限—" + str(values["fix_term_level"]) + "天"\
                                 "；笔均金额—" + str(values["fix_buy_amt"]) + \
                                 "| 是否买过理财：{}".format(values["is_2y_buy_ifm"]) + \
                                 "；客户风险等级—" + str(values["risk_level"]) + \
                                 "| 近期活期占比：{}".format(values["cur_pert"])
        if re.search(r'到期|小微企业', data.loc[row, "tag"]):
            data.loc[row, "infos"] = data.loc[row, "infos"] + \
                                 "| 最近到期: 45天定期到期金额" + str(round(values["fix_future45d_expire_amt"]/10000, 1)) + "W " + \
                                 "；45天理财到期金额" + str(round(values["ifm_future45d_expire_amt"]/10000, 1)) + "W" + \
                                 "；60天定期理财到期金额" + str(round(values["fix_ifm_future2m_expire_amt"]/10000, 1)) + "W" + \
                                 "；过去2个月理财到期金额" + str(round(values["ifm_recent2m_expire_amt"] / 10000, 1)) + "W" + \
                                 "；过去2个月定期到期金额" + str(round(values["fix_recent2m_expire_amt"] / 10000, 1)) + "W"
        data.loc[row, "remark"] = params["remark"][values["cluster"]]["reco_reason"]
        data.loc[row, "telling"] = params["telling"][values["cluster"]]
    return data


def optu_priority(tag):
    if re.search(r'高净值|存利得', tag):
        return 3
    elif re.search(r"到期|小微企业主", tag):
        return 2
    else:
        return 1


def crm_template(threads_data):
    threads = pd.DataFrame()
    for row, values in threads_data.iterrows():
        threads.loc[row, "cust_no"] = row
        threads.loc[row, "cust_tab"] = values["tag"]        # 客群分类
        threads.loc[row, "optu_priority"] = optu_priority(values["tag"])
        threads.loc[row, "track_user"] = values["client_manager"]
        threads.loc[row, "optu_type"] = "10"
        threads.loc[row, "optu_case_id"] = "CASE0004-GDLKLS002"
        threads.loc[row, "optu_id"] = "GDLKLS" + values["manager_branch"] + "20181029"+"{:0>5}".format(row)
        threads.loc[row, "optu_name"] = "高端老客防流失精准线索"
        threads.loc[row, "optu_gener_date"] = "20181029"
        threads.loc[row, "optu_end_date"] = "20181129"
        threads.loc[row, "reco_prds"] = "01B-%||||,02B-%||||,03B-%||||,04B-%||||"       # 涉及推荐产品编号
        threads.loc[row, "reco_reason"] = values["remark"]          # 推荐理由 *****
        threads.loc[row, "reco_example"] = values["telling"]        # 话术   *****
        threads.loc[row, "reco_advantage"] = ""                     # 产品优势
        threads.loc[row, "reco_remark"] = values["infos"]           # 客户信息    *******
        threads.loc[row, "verify_tot_amt_min"] = ""                # 起买金额
        threads.loc[row, "verify_tot_amt_max"] = ""                 # 购买上限金额
        threads.loc[row, "month6_aum_avg"] = values["month6_aum_avg"]                  # 过去六个月aum
        threads.loc[row, "recent_aum_drop"] = values["recent_aum_drop"]
    return threads


# 生成下发线索
def get_opt_threads(features):
    handouts_crm_features = add_tag_remark(features)
    result_data = crm_template(handouts_crm_features)
    return result_data


# 按order_by排序，去掉最小的5%去掉最大的5%，抽取5.5%抽取
def split_five_perct_control_group(data, order_by):
    data_order = data.sort_values(by=order_by)
    data_rows_cnt = data.shape[0]
    data_ninety_percent = data_order.iloc[
                                 round(data_rows_cnt * 0.05):round(data_rows_cnt * 0.95), :]
    no_crm_leads = data_ninety_percent.sample(frac=0.055)       # 非下发线索
    crm_show_leads = data[~data.index.isin(no_crm_leads.index)]
    return no_crm_leads, crm_show_leads


def read_config(runtime=1):
    with open("config.json", encoding="utf-8") as f:
        params = json.load(f)
    if runtime == 1:
        clusters = ['小微企业主', '协议到期客户', '定期非到期户', '理财非到期户', '活期占比高客户', '提前支取客户',
                    '其他-存利得客户', '其他']
        cluster_tell = ['小微企业主', '协议到期客户', "非到期客户", "非到期客户", '活期占比高客户', "存利得客户",
                        '提前支取客户', '活期占比高客户']
        tell_cluster = {}
        for cluster_name, tell_sheet in zip(clusters, cluster_tell):
            tell = pd.read_excel("高端老客驻马店话术.xlsx", sheet_name=tell_sheet)
            telling_text = ''
            for row, values in tell.iterrows():
                telling_text += "#{}*{}".format(values["if"], values["then"])
            tell_cluster[cluster_name] = telling_text
        params["telling"] = tell_cluster
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(params, f)
    return params


if __name__ == '__main__':
    folder = "zhumadian"
    leads_folder = os.path.join(folder, "handout_list")
    params = read_config(runtime=2)
    leads_features = pd.read_excel(os.path.join(leads_folder, "leads_segments.xlsx"), index_col="client_no")
    crm_leads = get_opt_threads(leads_features)
    crm_leads.to_excel("驻马店分行高潜流失CRM线索标签明细1026.xlsx")

    crm_leads = pd.read_excel("驻马店分行高潜流失CRM线索标签明细1026.xlsx")
    no_crm_leads, crm_show_leads = split_five_perct_control_group(crm_leads, "month6_aum_avg")
    writer = pd.ExcelWriter("驻马店分行高潜流失CRM线索下发with对照组.xlsx")
    crm_show_leads.to_excel(writer, sheet_name="下发组", index=None)
    no_crm_leads.to_excel(writer, sheet_name="对照组", index=None)
    writer.close()

















