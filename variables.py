import pandas as pd
import re
import json


"""
    变量配置
"""
# variable quality is terrible or it is useless(highly correlated with definition of churn or not useful because a
# salesperson also knows that recent aum drop is dangerous )
delete_items = ["is_loan", "current_ledger_bal",
                "fixed_ledger_bal", "cfm_ledger_bal", "avg_ifm_client_ratio",
                "max_acct_dta_ledger_bal", "max_acct_cod_ledger_bal", "max_cfm_ledger_bal",
                "ifm_first_trans_date", "ifm_first_tran_amt", "ifm_last_trans_date", "ifm_last_tran_amt",
                "last_one_month_max_tran_amt", "last_one_month_min_tran_amt",
                "ifm_tran_count", "ifm_tran_amt", "ifm_year_avg_month_sum",
                "month1_aum_change", "month3_aum_change", "three_month_ledger_over_year", "three_month_ledger_over_six"]


model_variables = pd.read_excel("./config/final_columns_240.xlsx")["name"].tolist()

# standardizedVariables = [values["name"] for _, values in variable_list.iterrows()
#                          if re.search(r'次数|时长|变化', values["变量含义"])].append(["bank_year", "prdt_hist_hold"])

toStringVariables = ["open_acct_org", "oth_bank_code"]
customerType = ["is_salary", "is_sign_djf", "is_hnkh", "is_bil_mchnt", "is_open_sms",
                "is_open_pibs", "is_open_pmbs", "is_emp"]
is_buy_productType = ["last_three_month_is_buy_tda", "last_three_month_is_buy_ifm",
                      "last_three_month_is_buy_jj",	"last_three_month_is_buy_insure"]
yesno_columns = customerType + is_buy_productType

# logVariables = [values["name"] for _, values in variable_list.iterrows()
#                 if re.search(r'余额|金额|AUM$', values["变量含义"])]

cluster_x = [
             # 产品活跃度
             "NumofProductRecentBuy",
             "prdt_hist_hold",
             "mobile_bank_cnt_12",
             "is_hist_risk_prdt",
             # 资产雄厚
             "month6_aum_avg",
             "month12_aum_avg",
             # 资金下滑严重度
             "last_six_month_d_max_tran_amt_over_aumavg",
             "last_six_month_d_max_tran_amt_over_avg_amt_per_tran",
             "last_six_month_c_max_tran_amt",
             "cross_bank_amt_per_tran_3",
             "consume_amt_1_12",
             # 理财
             "cfm_compare_aum_1",      # 过去一个月理财占AUM比
             "cfm_compare_aum_12",
             "ifm_sum_amt_avg",        # 历史上理财平均最低起买金额
             "ifm_avg_tran_amt",       # 历史理财笔均金额
             "ifm_days_avg",           # 历史上理财平均期限
             "ifm_guest_rate_avg",     # 历史理财平均利率
             "ifm_year_count",         # 近一年理财交易次数
             "ifm_year_sum",           # 近一年理财交易金额
             "last_three_month_is_buy_ifm", # 过去三个月是否买理财
             # 定期
             "deposit_compare_aum_1",  # 过去一个月储蓄占aum比
             "deposit_compare_aum_12",
             "fix_days_avg",           # 历史定期平均期限
             "fix_sum_amt_avg",        # 历史定期笔均金额
             "interest_rate_avg",      # 历史上定期平均利率
             "fix_rate_avg_12m",       # 过去12个月平均利率
             "last_12m_fix_term_avg",  # 过去12个月定期平均期限
             "last_12m_fix_buy_cnt",   # 过去12个月定期购买次数
             "last_12m_fix_buy_amt",   # 过去12个月定期金额
             # 可用资金
             "cur_1m_compare_aum_rate",
             "cur_12m_compare_aum_rate",
             "expire_fix_ifm_acct_amt_over_aum",
             "fix_ifm_acct_amt",
             # 其他
             "bank_year",
             "age",
             "gender",
             "is_salary",
             "last_one_year_avg_df_amt"
             ]

churn_columns = pd.read_excel("config/tranX.xlsx")["name"].tolist()
cluster_columns = pd.read_excel("config/cluster_columns.xlsx")["name"].tolist()

params = {
    "delete_items": delete_items,
    "toStringVariables": toStringVariables,
    "yesno_columns": customerType + is_buy_productType,
    "all_variables_X_y": model_variables,                   # 最初的流失模型240个标签，最早
    "customerType": customerType,
    "downsale_all_columns": churn_columns,                  # 最终加上理财标签，和post 名单标签 377
    "pre_cluster_x": cluster_x,                             # 用于聚类的最原始字段
    "cluster_x": cluster_columns
}

if __name__ == '__main__':

    json.dump(params, "config/params.json")

