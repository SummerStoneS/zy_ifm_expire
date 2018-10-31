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


model_variables = pd.read_excel("config/final_columns_240.xlsx")["name"].tolist()

# standardizedVariables = [values["name"] for _, values in variable_list.iterrows()
#                          if re.search(r'次数|时长|变化', values["变量含义"])].append(["bank_year", "prdt_hist_hold"])

toStringVariables = ["open_acct_org", "oth_bank_code"]
customerType = ["is_salary", "is_sign_djf", "is_hnkh", "is_bil_mchnt", "is_open_sms",
                "is_open_pibs", "is_open_pmbs", "is_emp"]
is_buy_productType = ["last_three_month_is_buy_tda", "last_three_month_is_buy_ifm",
                      "last_three_month_is_buy_jj",	"last_three_month_is_buy_insure"]
yesno_columns = customerType + is_buy_productType

final_columns = pd.read_excel("config/tranX.xlsx")["name"].tolist()
logVariables = pd.read_excel("config/tranX.xlsx", sheet_name="log")["name"].tolist()

categoricals = pd.read_excel("config/tranX.xlsx", sheet_name="categorical")["name"].tolist()
features_with_y = pd.read_excel("config/tranX.xlsx", sheet_name="before_y")["name"].tolist()

params = {
    "delete_items": delete_items,
    "toStringVariables": toStringVariables,
    "yesno_columns": customerType + is_buy_productType,
    "all_variables_X_y": model_variables,                   # 最初的流失模型240个标签，最早
    "customerType": customerType,
    "all_features": final_columns,                  # 最终加上理财标签，和post 名单标签 377
    "log_variables": logVariables,
    "categoricals": categoricals,
    "features_before_y_define": features_with_y
}


# 每个类别的recall阈值
alpha_dict = {"y1_up_y2_1": 0.62,
              "y1_up_y2_0": 0.66,
              "y1_even_y2_1": 0.67,
              "y1_even_y2_0": 0.67,
              "y1_down_y2_1": 0.73,
              "y1_down_y2_0": 0.73}

if __name__ == '__main__':

    json.dump(params, "config/params.json")

