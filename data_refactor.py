"""
@time: 8/7/2018 5:58 PM

@author: 柚子
"""
import pandas as pd
from datetime import datetime
import re
import numpy as np


def is_new_customer(x, new_year):
    """
    :return: 20131227.0
    """
    date_string = str(x)[:8]
    open_account_date = datetime.strptime(date_string, "%Y%m%d")
    if open_account_date.year == new_year:
        return 1
    else:
        return 0


def tag_new_customer(data, new_customer_base_year=2017):
    data = data[data["open_acct_dt"].notnull()]
    data["is_new_customer"] = data["open_acct_dt"].apply(lambda x: is_new_customer(x, new_customer_base_year))
    del data["open_acct_dt"]
    return data


def num_to_str(x):
    if np.isnan(x):
        return np.nan
    else:
        return str(round(x))


def numcolumns_to_str(data, columns):
    for column in columns:
        data[column] = data[column].apply(num_to_str)
    return data


def is_male(id_string):
    x = id_string[-2]
    if int(x) % 2 == 0:
        return 0
    else:
        return 1


def transform_age_gender(data):
    data["gender"] = data["gender"].replace({"男": 1, "女": 0, "未知": np.nan})
    data = data[data["age"] < 100]
    return data


def cal_age_from_datestr(id_string):
    if len(id_string) == 15:
        datestring = "19" + id_string[6:12]
    elif len(id_string) == 18:
        datestring = id_string[-12:-4]
    else:
        return 0
    try:
        birth_date = datetime.strptime(datestring, '%Y%m%d')
    except:
        print(datestring)
        return 0
    age = datetime(2018, 6, 30) - birth_date
    return round(age.days/365)


class IdProcessor:
    def __init__(self, ori_data):
        self.filtered_data = ori_data

    def remove_abnormal_id(self):
        self.filtered_data = self.filtered_data[(self.filtered_data["cert_no"].str.len() == 15) |
                                                (self.filtered_data["cert_no"].str.len() == 18)]

    def cal_age(self):
        self.filtered_data["age"] = self.filtered_data["cert_no"].apply(cal_age_from_datestr)

    def remove_abnormal_age(self):
        self.filtered_data = self.filtered_data[self.filtered_data["age"] > 0]

    def cal_gender(self):
        self.filtered_data["gender"] = self.filtered_data["cert_no"].apply(is_male)

    def delete_cert_no(self):
        del self.filtered_data["cert_no"]

    def id_to_gender_age(self):
        self.remove_abnormal_id()
        self.cal_age()
        self.remove_abnormal_age()
        self.cal_gender()
        self.delete_cert_no()
        return self.filtered_data


def convert_yesno_to_num(model_data, convert_columns):
    for yesno_col in convert_columns:
        if model_data[yesno_col].isnull().sum() == len(model_data):
            model_data[yesno_col] = 0
        else:
            model_data[yesno_col] = model_data[yesno_col].replace({'Y': 1, 'N': 0})
    return model_data


def process_basic_info(original_data, customerType_list):
    data = original_data.copy()
    data["customerType_quantity"] = data[customerType_list].sum(axis=1)
    del data["is_bil_mchnt"], data["is_hnkh"]
    return data


def process_historical_transaction(history_transaction_data):
    """
    :param history_transaction_data: 最近1,3,6,12个月的最大、最小、sum入账出账交易次数金额
    :return: 变成相对之前的变化
    """

    lag_time_periods = ["one_month", "three_month", "six_month", "one_year"]
    lag_time_periods.reverse()

    def replace_long_period_count_amt(model_data):
        data = model_data.copy()
        for long_pos in range(len(lag_time_periods) - 1):  # 近3个月的不包括近1个月的
            for transaction_type in ["c", "d"]:
                for summary_type in ["count", "amt"]:
                    long_period_name = "_".join(
                        ["last", lag_time_periods[long_pos], transaction_type, "tran", summary_type])
                    short_period_name = "_".join(
                        ["last", lag_time_periods[long_pos + 1], transaction_type, "tran", summary_type])
                    data[long_period_name] = data[long_period_name] - data[short_period_name]
            for summary_type in ["pmbs_login_count"]:
                long_period_name = "_".join(["last", lag_time_periods[long_pos], summary_type])
                short_period_name = "_".join(["last", lag_time_periods[long_pos + 1], summary_type])
                data[long_period_name] = data[long_period_name] - data[short_period_name]
        return data

    def cal_max_over_min():
        for period in lag_time_periods:
            for transaction_type in ["_c", "_d"]:
                max_name = "last_" + period + transaction_type + "_max_tran_amt"
                min_name = "last_" + period + transaction_type + "_min_tran_amt"
                new_data[period + transaction_type + "_max_over_min"] = data[max_name] / (data[min_name]+0.1)

    def cal_amt_per_transac():
        for period in lag_time_periods:
            for transaction_type in ["c", "d"]:     # 计算笔均入账金额，笔均出账金额
                avg_amt_per_tran_name = "last" + "_" + period + "_" + transaction_type + "_avg_amt_per_transac"
                amount_name = "_".join(["last", period, transaction_type, "tran", "amt"])
                count_name = "_".join(["last", period, transaction_type, "tran", "count"])
                new_data[avg_amt_per_tran_name] = data[amount_name] / (data[count_name] + 0.1)

    def cal_amt_count_change():
        for long_period in lag_time_periods[:-1]:
            for short_period in lag_time_periods[lag_time_periods.index(long_period)+1:]:
                for transaction_type in ["c", "d"]:
                    for summary in ["tran_count", "tran_amt", "max_tran_amt", "min_tran_amt", "avg_amt_per_transac"]:
                        new_itemname = "_".join([short_period, long_period, transaction_type, summary])
                        short_period_name = "_".join(["last", short_period, transaction_type, summary])
                        long_period_name = "_".join(["last", long_period, transaction_type, summary])
                        if summary == "avg_amt_per_transac":
                            new_data[new_itemname] = new_data[short_period_name] / (new_data[long_period_name] + 0.1)
                        else:
                            new_data[new_itemname] = data[short_period_name] / (data[long_period_name] + 0.1)

                pmbs_change = "_".join([short_period, long_period, "pmbs_login_count"])
                new_data[pmbs_change] = data["_".join(["last", short_period, "pmbs_login_count"])] / \
                                        (data["_".join(["last", short_period, "pmbs_login_count"])] + 0.1)

    def cal_d_over_c():
        for period in lag_time_periods:
            for summary in ["count", "amt"]:
                item_c = "_".join(["last", period, "c", "tran", summary])
                item_d = "_".join(["last", period, "d", "tran", summary])
                new_data[period + "_d_over_c_" + summary] = data[item_d] / (data[item_c] + 0.1)
            for compar in ["max", "min"]:
                compar_c = "_".join(["last", period, "c", compar, "tran", "amt"])
                compar_d = "_".join(["last", period, "d", compar, "tran", "amt"])
                new_data[period + "_d_over_c_" + compar + "_amt"] = data[compar_d] / (data[compar_c]+0.1)

    def fix_deposit_transac_owned_change():
        new_data["one_month_over_four_month_fixdeposit"] = data["fix_1month_coun"] / (data["fix_4month_coun"] + 0.1)
        new_data["one_month_over_12_month_fixdeposit"] = data["fix_1month_coun"] / (data["fix_12month_coun"] + 0.1)
        new_data["four_month_over_12_month_fixdeposit"] = data["fix_4month_coun"] / (data["fix_12month_coun"] + 0.1)

    def ledger_bal_change():
        new_data["three_month_ledger_over_six"] = data["last_three_month_max_aum_ledger_bal"] / \
                                                  data["last_six_month_max_aum_ledger_bal"]
        new_data["six_month_ledger_over_year"] = data["last_six_month_max_aum_ledger_bal"] / \
                                                 data["last_one_year_max_aum_ledger_bal"]
        new_data["three_month_ledger_over_year"] = data["last_three_month_max_aum_ledger_bal"] / \
                                                  data["last_one_year_max_aum_ledger_bal"]

    def remove_ifm_overlap():
        ifm_period = ["month", "qual", "half", "year"]
        ifm_period.reverse()
        for period_pos in range(len(ifm_period) - 1):
            for summary_type in ["count", "sum"]:
                long_period_name = "_".join(["ifm", ifm_period[period_pos], summary_type])
                short_period_name = "_".join(["ifm", ifm_period[period_pos + 1], summary_type])
                data[long_period_name] = data[long_period_name] - data[short_period_name]

        for period in ifm_period:
            amount = "_".join(["ifm", period, "sum"])
            count = "_".join(["ifm", period, "count"])
            new_data["ifm_" + period + "avg_amt_per_tran"] = data[amount] / (data[count] + 0.1)

        for long_period in ifm_period[:-1]:
            for short_period in ifm_period[ifm_period.index(long_period) + 1:]:
                for summary in ["count", "sum"]:
                    long_p_name = "_".join(["ifm", long_period, summary])
                    short_p_name = "_".join(["ifm", short_period, summary])
                    change_name = "_".join([short_period, long_period, "ifm", summary])
                    new_data[change_name] = data[short_p_name] / (data[long_p_name] + 0.1)

        new_data["ifm_max_over_min"] = data["ifm_year_max"] / (data["ifm_year_min"] + 0.1)

    def count_product_holdings_num():
        product_name = ["tda", "ifm", "jj", "insure"]
        is_buy_product_recent = ['last_three_month_is_buy_' + x for x in product_name]
        new_data["NumofProductRecentBuy"] = data[is_buy_product_recent].sum(axis=1)

    data = replace_long_period_count_amt(history_transaction_data)
    new_data = pd.DataFrame()
    cal_max_over_min()
    cal_amt_per_transac()
    cal_amt_count_change()
    cal_d_over_c()
    fix_deposit_transac_owned_change()
    ledger_bal_change()
    remove_ifm_overlap()
    count_product_holdings_num()
    new_data["deposit_expire_amt_per_coun"] = data["fix_ifm_acct_amt"] / data["fix_ifm_acct_coun"]
    new_data["avg_df_change"] = data["last_three_month_avg_df_amt"] / (data["last_one_year_avg_df_amt"] + 0.1)
    new_data["df_amt_per_count"] = data["df_count"] / (data["df_amt"] + 0.1)
    new_data["month6_month12_aum_avg"] = data["month6_aum_avg"] / (data["month12_aum_avg"] + 0.1)
    new_data["month1_c_max_over_12aum_avg"] = data["last_one_month_c_max_tran_amt"] / (data["month12_aum_avg"] + 0.1)
    new_data["month1_d_max_over_12aum_avg"] = data["last_one_month_d_max_tran_amt"] / (data["month12_aum_avg"] + 0.1)
    new_data["expire_fix_ifm_acct_amt_over_aum"] = data["fix_ifm_acct_amt"] / (data["month12_aum_avg"] + 0.1)
    amt_columns = ["ifm_recent2m_withdraw_amt", "ifm_recent2m_expire_amt", "fix_recent2m_withdraw_amt",
                   "fix_recent2m_expire_amt", "fix_ifm_future3m_expire_amt"]
    for col in amt_columns:
        new_data[col + "_over_aum6"] = data[col] / (data["month6_aum_avg"] + 0.1)

    return new_data


def channel_analysis(data2):
    """
    :param data2:
    :return: transaction channel preference using channel frequency of total transactions as preference score
    """
    channels = ["net_silver_cnt_12",    # 网银
                "counter_cnt_12",
                "large_small_pay_cnt_12",
                "unionpay_cnt_12",
                "mobile_bank_cnt_12",
                "wechat_cnt_12",
                "atm_cnt_12",
                "alipay_cnt_12",
                "multimedia_cnt_12",
                "oth_cnt_12"]
    for channel in channels:
        data2[channel] = data2[channel] / data2["all_cnt_12"]
    del data2["all_cnt_12"]     # 总交易次数


def outflow_cross_bank_and_consume(data2):
    """
    :param data2:
    :return: 消费金额和跨行转出次数，金额
    """
    time_periods = ["12", "6", "3", "1"]

    def remove_long_period_overlap():
        for type in ["consume_amt", "cross_bank_cnt", "cross_bank_amt"]:
            for time_period in range(len(time_periods)-1):
                consume_col = "_".join([type, time_periods[time_period]])
                consume_col_previous = "_".join([type, time_periods[time_period+1]])
                data2[consume_col] = data2[consume_col] - data2[consume_col_previous]

    def cal_cross_bank_amt_per_tran():
        for time_period in time_periods:
            amt_per_tran = "cross_bank_amt_per_tran_" + time_period
            data2[amt_per_tran] = data2["cross_bank_amt_"+time_period] / (data2["cross_bank_cnt_"+time_period]+0.1)

    def cal_change():
        for type in ["consume_amt", "cross_bank_cnt", "cross_bank_amt", "cross_bank_amt_per_tran"]:
            for long_period in time_periods[:-1]:
                for short_period in time_periods[time_periods.index(long_period)+1:]:
                    new_col = "_".join([type, short_period, long_period])
                    short_p_col = "_".join([type, short_period])
                    long_p_col = "_".join([type, long_period])
                    print(short_p_col, long_p_col)
                    data2[new_col] = data2[short_p_col] / (data2[long_p_col]+0.1) - 1

    def delete_abs_value():
        for time_period in time_periods:
            for type in ["consume_amt", "cross_bank_amt", "cross_bank_cnt"]:
                del_col = "_".join([type, time_period])
                del data2[del_col]

    remove_long_period_overlap()
    cal_cross_bank_amt_per_tran()
    cal_change()
    delete_abs_value()
    convert_yesno_to_num(data2, ["is_new_alipay_3"])


def fine_tune_steps(data):
    # 过去6个月最大出账金额/过去的平均AUM； /平均笔均金额
    data["last_six_month_d_max_tran_amt_over_aumavg"] = data["last_six_month_d_max_tran_amt"] / data["month6_aum_avg"]
    data["last_six_month_d_max_tran_amt_over_avg_amt_per_tran"] = data["last_six_month_d_max_tran_amt"] / \
                                                                  (data["last_one_year_d_avg_amt_per_transac"] + 0.1)
    del data["last_six_month_d_max_tran_amt"]
    del data["last_one_year_d_tran_amt"]
    bin_d_amt = pd.qcut(data["last_six_month_d_tran_amt"], q=[0.2, 0.4, 0.6, 0.8, 1.], labels=[1, 2, 3, 4])
    data["bin_last_six_month_d_tran_amt"] = bin_d_amt
    del data["last_six_month_d_tran_amt"]
    return data


def dummy_bin_features(data, bin_features):
    """
    :param data:
    :param bin_features: 离散化后的
    :return: one hot bin过的变量
    """
    for bin_feature in bin_features:
        dummy_bin_d_amt = pd.get_dummies(data[bin_feature], prefix=bin_feature)
        data = pd.concat([data, dummy_bin_d_amt], axis=1)
        del data[bin_feature]
    return data


def log_amount_columns(original_data, amt_columns):
    data = original_data.copy()
    for amt_column in amt_columns:
        data[amt_column] = data[amt_column].apply(lambda x: np.log(x+3))
    return data


def standardize_columns(original_data, n_columns):
    data = original_data.copy()
    for column in n_columns:
        data[column] = (data[column]-data[column].mean())/data[column].std()
    return data


def bin_columns(data, columns):
    """
    :param data:
    :param columns: 需要被bin的字段
    :return: 把跟流失直接相关的字段bin,增强变量的意义
    """
    for column in columns:
        bin_column = pd.cut(data[column], [-2, -0.2, -0.1, 0, data[column].max()],
                                      labels=[-2, -1, 0, 1])
        bin_column.name = "bin_" + column
        data = pd.concat([data, bin_column], axis=1)
        del data[column]
    return data


def delete_item(data, delete_columns):
    for column in delete_columns:
        if column in data.columns:
            del data[column]





