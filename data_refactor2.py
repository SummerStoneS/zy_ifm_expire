"""
@time: 8/28/2018 3:05 PM

@author: 柚子
"""
import numpy as np
import datetime

delete_items = ["client_branch", "client_manager", "manager_education",	"manager_major", "manager_work_years",
                "manager_post_properties", "manager_job_description", "manager_unit", "manager_native_place"]


def delete_columns(data, del_items):
    for col in del_items:
        if col in data.columns:
            del data[col]
    return data


def parse_yymmdd(date):
    if isinstance(date, (np.float64, float)):
        if np.isnan(date):
            return np.nan
        datee = str(int(date))
        datee = datetime.datetime.strptime(datee, '%Y%m%d')
        days = (datetime.datetime(2018, 6, 30)-datee).days
        days /= 365
        return days


def process_ifm_manager(data):
    """
    :param data:
    :return: 处理理财经理，理财经理的名字，定期理财销售占比
    """
    data = delete_columns(data, delete_items)
    data["fix_12m_sum_sale"] /= data["fix_12m_sum_sale"] + data["ifm_12m_sum_sale"] + 1
    data["ifm_12m_sum_sale"] /= data["fix_12m_sum_sale"] + data["ifm_12m_sum_sale"] + 1
    # dummy_branch = pd.get_dummies(data["client_branch"], prefix="branch")
    # # data = pd.concat([data, dummy_branch], axis=1)
    # # # data = data.drop("client_branch", axis=1)
    # # del data["client_branch"]
    # data["client_manager"] = data["client_manager"].apply(lambda x: str(round(x)) if not np.isnan(x) else np.nan)
    data["manager_current_position_time"] = data["manager_current_position_time"].map(parse_yymmdd)
    return data


class HistorydataManager:
    def __init__(self, data, periods):
        self.timeperiods = periods
        self.data = data
        self.items = []

    def cal_amt_over_cnt(self, prefix_tail):
        for item in prefix_tail:
            self.items.extend([(item[0], item[1][0]), (item[0], item[1][1]),
                               (item[0], "_".join([item[1][0], "over", item[1][1]]))])
            for time_period in self.timeperiods:
                amt_name = "_".join([item[0], time_period, item[1][0]])
                cnt_name = "_".join([item[0], time_period, item[1][1]])
                if amt_name in self.data.columns:
                    amt_cnt_name = "_".join([item[0], time_period, item[1][0], "over", item[1][1]])
                    self.data[amt_cnt_name] = self.data[amt_name] / (self.data[cnt_name] + 1)

    def refine_fix_rate_max_min_avg(self):
        methods = ["max", "min", "avg"]
        for method in methods:
            self.items.extend([("_".join(["fix_rate", method]), "")])
            for time_period in self.timeperiods:
                fix_rate_summary = "_".join(["fix_rate", method, time_period])
                self.data[fix_rate_summary] = self.data[fix_rate_summary].apply(lambda x: 0 if x == -2 else x)

            for base_method in methods[methods.index(method)+1:]:
                self.items.extend([("_".join(["fix_rate", method, "over", base_method]), "")])
                for time_period in self.timeperiods:
                    base = "_".join(["fix_rate", base_method, time_period])
                    top = "_".join(["fix_rate", method, time_period])
                    top_over_base = "_".join(["fix_rate", method, "over", base_method, time_period])
                    self.data[top_over_base] = self.data[top] / (self.data[base] + 0.0001)

    def cal_months_rolling(self, new_items):
        self.items.extend(new_items)
        for short_period in self.timeperiods:
            for long_period in self.timeperiods[self.timeperiods.index(short_period)+1:]:
                for col_name in self.items:
                    if col_name[1] != "":
                        a = '_'
                    else:
                        a = ''
                    short_p_col = "_".join([col_name[0], short_period]) + a + col_name[1]
                    long_p_col = "_".join([col_name[0], long_period]) + a + col_name[1]
                    new_col_name = "_".join([col_name[0], short_period, long_period]) + a + col_name[1]
                    if long_p_col in self.data.columns:
                        self.data[new_col_name] = self.data[short_p_col] / (self.data[long_p_col] + 0.0001) - 1

    def replace_null_with_zero(self, prefix_tail):
        for time_period in self.timeperiods:
            name = "_".join([prefix_tail[0], time_period, prefix_tail[1]])
            self.data[name] = self.data[name].apply(lambda x: 0 if x == -2 else x)


def process_history_new_ifm_fix_features(data):
    data["is_hist_risk_prdt"] = data["is_hist_risk_prdt"].replace({"Y": 1, "N": 0})
    data = process_ifm_manager(data)
    processor = HistorydataManager(data, ['1m', '3m', '6m', '12m'])
    processor.replace_null_with_zero(("cur", "compare_aum_rate"))
    processor.cal_amt_over_cnt([["last", ("fix_buy_amt", "fix_buy_cnt")], ["next", ("maturity_bal", "maturity_cnt")]])
    processor.refine_fix_rate_max_min_avg()
    processor.cal_months_rolling([("cur", "compare_aum_rate")])
    return processor.data




