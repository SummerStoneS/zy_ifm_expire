"""
@time: 8/23/2018 10:39 PM

@author: 柚子
"""
import os
from data_refactor import *
from variables import params
from data_refactor2 import process_history_new_ifm_fix_features, parse_yymmdd


def get_source_features_data(time_period='20180831'):
    """
    :param time_period: aum6个月平均大于5万的高端客群，6个月的起始月份
    :return: 最原始的features
    """
    data = pd.DataFrame()
    file_list = list(range(1, 11))
    file_list.pop(6)            # 07 file is aum used for churn definition
    for i in file_list:
        filename = time_period + "0" + str(i) + ".csv"
        file_path = os.path.join(folder, 'data', time_period, filename)
        data_i = pd.read_csv(open(file_path), index_col="client_no")
        print(len(data_i), len(data_i.index.unique()))
        data = pd.concat([data, data_i], axis=1)
    delete_item(data, params["delete_items"])
    return data


def transform_and_reshape_features(data_x_y):
    """
    :param data_x_y:
    :return: 属性规约，转换，计算新meaningful变量
    """
    data_x_y = tag_new_customer(data_x_y, new_customer_base_year=2018)         # 标记新员工
    convert_yesno_to_num(data_x_y, params["yesno_columns"])      # "Y","N" to 1,0
    numcolumns_to_str(data_x_y, params["toStringVariables"])
    # id_processor = IdProcessor(data_x_y)
    # data_with_age_and_gender = id_processor.id_to_gender_age()

    data_with_age_and_gender = transform_age_gender(data_x_y)
    personinfo_finished = process_basic_info(data_with_age_and_gender, params["customerType"])
    personinfo_finished.describe().T.to_csv(os.path.join(folder_name, "describe_initial_data.csv"))

    channel_analysis(personinfo_finished)                                      # 渠道偏好分析
    outflow_cross_bank_and_consume(personinfo_finished)                        # 跨行转账次数and消费（通过交易渠道来）
    calculated_columns = process_historical_transaction(personinfo_finished)
    combine_data = pd.concat([personinfo_finished, calculated_columns], axis=1)
    combine_data = fine_tune_steps(combine_data)       # 最后调优的
    combine_data = dummy_bin_features(combine_data, bin_features=["bin_last_six_month_d_tran_amt"])
    combine_data.describe().T.to_excel(os.path.join(folder_name, "all_available_features_describe.xlsx"))
    return combine_data


def prepare_x_features_and_y_churn(folder_name, time_period):

    # load 原始X；define churn y
    data_X = get_source_features_data(time_period=time_period)
    data_file = time_period + ".h5"
    data_X.to_hdf(os.path.join(folder_name, data_file), key="source")

    # 对变量做删除、 变换、create new features...
    transformed_x_and_y = transform_and_reshape_features(data_X)
    modeling_data = transformed_x_and_y[params["all_variables_X_y"]]
    modeling_data.to_hdf(os.path.join(folder_name, data_file), key="preRF")
    return modeling_data


def get_source_data(time_period):
    """
    :param time_period: aum6个月平均大于5万的高端客群，6个月的终止月份，or 预测的起始月份
    :return: 新增的feature 和 高端老客防流失里的features
    """
    data = pd.DataFrame()
    for i in range(0, 7):
        filename = time_period + "0" + str(i) + ".csv"
        file_path = os.path.join(folder, "data", time_period, 'new', filename)
        data_i = pd.read_csv(open(file_path), index_col="client_no")
        print(len(data_i), len(data_i.index.unique()))
        data = pd.concat([data, data_i], axis=1)
    data = data[data["end_date"].notnull()]
    old_data = pd.read_hdf(os.path.join(folder, "data", time_period, time_period + ".h5"), key="preRF")
    combine_data = pd.merge(old_data, data, how='left', left_index=True, right_index=True)
    combine_data.loc[:, "expire_date_from_now"] = combine_data["end_date"].apply(lambda x: parse_yymmdd(x, period=time_period))
    return combine_data


folder = 'zhumadian'                  # 每次换试点的省修改
if __name__ == '__main__':

    time_periods = ["20180907", "20180914", "20180815", "20180719", "20180608", "20180507", "20180405"]
    # 如果是预测数据，flag="predict"
    flag = "predict"
    predict_periods = ["20181029"]
    if flag == "predict":
        time_periods = predict_periods

    # 处理高潜downsale标签
    for time_period in time_periods:
        folder_name = os.path.join(folder, "data", time_period)
        prepare_x_features_and_y_churn(folder_name, time_period)

    # 处理理财模型新增标签
    all_periods_data = pd.DataFrame()
    for time_period in time_periods:
        new_data = get_source_data(time_period=time_period)
        processed_new_data = process_history_new_ifm_fix_features(new_data)
        if flag != "predict":
            processed_new_data = processed_new_data[params["features_before_y_define"]]
        else:
            processed_new_data = processed_new_data[params["all_features"]]
        processed_new_data.describe().T.to_excel(os.path.join(folder, "data", time_period,
                                                              "preRF_variables_describe.xlsx"))
        processed_new_data.to_hdf(os.path.join(folder, "data", time_period, "new_data.h5"), key="preRF")



