"""
@time: 8/23/2018 10:39 PM

@author: 柚子
"""
import os
from data_refactor import *
from variables import params
from churn_define import tag_churn
from data_refactor2 import process_history_new_ifm_fix_features


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


def cal_churn_y(time_period='20170401', method=tag_churn, kwargs={}):
    """
    :param time_period:
    :param method: give a way of how to define actual actual true using 9month aum data
    :param kwargs: params of churn definition function
    :return: 定义流失标签 actual churn，跟features合并
    """
    y_filename = time_period + "07.csv"
    y_file_path = os.path.join(folder,  'data', time_period, y_filename)
    data_y = pd.read_csv(open(y_file_path), index_col="client_no")
    method(data_y, **kwargs)
    return data_y


def transform_and_reshape_features(data_x_y):
    """
    :param data_x_y:
    :return: 属性规约，转换，计算新meaningful变量
    """
    data_x_y = tag_new_customer(data_x_y, new_customer_base_year=2017)         # 标记新员工
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


def prepare_x_features_and_y_churn(folder_name, time_period, churn_func=tag_churn, churn_func_kwargs={}):

    # load 原始X；define churn y
    data_X = get_source_features_data(time_period=time_period)
    churn_y = cal_churn_y(time_period=time_period, method=churn_func, kwargs=churn_func_kwargs)
    source_data = pd.concat([data_X, churn_y["is_churn"]], axis=1)
    source_data = source_data[source_data["is_churn"].notnull()]
    data_file = time_period + ".h5"
    source_data.to_hdf(os.path.join(folder_name, data_file), key="source")

    # 对变量做删除、 变换、create new features...
    transformed_x_and_y = transform_and_reshape_features(source_data)
    modeling_data = transformed_x_and_y[params["all_variables_X_y"]]
    modeling_data.to_hdf(os.path.join(folder_name, data_file), key="preRF")
    return modeling_data


def get_source_data(time_period='20171231'):
    """
    :param time_period: aum6个月平均大于5万的高端客群，6个月的终止月份，or 预测的起始月份
    :return: 新增的feature 和 高端老客防流失里的features
    """
    data = pd.DataFrame()
    for i in range(1, 7):
        filename = time_period + "0" + str(i) + ".csv"
        file_path = os.path.join(folder, "data", time_period, 'new', filename)
        data_i = pd.read_csv(open(file_path), index_col="client_no")
        print(len(data_i), len(data_i.index.unique()))
        data = pd.concat([data, data_i], axis=1)

    old_data = pd.read_hdf(os.path.join(folder, "data", time_period, time_period + ".h5"), key="preRF")
    combine_data = pd.merge(old_data, data, how='left', left_index=True, right_index=True)

    return combine_data

folder = 'zhumadian'                  # 每次换试点的省修改
if __name__ == '__main__':

    time_periods = ["20180930", "20180630", "20180331", "20171231", "20170930"]         # 预测月份 8.31预测未来一个月是不是会downsale

    # 处理高潜downsale标签
    for time_period in time_periods:
        folder_name = os.path.join(folder, "data", time_period)
        prepare_x_features_and_y_churn(folder_name, time_period, churn_func=tag_churn,
                                       churn_func_kwargs={"old_month_num": 3})          # predict all drop
    #     prepare_x_features_and_y_churn(folder_name, time_period, churn_func=tag_is_churn,
    #                                    churn_func_kwargs={"old_month_num": 3})          # predict sharp drop

    # 处理理财模型新增标签
    all_periods_data = pd.DataFrame()
    for time_period in time_periods:
        new_data = get_source_data(time_period=time_period)
        processed_new_data = process_history_new_ifm_fix_features(new_data)
        processed_new_data.describe().T.to_excel(os.path.join(folder, "data", time_period,
                                                              "preRF_variables_describe.xlsx"))
        processed_new_data.to_hdf(os.path.join(folder, "data", time_period, "new_data.h5"), key="preRF")



