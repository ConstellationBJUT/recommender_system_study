"""
@Time : 2021/2/18 11:06 AM 
@Author : 猿小明
数据处理：根据用户点击日志，构建训练集和测试集
测试集：随机选出部分用户，每个用户最后一次点击
训练集：测试集用户去除最后一次点击 + 其他用户点击日志
说明：如果用户只有一条点击数据，该用户在训练集中未出现
"""

import os
import pandas as pd
from utils_log import Logger

# 日志输出文件
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info(f'数据处理: data_process')


def data_split(click_log):
    """
    生成训练集df_train: 训练集数据sample5万数据内每个用户点击去除最后一次 + testB数据
    生成测试集df_test: 训练集数据sample5万数据内每个用户最后一次点击 + testB每个用户（点击新闻id是-1）
    :param click_log: 点击日志,DataFrame
    :return:
    """
    # 清理重复数据
    click_log = click_log.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    # 按照用户和点击时间升序排列
    click_log = click_log.sort_values(by=['user_id', 'click_timestamp'])
    total_users = click_log['user_id'].drop_duplicates()
    log.debug(f'total_users num: {total_users.shape}')
    # 随机采样出一部分样本
    test_users = total_users.sample(50000)
    log.debug(f'test_users num: {test_users.shape}')

    # 构建测试集，测试用户最后一次点击
    df_test = click_log.groupby('user_id').tail(1)
    df_test = df_test[df_test['user_id'].isin(test_users)]

    # 构建训练集，原日志去除测试集内容
    df_train = click_log.append(df_test).drop_duplicates(keep=False)

    # 将训练和测试集写入文件，用于后续操作
    df_train.to_csv('data/my_train_set.csv', index=False)
    df_test.to_csv('data/my_test_set.csv', index=False)


if __name__ == '__main__':
    df_train_click = pd.read_csv('data/train_click_log.csv')
    df_test_click = pd.read_csv('data/testA_click_log.csv')
    # 合并所有点击日志
    df_click_log = df_train_click.append(df_test_click)
    data_split(df_click_log)
