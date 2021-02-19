"""
@Time : 2021/2/19 3:30 PM 
@Author : Xiaoming
根据item被点击次数进行热门召回
"""

import os
import signal
from collections import defaultdict

import multitasking
import pandas as pd
from tqdm import tqdm

from utils_log import Logger
from utils_evaluate import evaluate

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

# 初始化日志
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info('hot 召回 ')


@multitasking.task
def recall(df_test_part, hot_items, user_item_dict, worker_id):
    """
    生成测试数据集用户的召回数据，根据新闻相似度和用户点击历史数据
    :param df_test_part: 测试集部分数据块
    :param hot_items: 热门item列表
    :param user_item_dict:
    :param worker_id:
    :return:
    """
    data_list = []

    for user_id, item_id in tqdm(df_test_part[['user_id', 'click_article_id']].values):
        rank = {}

        if user_id not in user_item_dict:
            # 如果是新用户（包括被分到测试集中只有一次点击的用户），没有点击历史
            # 最热的前100作为召回
            df_new_user = pd.DataFrame()
            df_new_user['article_id'] = hot_items[:100]
            df_new_user['sim_score'] = [1/(score+1) for score in range(100)]
            df_new_user['user_id'] = user_id
            df_new_user['label'] = 0
            data_list.append(df_new_user)
            continue

        interacted_items = user_item_dict[user_id]
        # 选取该用户最后10次点击，查找相似item
        interacted_items = interacted_items[::-1][:10]

        # 找到item相似最多前100
        for loc, relate_item in enumerate(hot_items[:100]):
            # relate_item不能是最近点击的item
            if relate_item not in interacted_items:
                rank.setdefault(relate_item, 0)
                # 热门列表越靠后分值越小
                rank[relate_item] = 1/(loc+1)

        # 重新计算相似度得分后，取前100（可能不到100个）
        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_part = pd.DataFrame()
        df_part['article_id'] = item_ids
        df_part['sim_score'] = item_sim_scores
        df_part['user_id'] = user_id
        df_part['label'] = 0
        df_part.loc[df_part['article_id'] == item_id, 'label'] = 1

        data_list.append(df_part)

    df_part_data = pd.concat(data_list, sort=False)

    os.makedirs('result/hot_tmp', exist_ok=True)
    df_part_data.to_csv(f'result/hot_tmp/{worker_id}.csv', index=False)
    print(str(worker_id) + 'hot 结束')


if __name__ == '__main__':
    df_train = pd.read_csv('data/my_train_set.csv')
    df_test = pd.read_csv('data/my_test_set.csv')

    # 所有item一起计算
    all_click_df = df_train.append(df_test)
    hot_items_list = all_click_df['click_article_id'].value_counts().keys().tolist()

    # 用户历史点击
    user_items_df = df_train.groupby('user_id')['click_article_id'].apply(list).reset_index()
    user_items_dict = dict(zip(user_items_df['user_id'], user_items_df['click_article_id']))

    print('开始进行hot召回')

    # 根据进程数对划分召回任务
    n_split = max_threads
    all_users = df_test['user_id'].unique()
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹，该文件夹存放每个进程召回结果
    for path, _, file_list in os.walk('result/hot_tmp'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_test[df_test['user_id'].isin(part_users)]
        recall(df_temp, hot_items_list, user_items_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('result/hot_tmp'):
        for file_name in file_list:
            df_temp = pd.read_csv(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)

    # 计算召回指标
    total = df_test.user_id.nunique()

    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_data[df_data['label'].notnull()], total)

    log.debug(
        f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )
    # 保存召回结果
    df_data.to_csv('result/recall_hot.csv', index=False)
