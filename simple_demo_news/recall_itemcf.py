"""
@Time : 2021/2/18 3:09 PM 
@Author : Xiaoming
召回 itemcf，根据用户点击序列计算新闻相似度
"""

import math
import os
import signal
from collections import defaultdict
import pickle

import multitasking
import numpy as np
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
log.info('itemcf 召回 ')


def itemcf_sim(train_set):
    """
    计算点击物品之间的相似度，每2个item出现在同一个用户点击序列，即计算该2个item相似得分
    :param: train_set DataFrame 训练集已按时间正序
    :return: {item1_id: {item2_id: score}}
    """
    # 获取用户点击序列
    user_item_df = train_set.groupby('user_id')['click_article_id'].apply(list).reset_index()
    # 用户点击item序列字典 {user_id: [item1_id, item2_id,...]}
    user_item_dict = dict(zip(user_item_df['user_id'], user_item_df['click_article_id']))

    # 记录没个item被点击的次数，用于计算相似得分
    item_count_dict = defaultdict(int)

    # 相似结果{item1_id: {item2_id: score}}
    sim_dict = {}

    print('item cf 计算开始===')
    for user_id, items in tqdm(user_item_dict.items()):
        # 遍历用户点击的item
        for loc1, item in enumerate(items):
            item_count_dict[item] += 1
            sim_dict.setdefault(item, {})

            # 开始计算用户点击item序列内两两相似值
            for loc2, relate_item in enumerate(items):
                # 相同item不用计算相似
                if relate_item == item:
                    continue

                # 2个不同item初始相似值为0
                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重
                # 考虑文章的正向顺序点击和反向顺序点击，相似后出现权重高
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 两个item相距越远，权重越小
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 1 一个用户点击越多item，说明是活跃用户，通过除以math.log(1+len(items))降低权重
                # 2 将所有item和relate_item得分相加，同时出现越多，说明越相似
                sim_dict[item][relate_item] += loc_weight / math.log(1 + len(items))

    # 如果两个item被同时点击次数太多，降低相似得分，防止过热item
    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / math.sqrt(item_count_dict[item] * item_count_dict[relate_item])
    print('item cf 计算结束...')

    return sim_dict, user_item_dict


@multitasking.task
def recall(df_test_part, item_sim, user_item_dict, worker_id):
    """
    生成测试数据集用户的召回数据，根据新闻相似度和用户点击历史数据
    :param df_test_part: 测试集部分数据块
    :param item_sim:
    :param user_item_dict:
    :param worker_id:
    :return:
    """
    data_list = []

    for user_id, item_id in tqdm(df_test_part[['user_id', 'click_article_id']].values):
        rank = {}

        if user_id not in user_item_dict:
            # 如果是新用户（包括被分到测试集中只有一次点击的用户），没有点击历史
            # 新用户没有相似item，需要利用其它策略补全召回
            continue

        interacted_items = user_item_dict[user_id]
        # 选取该用户最后2次点击，查找相似item
        interacted_items = interacted_items[::-1][:2]

        for loc, item in enumerate(interacted_items):
            # 找到item相似最多前200
            for relate_item, wij in sorted(item_sim[item].items(), key=lambda d: d[1], reverse=True)[0:200]:
                # relate_item不能是最近点击的item
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    # interacted_items里边loc越大距离现在更久，降低相似度得分
                    rank[relate_item] += wij * (0.7 ** loc)

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

    os.makedirs('result/itemcf_tmp', exist_ok=True)
    df_part_data.to_csv(f'result/itemcf_tmp/{worker_id}.csv', index=False)
    print(str(worker_id) + 'recall 结束')


if __name__ == '__main__':
    df_train = pd.read_csv('data/my_train_set.csv')
    df_test = pd.read_csv('data/my_test_set.csv')

    if not os.path.isfile('result/itemcf_sim.pkl'):
        # 已计算过相似度，无需再次计算
        item_sim_dict, user_items_dict = itemcf_sim(df_train)

        # 将item相似字典写入文件
        with open('result/itemcf_sim.pkl', 'wb') as f:
            pickle.dump(item_sim_dict, f)
    else:
        # 获取用户点击序列
        user_items_df = df_train.groupby('user_id')['click_article_id'].apply(list).reset_index()
        # 用户点击item序列字典 {user_id: [item1_id, item2_id,...]}
        user_items_dict = dict(zip(user_items_df['user_id'], user_items_df['click_article_id']))
        with open('result/itemcf_sim.pkl', 'rb') as f:
            item_sim_dict = pickle.load(f)

    print('开始进行itemcf召回')

    # 根据进程数对划分召回任务
    n_split = max_threads
    all_users = df_test['user_id'].unique()
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹，该文件夹存放每个进程召回结果
    for path, _, file_list in os.walk('result/itemcf_tmp'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_test[df_test['user_id'].isin(part_users)]
        recall(df_temp, item_sim_dict, user_items_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('result/itemcf_tmp'):
        for file_name in file_list:
            df_temp = pd.read_csv(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)

    # 计算召回指标
    log.info(f'计算召回指标')

    total = df_test.user_id.nunique()

    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_data[df_data['label'].notnull()], total)

    log.debug(
        f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )
    # 保存召回结果
    df_data.to_csv('result/recall_itemcf.csv', index=False)
