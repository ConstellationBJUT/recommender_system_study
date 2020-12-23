"""
@Time : 2020/12/9 4:11 PM 
@Author : 猿小明
din 构建自有的train和dev数据集
"""
import pandas as pd
import numpy as np


def add0_2_news():
    """
    在news_vec第一个位置添加0向量
    在news_id_index字典添加，N0,0
    :return:
    """
    news_vec = np.load('../data_demo/dev/dev_news_vec.npy')
    n0 = np.zeros_like(news_vec[0])
    news_vec1 = np.insert(news_vec, 0, values=n0, axis=0)
    np.save('dev/dev_news_vec.npy', news_vec1)

    news_id_index = np.load('../data_demo/dev/dev_news_id_index.npy', allow_pickle=True).item()
    nids = []
    nindexs = []
    nids.append('N0')
    nindexs.append(0)
    for nid, nindex in news_id_index.items():
        nids.append(nid)
        nindexs.append(nindex + 1)
        if nindex == 0:
            print(nid)
    np.save('dev/dev_news_id_index.npy', dict(zip(nids, nindexs)))

# add0_2_news()


def build_index_data_set():
    """
    构建数据，(uid, n1, n2, n3, n4, n5, nindex, label)
    (n1~n5历史点击新闻index列表，不足补0，nindex预测新闻index，点击标签)
    :return:
    """
    df = pd.read_csv('../data_demo/dev/behaviors_sort.tsv', sep='\t', header=None)
    df[3] = df[3].fillna('')
    nid_index = np.load('dev/dev_news_id_index.npy', allow_pickle=True).item()
    uids = []
    labels = []
    nindex = []
    n1s = []
    n2s = []
    n3s = []
    n4s = []
    n5s = []
    ns = [n1s, n2s, n3s, n4s, n5s]
    for row_num, item in df.iterrows():
        # 曝光和点击
        impres = item[4].split(' ')
        for impre in impres:
            uids.append(item[1])
            nid, label = impre.split('-')
            nindex.append(nid_index.get(nid))
            labels.append(label)

            # 点击历史记录最多取5个
            clicks = str(item[3]).strip()
            if clicks:
                news_ids = clicks.split(' ')
                news_ids = news_ids[-5:]
                click_num = len(news_ids)
                for i in range(click_num):
                    ns[i].append(nid_index.get(news_ids[i]))
                for i_ex in range(5 - click_num):
                    ns[5 - i_ex - 1].append(0)
            else:
                for nn in ns:
                    nn.append(0)

    new_df = pd.DataFrame(data={'uid': uids, 'n1': n1s, 'n2': n2s,
                                'n3': n3s, 'n4': n4s, 'n5': n5s,
                                'nindex': nindex, 'label': labels})
    new_df = new_df.sample(frac=1)
    new_df.to_csv('dev/dev_index.csv', index=False)


# build_index_data_set()


def copy_sample_train():
    """
    正例（点击新闻事件太少）
    复制正例和随机排序训练集
    :return:
    """
    train = pd.read_csv('train/train_index.csv')
    one = train[train['label'] == 1]
    new_train = train.append(one, ignore_index=True)
    new_train = new_train.append(one, ignore_index=True)
    new_train = new_train.append(one, ignore_index=True)
    new_train = new_train.append(one, ignore_index=True)
    new_train = new_train.append(one, ignore_index=True)
    new_train = new_train.sample(frac=1)
    new_train.to_csv('train/copy1_5train_index.csv', index=False)
    print(new_train.shape)
    print(new_train.groupby('label').count())

# copy_sample_train()

