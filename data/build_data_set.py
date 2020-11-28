"""
@Time : 2020/11/6 19:30 AM
@Author : 猿小明
通过用户行为数据，构建数据集
"""
import numpy as np
import pandas as pd


def behaviors_sort():
    """
    用行为按照时间正序排序
    :return:
    """
    import time
    df = pd.read_csv('dev/behaviors.tsv', sep='\t', header=None)
    print(df.shape)
    # 时间格式太奇怪，先转换为timestamp，在转为置顶格式时间字符串
    df[5] = df[2].map(lambda x: time.mktime(time.strptime(x, "%m/%d/%Y %I:%M:%S %p")))
    df[6] = df[5].map(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
    new_df = df.sort_values(6, axis=0).drop(5, axis=1)
    new_df.to_csv('dev/behaviors_sort.tsv', sep='\t', index=False)
    print(new_df.shape)


def load_news_data():
    """
    加载新闻向量和id、index字典
    :return:
    """
    news_vec = np.load('dev/dev_news_vec.npy')
    news_id_index = np.load('dev/dev_news_id_index.npy', allow_pickle=True).item()
    return news_vec, news_id_index


def get_user_vec(nid_list, news_vec_matrix, news_id_index):
    """
    根据点击新闻id生成用户向量，平均值
    :param nid_list: 点击新闻id list
    :param news_vec_matrix: 新闻向量矩阵
    :param news_id_index: 新闻id、index字典
    :return:
    """
    user_vec = np.zeros(shape=(128,))
    for nid in nid_list:
        index = news_id_index.get(nid)
        if index:
            user_vec = user_vec + news_vec_matrix[index]
    user_vec = np.divide(user_vec, len(nid_list))
    return user_vec


def build_user_vec_matrix():
    """
    构建用户向量，
    1、用户点击新闻向量的平均值
    2、无点击行为，其它用户的平均值
    :return:
    """
    df = pd.read_csv('dev/behaviors_sort.tsv', sep='\t', header=None)
    df[3] = df[3].fillna('')
    news_click = df.groupby(by=1)[3].apply(lambda x: ' '.join(x))
    print(news_click.shape)

    news_vec_matrix, news_id_index = load_news_data()

    uid_index = {}
    user_index = 0
    user_vec_matrix = []
    for uid, clicks in news_click.iteritems():
        uid_index[uid] = user_index
        user_index += 1
        clicks = clicks.strip()
        if clicks:
            news_ids = clicks.split(' ')
            user = get_user_vec(news_ids, news_vec_matrix, news_id_index)
            user_vec_matrix.append(user)
        else:
            user = np.mean(user_vec_matrix, axis=0)
            user_vec_matrix.append(user)
    np.save('dev/dev_user_vec.npy', user_vec_matrix)
    np.save('dev/dev_uid_index.npy', uid_index)


def build_train_set():
    """
    构建模型应用数据，uid，nid，label
    :return:
    """
    df = pd.read_csv('dev/behaviors_sort.tsv', sep='\t', header=None)
    uids = []
    nids = []
    labels = []
    for row, item in df.iterrows():
        impres = item[4].split(' ')
        for impre in impres:
            nid, label = impre.split('-')
            uids.append(item[1])
            nids.append(nid)
            labels.append(label)

    new_df = pd.DataFrame(data={'uid': uids, 'nid': nids, 'label': labels})
    new_df.to_csv('dev/dev.csv', index=False)


def build_index_train_set():
    """
    构建模型应用数据，uindex，nindex，label
    :return:
    """
    df = pd.read_csv('train/behaviors_sort.tsv', sep='\t', header=None)
    nid_index = np.load('train/train_news_id_index.npy', allow_pickle=True).item()
    uid_index = np.load('train/train_user_id_index.npy', allow_pickle=True).item()
    labels = []
    uindex = []
    nindex = []
    for row, item in df.iterrows():
        impres = item[4].split(' ')
        for impre in impres:
            nid, label = impre.split('-')
            uindex.append(uid_index.get(item[1]))
            nindex.append(nid_index.get(nid))
            labels.append(label)

    new_df = pd.DataFrame(data={'uindex': uindex, 'nindex': nindex, 'label': labels})
    new_df.to_csv('train/train_index.csv', index=False)


# build_index_train_set()

def copy_sample_train():
    """
    正例（点击新闻事件太少）
    复制正例和随机排序训练集
    :return:
    """
    train = pd.read_csv('train/train_index.csv')
    # print(train.groupby(by='flag').count())
    one = train[train['label'] == 1]
    # print(zero.shape)
    new_train = train.append(one, ignore_index=True)
    new_train = new_train.append(one, ignore_index=True)
    new_train = new_train.append(one, ignore_index=True)
    new_train = new_train.append(one, ignore_index=True)
    new_train = new_train.append(one, ignore_index=True)
    new_train = new_train.sample(frac=1)
    new_train.to_csv('train/copy1_5train_index.csv', index=False)
    # print(new_train.shape)
    # print(new_train.groupby('flag').count())

# copy_sample_train()


# 通过tf批量数据载入再转换，无需此方法去生成向量格式数据集
def behavior_train_vec():
    """
    将用户点击和曝光新闻数据集，转换为向量格式，1.9G npy文件：
    x：用户向量+新闻向量，y: 标签（0未点击，1点击）
    :return:
    """
    x_train = []
    y_train = []
    news_vec = np.load('train/train_news_vec.npy')
    nid_index = np.load('train/train_news_id_index.npy', allow_pickle=True).item()
    user_vec = np.load('train/train_user_vec.npy')
    uid_index = np.load('train/train_user_id_index.npy', allow_pickle=True).item()

    df = pd.read_csv('train/train.csv')
    for index, item in df.iterrows():
        y_train.append(item['label'])
        uvec = user_vec[uid_index.get(item['uid'])]
        nvec = news_vec[nid_index.get(item['nid'])]
        x_train.append(np.hstack((uvec, nvec)))
    np.save('train/behavior_train.npy', {'x_train': x_train, 'y_train': y_train})

# behavior_train_vec()

# df = pd.read_csv('train/train.csv')
# print(df.shape)
# print(df[df['label'] == 0].shape)
# print(df[df['label'] == 1].shape)

