"""
@Time : 2020/12/1 19:22 AM
@Author : 猿小明
"""


import os

import numpy as np
import tensorflow as tf
from deep_cross_model import get_model


def train():
    """
    通过train_index.csv，结构为uindex,nindex,label
    预处理数据，构建数据集 uvec+nvec, label
    :return:
    """

    def load_dataset(csv_path, shuffle=True):
        return tf.data.experimental.make_csv_dataset(
            csv_path,
            batch_size=256,  # 设置batch size
            shuffle=shuffle,  # 设置随机排列
            label_name='label',  # 设置标签所在的列名
            na_value='?',
            num_epochs=1,
            ignore_errors=True)

    def load_vec():
        news_vec = np.load('../data_demo/train/train_news_vec.npy')
        user_vec = np.load('../data_demo/train/train_user_vec.npy')
        return tf.constant(news_vec, dtype=tf.float32), tf.constant(user_vec, dtype=tf.float32)

    nvec, uvec = load_vec()

    def embedding(x, y):
        x_news = tf.concat([tf.nn.embedding_lookup(uvec, x['uindex']), tf.nn.embedding_lookup(nvec, x['nindex'])], axis=1)
        x_news = tf.expand_dims(x_news, -1)
        return x_news, y

    train_data = load_dataset('../data_demo/train/copy1_5train_index.csv')
    train_data = train_data.map(embedding)
    dev_data = load_dataset('../data_demo/dev/dev_index.csv')
    dev_data = dev_data.map(embedding)

    model = get_model()

    class_weight = {0: 0.18, 1: 0.82}

    checkpoint_save_path = 'model/deep_cross.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        print('--------------load the model-------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True)

    model.fit(train_data, epochs=5, class_weight=class_weight, validation_data=dev_data, callbacks=cp_callback)

    model.summary()


train()

