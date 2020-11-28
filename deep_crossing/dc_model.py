"""
@Time : 2020/11/4 22:42 PM
@Author : 猿小明
"""
import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Activation
from tensorflow.keras import Model


def dc1():
    """
    神经网络二分类
    通过train.csv，uid、nid、label
    需要加载事先构建训练向量npy文件1.9G，结构为（uvec+nvec）,label
    内存消耗高，加载数据慢
    :return:
    """
    data = np.load('../data_demo/train/behavior_train.npy', allow_pickle=True).item()
    raw_train_data = data['x_train']
    raw_train_label = data['y_train']

    train_data = tf.data.Dataset.from_tensor_slices((raw_train_data, raw_train_label))

    train_data = train_data.shuffle(256).batch(128)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    class_weight = {0: 1/811630 * 846377/2, 1: 1/34747 * 846377/2}

    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC'])

    model.fit(train_data, epochs=20, class_weight=class_weight)

    model.summary()


def dc2():
    """
    盛景网络二分类，添加验证集，随训练输出结果
    通过train_index.csv，结构为uindex,nindex,label
    需要加载事先构建训练向量npy文件1.9G，结构为（uvec+nvec）,label
    内存消耗高，加载数据慢
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
        return tf.concat([tf.nn.embedding_lookup(uvec, x['uindex']), tf.nn.embedding_lookup(nvec, x['nindex'])],
                         axis=1), y

    train_data = load_dataset('../data_demo/train/copy1_5train_index.csv')
    train_data = train_data.map(embedding)
    dev_data = load_dataset('../data_demo/dev/dev_index.csv')
    dev_data = dev_data.map(embedding)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    class_weight = {0: 0.18, 1: 0.82}

    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC'])

    checkpoint_save_path = 'model/dc.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        print('--------------load the model-------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=False)

    model.fit(train_data, epochs=5, class_weight=class_weight, validation_data=dev_data, callbacks=cp_callback)

    model.summary()


# dc2()


class ResnetBlock(Model):
    """
    Resnet基础单元
    """
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv1D(filters, 3, strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv1D(filters, 3, strides=strides, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        if residual_path:
            self.down_c1 = Conv1D(filters, 1, strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs

        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)
        # print(y.shape, residual.shape)
        out = self.a2(y + residual)
        return out


class Resnet18(Model):
    """
    Resnet网路，initial_filters=256是论文结构，这里单机训练困难
    """
    def __init__(self, block_list, initial_filters=4):
        super(Resnet18, self).__init__()
        self.num_block = len(block_list)
        self.block_list = block_list
        self.out_filters = initial_filters

        self.c1 = Conv1D(self.out_filters, 3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id != 0 and layer_id == 0:
                    block = ResnetBlock(self.out_filters, strides=1, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)
            self.out_filters *= 2
        self.p1 = tf.keras.layers.GlobalAveragePooling1D()
        self.f1 = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


def dc3():
    """
    Resnet，加载数据后再进行数据批量向量化，节约内存
    通过train_index.csv，结构为uindex,nindex,label
    批量预处理数据，构建数据集 uvec+nvec, label
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

    model = Resnet18([2, 2, 2])

    class_weight = {0: 0.18, 1: 0.82}

    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC'])

    model.fit(train_data, epochs=20, class_weight=class_weight, validation_data=dev_data)

    model.summary()


# dc3()
