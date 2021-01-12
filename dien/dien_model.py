"""
@Time : 2020/12/30 19:20 AM
@Author : 猿小明
"""
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Layer, ReLU
from tensorflow.keras import Model


class Attention(Model):
    """
    cross网络
    """
    def __init__(self, hiddens):
        """
        :param hiddens: DNN隐藏层各层节点数
        """
        super(Attention, self).__init__()
        self.deep_model = tf.keras.models.Sequential()
        # self.deep_model.add(tf.keras.layers.BatchNormalization())
        for hidden in hiddens:
            self.deep_model.add(Dense(hidden, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))
        self.deep_model.add(Dense(1, activation='sigmoid'))

    def call(self, inputs, click_news):
        user_click = inputs
        news = click_news
        result = []
        for i in range(user_click.shape[1]):
            if inputs.shape[-1] == click_news.shape[-1]:
                sub = news - user_click[:, i, :]
                input_emb = tf.concat([user_click[:, i, :], sub, news, user_click[:, i, :]*news], axis=-1)
            else:
                input_emb = tf.concat([user_click[:, i, :], news], axis=-1)
            ai = self.deep_model(input_emb)
            result.append(ai)
        return result


class Bilinear(Layer):
    def __init__(self, units):
        super(Bilinear, self).__init__()
        self.linear_act = Dense(units, activation=None, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2())
        self.linear_noact = Dense(units, activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, a, b, gate_b=None):
        if gate_b is None:
            return tf.keras.activations.sigmoid(self.linear_act(a) + self.linear_noact(b))
        else:
            return tf.keras.activations.tanh(self.linear_act(a) + tf.math.multiply(gate_b, self.linear_noact(b)))


class AUGRU(Layer):
    def __init__(self, units):
        super(AUGRU, self).__init__()

        self.u_gate = Bilinear(units)
        self.r_gate = Bilinear(units)
        self.c_memo = Bilinear(units)

    def call(self, inputs, state, att_score):
        u = self.u_gate(inputs, state)
        r = self.r_gate(inputs, state)
        c = self.c_memo(inputs, state, r)

        u_ = att_score * u
        final = (1 - u_) * state + u_ * c

        return final


class DIEN(Model):
    def __init__(self, units, hiddens):
        super(DIEN, self).__init__()
        self.gru = GRU(units, return_sequences=True)
        self.attention = Attention([128, 64])
        self.augru = AUGRU(units)
        self.deep_model = tf.keras.models.Sequential()
        self.deep_model.add(tf.keras.layers.BatchNormalization())
        for hidden in hiddens:
            self.deep_model.add(Dense(hidden, activation=None, kernel_regularizer=tf.keras.regularizers.l2()))
            self.deep_model.add(ReLU(negative_slope=0.0003))
        self.f = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        # GRU网络，获得每个点击新闻向量对应gru单元向量
        hs = self.gru(inputs[:, 0:-1, :])
        # att_inputs = tf.concat([hs, inputs[:, -1:, :]], axis=1)
        # gru输出向量与预测新闻向量送入attention，获取对应权重
        ats = self.attention(hs, inputs[:, -1, :])
        h0 = tf.zeros_like(hs[:, 0, :])
        # 送入augru（与gru单元数相同），获得最后一个输出
        for i in range(hs.shape[1]):
            h0 = self.augru(hs[:, i, :], h0, ats[i])
        deep_input = tf.concat([h0, inputs[:, -1, :]], axis=-1)
        # 拼接最后输出与预测新闻向量，送入dnn
        x = self.deep_model(deep_input)
        y = self.f(x)
        return y


def get_model():
    units = 128
    model = DIEN(units, [200, 80])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC'])
    return model
