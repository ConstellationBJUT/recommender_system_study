"""
@Time : 2020/12/3 19:45 AM
@Author : 猿小明
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class Cross(Model):
    """
    cross网络
    """
    def __init__(self, embed_dim, layer_num):
        """
        :param embed_dim: 输入向量长度
        :param layer_num: 网络层数
        """
        super(Cross, self).__init__()
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.ws = []
        self.bs = []
        for i in range(layer_num):
            # 变量加了不同的name，否则保存h5模型会报错
            self.ws.append(tf.Variable(tf.random.truncated_normal(shape=(embed_dim, 1), stddev=0.01)))
            self.bs.append(tf.Variable(tf.zeros(shape=(embed_dim, 1))))

    def call(self, inputs, training=None, mask=None):
        x0 = inputs
        xl = x0
        for w, b in zip(self.ws, self.bs):
            xl_T = tf.reshape(xl, [-1, 1, self.embed_dim])
            xlTw = tf.tensordot(xl_T, w, axes=1)
            xl = x0 * xlTw + b + xl
        return xl


class Deep(Model):
    """
    DNN网络
    """
    def __init__(self, hiddens):
        """
        :param hiddens: 每个隐藏层神经元数
        """
        super(Deep, self).__init__()
        self.hiddens = hiddens
        self.deep_model = tf.keras.models.Sequential()
        for hidden in hiddens:
            self.deep_model.add(Dense(hidden, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))

    def call(self, inputs, training=None, mask=None):
        return self.deep_model(inputs)


class DeepCross(Model):
    """
    cross+deep+lr网络
    """
    def __init__(self, embed_dim, cross_layer_num, deep_hiddens):
        super(DeepCross, self).__init__()
        self.cross = Cross(embed_dim, cross_layer_num)
        self.deep = Deep(deep_hiddens)
        self.f1 = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2())
        self.embed_dim = embed_dim
        self.cross_layer_num = cross_layer_num
        self.deep_hiddens = deep_hiddens

    def call(self, inputs):
        xc = self.cross(inputs)
        xc = tf.squeeze(xc, axis=2)
        xd = tf.squeeze(inputs, axis=2)
        xd = self.deep(xd)
        y = self.f1(tf.concat([xc, xd], axis=-1))
        return y


class DeepCrossTest(Model):
    """
    手动创建，每多一层，手动添加一层
    """
    def __init__(self, embed_dim):
        super(DeepCrossTest, self).__init__()
        self.embed_dim = embed_dim
        self.w0 = tf.Variable(tf.random.truncated_normal(shape=(embed_dim, 1), stddev=0.01))
        self.b0 = tf.Variable(tf.zeros(shape=(embed_dim, 1)))
        self.w1 = tf.Variable(tf.random.truncated_normal(shape=(embed_dim, 1), stddev=0.01))
        self.b1 = tf.Variable(tf.zeros(shape=(embed_dim, 1)))

        self.f1 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.f2 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())

        self.f3 = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x0 = inputs
        x0_T = tf.reshape(x0, [-1, 1, self.embed_dim])  # (None,1,256)
        x0Tw0 = tf.tensordot(x0_T, self.w0, axes=1)  # (None,1)
        x1 = x0*x0Tw0 + self.b0 + x0
        x1_T = tf.reshape(x1, [-1, 1, self.embed_dim])  # (None,1,256)
        x1Tw1 = tf.tensordot(x1_T, self.w1, axes=1)  # (None,1)
        x2 = x0 * x1Tw1 + self.b1 + x1
        x2 = tf.squeeze(x2, axis=2)
        x = tf.squeeze(inputs, axis=2)
        x = self.f1(x)
        x = self.f2(x)
        y = tf.concat([x2, x], axis=-1)
        y = self.f3(y)
        return y


def get_model():
    model = DeepCross(256, 2, [64, 64])
    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC'])
    return model
