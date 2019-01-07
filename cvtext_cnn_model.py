import tensorflow as tf
import os
class PictureTextCNNConfig(object):
    """LSTM配置参数"""
    def __init__(self):
        self.maxSeqLength = 200
        self.dimensions = 200
        self.batchSize = 24
        self.hidden_dim = 128
        self.num_classes = 2
        self.num_filters = 100
        self.kernel_size1 = 2
        self.kernel_size2 = 3
        self.kernel_size3 = 4
        self.dropout_keep_prob = 0.5
        self.learning_rate = 1e-3
        self.word2vectors = []
        self.num_epochs = 1000
        self.save_per_batch = 20
        self.print_per_batch = 20

class PictureTextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self,config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, shape=[self.config.batchSize, self.config.maxSeqLength,self.config.dimensions], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[self.config.batchSize, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # self.save_dir = 'checkpoints/pekingtextcnn'
        # self.save_path = os.path.join(self.save_dir,'best_validation')
        self.cnn()
    def cnn(self):
        """CNN模型"""
        with tf.device('/gpu:0'):
            embedding_inputs = tf.cast(self.input_x, tf.float32)
        with tf.name_scope("cnn"):
            # CNN layer
            conv1 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size1, name='conv1')
            # global max pooling
            #可以设置多个不同的k_size然后对其进行拼接，tf.concat即可
            gmp1 = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1')

            conv2 = tf.layers.conv1d(embedding_inputs,self.config.num_filters,self.config.kernel_size2,name='conv2')
            gmp2 = tf.reduce_max(conv2,reduction_indices=[1],name = 'gmp2')

            conv3 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size3, name='conv3')
            gmp3 = tf.reduce_max(conv3, reduction_indices=[1], name='gmp3')

            gmp = tf.concat([gmp1,gmp2,gmp3],1)




        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp,self.config.hidden_dim,name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
