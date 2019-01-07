import tensorflow as tf
import os


class PictureTextLSTMConfig(object):
    """LSTM配置参数"""

    def __init__(self):
        self.maxSeqLength = 200
        self.batchSize = 24
        self.num_layers = 2
        self.dimensions = 200
        self.lstmUnits = 128
        self.numClasses = 2
        self.dropout_keep_prob = 0.8
        self.rnn = 'lstm'
        self.output_keep_prob = 0.75
        self.word2vectors = []
        self.num_epochs = 5000
        self.save_per_batch = 20
        self.print_per_batch = 20


class PictureTextLSTM(object):
    """文本分类，LSTM模型"""

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32,
                                      shape=[self.config.batchSize, self.config.maxSeqLength, self.config.dimensions],
                                      name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[self.config.batchSize, self.config.numClasses], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # self.save_dir = 'checkpoints/pekingtextcnn'
        # self.save_path = os.path.join(self.save_dir, 'best_validation')
        # self.weight = tf.Variable(tf.truncated_normal([self.config.lstmUnits, self.config.numClasses]))
        # self.bias = tf.Variable(tf.constant(0.1, shape=[self.config.numClasses]))
        self.lstm()

    def lstm(self):
        """rnn模型"""

        def lstm_cell():  # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.lstmUnits)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.lstmUnits)

        def dropout():  # 为每一个rnn核后面加一个dropout层
            if self.config.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        """词向量映射"""
        with tf.device('/cpu:0'):
            # embedding_inputs = tf.nn.embedding_lookup(self.config.word2vectors, self.input_x)
            # embedding_inputs = tf.cast(embedding_inputs, tf.float32)
            embedding_inputs = tf.cast(self.input_x, tf.float32)
        with tf.name_scope("rnn"):
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            # lstmCell = tf.contrib.rnn.BasicLSTMCell(self.config.lstmUnits)  # lstmUnits为cell中隐藏层神经元的个数
            # lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=self.config.output_keep_prob)
            value, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            value = tf.transpose(value, [1, 0, 2])
            last = tf.gather(value, int(value.get_shape()[0]) - 1)
        """使用双向LSTM进行实验"""
        # with tf.name_scope("Bilstm"):
        #     stacked_rnn_fw = []
        #     for _ in range(self.config.num_layers):
        #         fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.lstmUnits, forget_bias=1.0, state_is_tuple=True)
        #         stacked_rnn_fw.append(fw_cell)
        #     lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
        #     stacked_rnn_bw = []
        #     for _ in range(self.config.num_layers):
        #         bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.lstmUnits, forget_bias=1.0, state_is_tuple=True)
        #         stacked_rnn_bw.append(bw_cell)
        #     lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        #     outputs, outputs_state = tf.nn.bidirectional_dynamic_rnn(
        #         lstm_fw_cell_m, lstm_bw_cell_m, embedding_inputs,
        #         sequence_length=self.config.maxSeqLength,
        #         dtype=tf.float32, time_major=True,
        #     )
        #     lstmoutputs = tf.concat(outputs, 2)
        #     last = lstmoutputs[-1]
        with tf.name_scope("score"):
            fc = tf.layers.dense(last, self.config.lstmUnits, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            self.logits = tf.layers.dense(fc, self.config.numClasses, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            # self.prediction = tf.nn.softmax(tf.matmul(last, self.weight) + self.bias)
        with tf.name_scope("optimize"):

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))  # 计算交叉熵
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)  # 随机梯度下降最小化loss
        with tf.name_scope("accuracy"):
            correctPred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)  # 与标签进行判断
            self.accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
