from __future__ import print_function
import os

import numpy as np
import tensorflow as tf
from sklearn import metrics

from .cv_lstm_model import PictureTextLSTMConfig, PictureTextLSTM
from .dataprocess import batch_iter, get_train, get_test, get_data, get_split_data

text_embedding_path = 'G:/data_set/len_200_x.npy'
labels_path = 'G:/data_set/y.npy'


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, config.batchSize)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 0.5)
        loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


def train(x_train, y_train, x_val, y_val, num):
    save_dir = 'checkpoints/pekinglstm' + str(num)
    save_path = os.path.join(save_dir, 'best_validation')

    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/pekinglstm'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    #
    # x_train,y_train = get_train(ids,labels)
    # x_val,y_val = get_test(ids,labels)



    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')

    # 总批次
    total_batch = 0
    best_acc_train = 0.0
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 500  # 如果超过1000轮未提升，提前结束训练
    batch_number = 0
    average_train_acc = 0.0
    average_val_acc = 0.0
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)

        batch_train = batch_iter(x_train, y_train, config.batchSize)

        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 0.5
                loss_train, acc_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)
                average_train_acc += acc_train
                average_val_acc += acc_val
                batch_number += 1

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_train = acc_train
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, improved:{5}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, improved_str))

            session.run(model.optimizer, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            msg1 = 'Best train_Acc :{0:>6.2f},Best Val_acc : {1:>6.2f}'
            print(msg1.format(best_acc_train, best_acc_val))
            msg2 = 'Average train_Acc :{0:>7.4f},Average Val_acc : {1:>7.4f}'
            print(msg2.format(average_train_acc / batch_number, average_val_acc / batch_number))
            break


def lstmtest(save_path):
    print("Loading test data...")
    ids = get_data(text_embedding_path)
    labels = get_data(labels_path)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, ids, labels)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = config.batchSize
    data_len = len(ids)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(labels, 1)
    y_pred_cls = np.zeros(shape=len(ids), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: ids[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=['useful', 'useless']))

    # # 混淆矩阵
    # print("Confusion Matrix...")
    # cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    # print(cm)


if __name__ == "__main__":
    # if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
    #     raise ValueError("""usage: python run_rnn.py [train / test]""")
    config = PictureTextLSTMConfig()
    model = PictureTextLSTM(config)
    ids = get_data(text_embedding_path)
    labels = get_data(labels_path)
    datafolders = get_split_data(ids, labels)
    i = 0
    for train_x_index, test_index in datafolders:
        train_x = []
        train_y = []
        val_x = []
        val_y = []

        print("第{:d}折开始训练".format(i))
        # for each in train_x_index:
        #     train_x.append(ids[each])
        #     train_y.append(labels[each])
        # for each in test_index:
        #     val_x.append(ids[each])
        #     val_y.append(labels[each])
        # train_x = np.array(train_x)
        # train_y = np.array(train_y)
        # val_x = np.array(val_x)
        # val_y = np.array(val_y)
        train_x = ids[train_x_index]
        train_y = labels[train_x_index]
        val_x = ids[test_index]
        val_y = labels[test_index]

        train(train_x, train_y, val_x, val_y, i)
        print("第{:d}折训练结束".format(i))
        i += 1
        # if sys.argv[1] == 'train':
        #     train()
        # else:
        #     test()
