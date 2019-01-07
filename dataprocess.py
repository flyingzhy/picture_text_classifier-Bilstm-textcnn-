
import numpy as np
from sklearn.model_selection import StratifiedKFold

def get_data(filepath):
    matrix = np.load(filepath)
    return matrix
def get_train(text_embeddings,labels):
    x_train = text_embeddings[:4900]
    y_train = labels[:4900]
    return x_train,y_train
def get_test(text_embeddings,labels):
    x_test = text_embeddings[4900:]
    y_test = labels[4900:]
    return x_test,y_test
def batch_iter(x, y, batch_size):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    # num_batch = 5000
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = (i + 1) * batch_size
        if end_id <= data_len:
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
        else:
            break
def get_split_data(text_embeddings,labels):
    stratified_folder = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    batch_data = stratified_folder.split(text_embeddings, np.zeros(shape=(labels.shape[0], 1)))
    return batch_data


