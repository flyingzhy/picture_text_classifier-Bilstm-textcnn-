# picture_text_classifier
利用textcnn和双向LSTM实现文本和图片数据的融合
数据预处理步骤为：先将图片中的实体同构ImageAI识别出来，然后寻找其词向量与文本得到的词向量进行拼接。
实验数据及处理过程在这里不予展示。
特别参考：https://github.com/gaussic/text-classification-cnn-rnn