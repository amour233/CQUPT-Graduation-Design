import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import jieba
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# 1. 数据预处理
# 读取数据集
df = pd.read_csv('data/weibo_senti_100k.csv')

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

# 读取停用词
with open('data/hit_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f.readlines()]

# 2. 文本向量化
# 分词处理
X_train_word_list = [[word for word in jieba.cut(sentence) if word not in stopwords] for sentence in X_train]
X_test_word_list = [[word for word in jieba.cut(sentence) if word not in stopwords] for sentence in X_test]

# 训练word2vec模型
word2vec = Word2Vec(X_train_word_list, vector_size=100, window=5, min_count=1, sg=0)

# 将句子转换为向量
def sentence_to_vector(sentence, model):
    words = [word for word in sentence if word in model.wv]
    if not words:
        return np.zeros(model.vector_size)  # 如果句子中没有单词在词汇表中，则返回零向量
    return np.mean(model.wv[words], axis=0)  # 对句子中的单词向量取平均

train_vectors = [sentence_to_vector(sentence, word2vec) for sentence in X_train_word_list]
test_vectors = [sentence_to_vector(sentence, word2vec) for sentence in X_test_word_list]

# 3. 模型训练
svm_model = SVC(kernel='linear')
svm_model.fit(train_vectors, y_train)

# 4. 模型评估
svm_pred = svm_model.predict(test_vectors)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_confusion_matrix = confusion_matrix(y_test, svm_pred)

print("SVM 混淆矩阵:")
print(svm_confusion_matrix)
print("SVM Accuracy:", svm_accuracy, "F1 Score:", svm_f1, "Recall:", svm_recall)