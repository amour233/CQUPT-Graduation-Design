import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
import jieba
# 加载微博数据
data = pd.read_csv('data/weibo_senti_100k.csv')

# 读取停用词
with open('data/sogou_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f.readlines()]

# 预处理所有文本
preprocessed_texts = data['review']
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(preprocessed_texts, data['label'], test_size=0.2, random_state=42)

# 分词处理
X_train_word_list = [[word for word in jieba.cut(sentence) if word not in stopwords] for sentence in X_train]
X_test_word_list = [[word for word in jieba.cut(sentence) if word not in stopwords] for sentence in X_test]

# 使用TF-IDF向量化文本
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 使用SVM模型进行训练和验证
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
svm_preds = svm_model.predict(X_test_tfidf)
svm_accuracy = accuracy_score(y_test, svm_preds)
svm_f1 = f1_score(y_test, svm_preds)
svm_recall = recall_score(y_test, svm_preds)

# 使用朴素贝叶斯模型进行训练和验证
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, nb_preds)
nb_f1 = f1_score(y_test, nb_preds)
nb_recall = recall_score(y_test, nb_preds)

# 使用逻辑回归模型进行训练和验证
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)
lr_accuracy = accuracy_score(y_test, lr_preds)
lr_f1 = f1_score(y_test, lr_preds)
lr_recall = recall_score(y_test, lr_preds)

print("SVM模型准确率:", svm_accuracy)
print("SVM模型F1值:", svm_f1)
print("SVM模型召回率:", svm_recall)
print("朴素贝叶斯模型准确率:", nb_accuracy)
print("朴素贝叶斯模型F1值:", nb_f1)
print("朴素贝叶斯模型召回率:", nb_recall)
print("逻辑回归模型准确率:", lr_accuracy)
print("逻辑回归模型F1值:", lr_f1)
print("逻辑回归模型召回率:", lr_recall)
