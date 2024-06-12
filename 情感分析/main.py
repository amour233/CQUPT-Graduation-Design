import joblib
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# 加载微博数据
data = pd.read_csv('data/weibo_senti_100k.csv')


# 预处理所有文本
preprocessed_texts = data['review']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(preprocessed_texts, data['label'], test_size=0.2, random_state=77)

# 使用TF-IDF向量化文本
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 使用SVM进行分类
clf = SVC(kernel='linear')
clf.fit(X_train_tfidf, y_train)

# 预测测试集
y_pred = clf.predict(X_test_tfidf)

# 计算准确率和F1值
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
joblib.dump(clf, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')