import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# 加载停用词列表
with open('data/sogou_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f.readlines()])

# 加载训练好的模型和TF-IDF向量化器
clf = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# 加载麒麟9000s评论数据
kiwi_data = pd.read_csv('data/微博评论_麒麟9000s.csv', encoding='gbk')

# 分词和去除停用词
# 分词和去除停用词
def preprocess_text(text):
    # 确保文本是字符串类型
    if not isinstance(text, str):
        return ''  # 如果不是字符串，返回空字符串
    words = jieba.lcut(text)
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

# 预处理所有文本
preprocessed_texts = kiwi_data['评论内容'].apply(preprocess_text)
# print(preprocessed_texts)
# 使用之前训练好的TF-IDF向量化器进行向量化
X_kiwi_tfidf = vectorizer.transform(preprocessed_texts)

# 使用训练好的SVM模型进行预测
kiwi_predictions = clf.predict(X_kiwi_tfidf)

# 将预测结果添加到原始CSV文件的最后一列
kiwi_data['label'] = kiwi_predictions

# 保存更新后的CSV文件
kiwi_data.to_csv('kiwi_9000s_comments_with_labels.csv', index=False, encoding='utf-8')

