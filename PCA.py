import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 读取文本
with open('text', 'r', encoding='utf-8') as file:
    text = file.read()

# 分词
words = jieba.lcut(text)

# 过滤"道"字上下文
context = []
for i in range(len(words)):
    if words[i] == '道':
        left = words[max(0, i-5):i]
        right = words[i+1:min(len(words), i+6)]
        context.append(' '.join(left + ['道'] + right))

# 使用TF-IDF提取特征
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(context)

# PCA分析
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA of "DAO"')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()