import jieba
import gensim
from gensim import corpora
import pyLDAvis.gensim_models
import webbrowser
import os

# 读取文本
with open('text', 'r', encoding='utf-8') as file:
    text = file.read()

#jieba分词
words = jieba.lcut(text)

# 去掉停用词，防止无关词语影响结果
stopwords = set(['的', '了', '是', '在', '和', '有', '就', '不', '人', '都'])
filtered_words = [word for word in words if word not in stopwords and len(word) > 1]

# 构建词袋模型
dictionary = corpora.Dictionary([filtered_words])
corpus = [dictionary.doc2bow(filtered_words)]

# 训练LDA模型
num_topics = 5  # 主题数量可以调整
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# 输出每个主题的关键词
for idx, topic in lda_model.print_topics(-1):
    print(f"主题 {idx}： {topic}")