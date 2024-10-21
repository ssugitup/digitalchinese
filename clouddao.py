import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# 读取文本
with open('text', 'r', encoding='utf-8') as f:
    text = f.read()

# 使用jieba进行中文分词
words = jieba.cut(text)

# 过滤出与"道"相关的词汇
related_words = []
for word in words:
    if '道' in word and word != '道':
        related_words.append(word)

# 统计关联词汇的频率
word_counts = Counter(related_words)

# 生成词云
wordcloud = WordCloud(
    font_path='LXGWWenKaiTC-Bold.ttf',  # 使用黑体字体显示中文
    width=800,
    height=400,
    background_color='white'
).generate_from_frequencies(word_counts)

# 显示词云
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()