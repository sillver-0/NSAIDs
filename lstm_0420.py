#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
review = pd.read_csv("Test_raw_3_9.csv")
review.head()


# 전처리 및 토큰화

# In[2]:


from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#단어의 원형으로 살펴보기 위해 WordNetLemmatizer 사용
lm = WordNetLemmatizer()
retokenize = RegexpTokenizer("[\w]+")
stop_words = set(stopwords.words('english'))
#stop_words에는 '나'가 소문자 i로만 포함되어있어서 대문자 I 추가
stop_words.add('I')
token_list = []

def tokenizer_l(text):
    words =retokenize.tokenize(text)
    token_list = []
    for word in words:
        if word not in stop_words:
            token_list.append(lm.lemmatize(word))
    return token_list


# In[3]:


review['token_review'] = review['review'].apply(tokenizer_l)


# In[4]:


review.head()


# In[5]:


len(review)


# In[7]:


length = []
for k in range(len(review)):
    length.append(len(review['token_review'][k]))
review['token_len'] = length


# In[8]:


review.head()


# In[9]:


tokens = []
for k in range(len(review)):
    a = review['token_review'][k]
    tokens.append(a)


# #gensim 이용해서 word2vec 모델 생성

# In[73]:


import gensim
model = gensim.models.Word2Vec(size = 1000, sg = 1, alpha = 0.025, min_alpha = 0.025, seed = 1234)
model.build_vocab(tokens)


# In[74]:


for epoch in range(30):
    model.train(tokens, total_examples = model.corpus_count, epochs = model.iter)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
    


# In[75]:


model.save('Word2vec0415_I.model')
model.most_similar('side', topn = 20)


# In[2]:


#저장된 word2vec 모델 불러오기
import gensim
model = gensim.models.word2vec.Word2Vec.load('Word2vec0415_I.model')


# #부작용을 통한 라벨링

# In[3]:


file = open('side_effect.txt', mode = 'r', encoding = 'utf-8')
se = []
side = file.read()
side = side.splitlines()
    
for line in side:
    se.append(line)
    file.close()
    
se[:5]


# In[4]:


import re
se_l = []
for word in se:
    word = re.sub('[-=+#/\?:^$.@*\"~&!"]', '', word)
    word = word.lower()
    se_l.append(word)
se_l[:5]


# In[5]:


side_effect = []
content = []
o_count = 0
no_count = 0

for word in se_l:
    try:
        cont = model.most_similar(word, topn = 20)
        side_effect.append(word)
        se_t = []
        se_t.append(word)
        for k in range(20):
            se_t.append(cont[k][0])
        content.append(se_t)
        o_count += 1
    except KeyError:
        no_count += 1
    
print(side_effect[:5])
print(content[:5])
print('count: {}' .format(o_count))
print('no count: {}' .format(no_count))


# In[7]:


print(side_effect)


# In[9]:


print(content[12])
print(content[23])
print(content[40])
print(content[1])
print(content[45])


# In[14]:


import itertools
side_effect_list = list(itertools.chain.from_iterable(content))


# In[15]:


counts = []
for line in tokens:
    count = 0
    for word in line:
        if word in side_effect_list:
            count += 1
    counts.append(count)
print(counts[:5])


# In[16]:


import numpy as np
np.mean(counts)


# In[17]:


np.median(counts)


# In[18]:


from collections import Counter as cc


cnt = cc(counts)
print(cnt.most_common()[:2]) # 상위 2개의 최빈값을 출력한다.


# In[19]:


max(counts)


# In[20]:


review['se_count'] = counts
review.head()


# In[22]:


label = []
for k in range(len(review)):
    if review['token_len'][k] < 13:
        if review['se_count'][k] < int(review['token_len'][k] / 2):
            labeling = 1
        else:
            labeling = 0
    else:
        if review['se_count'][k] < 13:
            labeling = 1
        else:
            labeling = 0
    label.append(labeling)
    
label[:5]


# In[23]:


review['label'] = label
review.head()


# In[24]:


same = 0
diff = 0
for k in range(len(review)):
    if review['rate'][k] == review['label'][k]:
        same += 1
    else:
        diff += 1
        
print(same)
print(diff)


# In[25]:


X_data = []
for sentence in review['review']:
    temp_X = []
    temp_X = tokenizer_l(sentence)
    X_data.append(temp_X)


# In[26]:


y_data = review['label']
print('리뷰의 개수: {}'.format(len(X_data)))
print('레이블의 개수: {}'.format(len(y_data)))


# In[27]:


from tensorflow.keras.preprocessing.text import Tokenizer
max_words = 38000
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(X_data)
sequences = tokenizer.texts_to_sequences(X_data)


# In[28]:


word_to_index = tokenizer.word_index
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key


# In[29]:


vocab_size = len(word_to_index) + 1
print('단어 집합의 크기: {}'.format((vocab_size)))


# In[30]:


n_of_train = int(38022 * 0.8)
n_of_test = int(38022 - n_of_train)
print(n_of_train)
print(n_of_test)


# In[31]:


X_data = sequences
print('리뷰의 최대 길이: %d' % max(len(l) for l in X_data))
print('리뷰의 평균 길이: %f' % (sum(map(len, X_data)) / len(X_data)))


# In[32]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 590 #전체 데이터셋의 길이 590로 맞춤
data = pad_sequences(X_data, maxlen = max_len)
print('data shape: ', data.shape)


# In[33]:


import numpy as np
X_test = data[n_of_train:]
y_test = np.array(y_data[n_of_train:])
X_train = data[:n_of_train]
y_train = np.array(y_data[:n_of_train])


# In[34]:


from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential


# In[35]:


model = Sequential()
model.add(Embedding(27709, 100))
model.add(LSTM(128, dropout = 0.2))
model.add(Dense(1, activation = 'sigmoid'))


# In[36]:


model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(X_train, y_train, epochs = 3, batch_size = 60, validation_split = 0.2)


# In[37]:


print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))


# In[38]:


model_json = model.to_json()
with open('lstm0421.json', 'w') as json_file:
    json_file.write(model_json)


# In[39]:


model.save_weights('lstm0421.h5')
print('saved model to disk')


# In[44]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
epochs = range(1, len(history.history['acc']) + 1)

plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc = 'upper left')
plt.show()


# In[44]:


model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(X_train, y_train, epochs = 5, batch_size = 60, validation_split = 0.2)


# In[45]:


print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))


# In[46]:


model_json = model.to_json()
with open('lstm0418_5.json', 'w') as json_file:
    json_file.write(model_json)


# In[47]:


model.save_weights('lstm0418_5.h5')
print('saved model to disk')


# In[ ]:


#모델 로드하기
from keras.models import model_from_json
json_file = open('lstm0412.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('lstm0412.h5')
print('loaded model from disk')

loaded_model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
score = loaded_model.evaluate(X_test, y_test, verbose = 0)
print('%s: %.2f%%' %(loaded_model.metrics_names[1], score[1] * 100))


# In[40]:


def predict_pn(review):
    tokenizer.fit_on_texts(review)
    medicine = tokenizer.texts_to_sequences(review)
    score = float(model.predict(medicine))
    
    return score


# In[41]:


celecoxib = pd.read_csv('celecoxib.csv')


# In[42]:


sentences_c = []
for sentence in celecoxib['review']:
    temp = []
    temp = tokenizer_l(sentence)
    sentences_c.append(temp)


# In[43]:


pos = 0
neg = 0

for sentence in sentences_c:
    score = predict_pn(sentence)
    
    if (score > 0.7):
        pos += 1
    elif (score > 0.5 and score <= 0.7):
        pos += 0.5
    elif (score <= 0.5 and score > 0.3):
        neg += 0.5
    elif (score <= 0.3):
        neg += 1
all = pos + neg
pos_per = pos / all * 100
neg_per = neg / all * 100

print("celecoxib 긍정확률: {:.2f}%" .format(pos_per))
print("celecoxib 부정확률: {:.2f}%" .format(neg_per))


# In[44]:


naproxen = pd.read_csv('naproxen.csv')


# In[45]:


sentences_n = []
for sentence in naproxen['review']:
    temp = []
    temp = tokenizer_l(sentence)
    sentences_n.append(temp)


# In[46]:


pos = 0
neg = 0

for sentence in sentences_n:
    score = predict_pn(sentence)
    
    if (score > 0.7):
        pos += 1
    elif (score > 0.5 and score <= 0.7):
        pos += 0.5
    elif (score <= 0.5 and score > 0.3):
        neg += 0.5
    elif (score <= 0.3):
        neg += 1
all = pos + neg
pos_per = pos / all * 100
neg_per = neg / all * 100

print("naproxen 긍정확률: {:.2f}%" .format(pos_per))
print("naproxen 부정확률: {:.2f}%" .format(neg_per))


# In[47]:


ibuprofen = pd.read_csv('ibuprofen.csv')


# In[48]:


sentences_i = []
for sentence in ibuprofen['review']:
    temp = []
    temp = tokenizer_l(sentence)
    sentences_i.append(temp)


# In[49]:


pos = 0
neg = 0

for sentence in sentences_i:
    score = predict_pn(sentence)
    
    if (score > 0.7):
        pos += 1
    elif (score > 0.5 and score <= 0.7):
        pos += 0.5
    elif (score <= 0.5 and score > 0.3):
        neg += 0.5
    elif (score <= 0.3):
        neg += 1
all = pos + neg
pos_per = pos / all * 100
neg_per = neg / all * 100

print("ibuprofen 긍정확률: {:.2f}%" .format(pos_per))
print("ibuprofen 부정확률: {:.2f}%" .format(neg_per))


# In[ ]:




