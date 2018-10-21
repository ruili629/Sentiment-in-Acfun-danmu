
# coding: utf-8

# # 一、SnowNLP

# In[1]:

with open("allacfundanmu_1.txt", encoding = "utf-8") as f:
    data = f.read().splitlines()
danmulist = []
for eachline in data:
    oneobs = eachline.split('\t')
    if len(oneobs) == 11:
        danmulist.append(oneobs[9])
from snownlp import SnowNLP
s = SnowNLP(danmulist[0])
print(s.words)
print(s.sentiments)


# In[4]:

from snownlp import SnowNLP
import numpy as np
import time
start = time.clock()
#long running
def sentclass(text):
    s = SnowNLP(text)
    return s.sentiments
vsentclass = np.vectorize(sentclass)
sentilist = vsentclass(danmulist)
print(len(sentilist))
print(sentilist[0:10])
end = time.clock()
print("runtime: %f s" % (end - start))
#would take 491s to run


# In[6]:

# from snownlp import SnowNLP
# import time
# start = time.clock()
# #long running
# sentilist = []
# for dan in danmulist:
#     s = SnowNLP(dan)
#     sentilist.append(s.sentiments)
# print(len(sentilist))
# print(sentilist[0:10])
# end = time.clock()
# print("runtime: %f s" % (end - start))
# #would take 581s to run


# In[13]:

def zero_one(sent):
    if sent > 0.5:
        return 1
    else:
        return 0
vzero_one = np.vectorize(zero_one)
senti = vzero_one(sentilist)
# senti[0:10]


# # 二、情感词典

# In[89]:

import jieba
import numpy as np

#打开词典文件，返回列表
def open_dict(Dict = 'hahah', path=r'/'):
    path = path + '%s.txt' % Dict
    dictionary = open(path, 'r', encoding='utf-8') #py3
    #dictionary = open(path, 'r')
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict

def judgeodd(num):
    if (num % 2) == 0:
        return 'even'
    else:
        return 'odd'

deny_word = open_dict(Dict = 'notdoc', path= r'third_examp/')
posdict = open_dict(Dict = 'positive', path= r'third_examp/')
negdict = open_dict(Dict = 'negative', path= r'third_examp/')

degree_word = open_dict(Dict = 'degreedoc', path= r'third_examp/')
mostdict = degree_word[degree_word.index('extreme')+1 : degree_word.index('very')]#权重4，即在情感词前乘以4
verydict = degree_word[degree_word.index('very')+1 : degree_word.index('more')]#权重3
moredict = degree_word[degree_word.index('more')+1 : degree_word.index('ish')]#权重2
ishdict = degree_word[degree_word.index('ish')+1 : degree_word.index('last')]#权重0.5

# def print_basics(dictl):
#     print(type(dictl))
#     print(len(dictl))
#     print(dictl[0])
#     print(dictl[1])
# print_basics(degree_word)
# print_basics(mostdict)
# print_basics(verydict)
# print_basics(moredict)
# print_basics(ishdict)
def sentiment_score_list(dataset):
    seg_sentence = dataset.split('。')
    count1 = []
    count2 = []
    for sen in seg_sentence: #循环遍历每一个评论
        segtmp = jieba.lcut(sen, cut_all=False)  #把句子进行分词，以列表的形式返回
        i = 0 #记录扫描到的词的位置
        a = 0 #记录情感词的位置
        poscount = 0 #积极词的第一次分值
        poscount2 = 0 #积极词反转后的分值
        poscount3 = 0 #积极词的最后分值（包括叹号的分值）
        negcount = 0
        negcount2 = 0
        negcount3 = 0
        for word in segtmp:
            if word in posdict:  # 判断词语是否是情感词
                poscount += 1
                c = 0
                for w in segtmp[a:i]:  # 扫描情感词前的程度词
                    if w in mostdict:
                        poscount *= 4.0
                    elif w in verydict:
                        poscount *= 3.0
                    elif w in moredict:
                        poscount *= 2.0
                    elif w in ishdict:
                        poscount *= 0.5
                    elif w in deny_word:
                        c += 1
                if judgeodd(c) == 'odd':  # 扫描情感词前的否定词数
                    poscount *= -1.0
                    poscount2 += poscount
                    poscount = 0
                    poscount3 = poscount + poscount2 + poscount3
                    poscount2 = 0
                else:
                    poscount3 = poscount + poscount2 + poscount3
                    poscount = 0
                a = i + 1  # 情感词的位置变化

            elif word in negdict:  # 消极情感的分析，与上面一致
                negcount += 1
                d = 0
                for w in segtmp[a:i]:
                    if w in mostdict:
                        negcount *= 4.0
                    elif w in verydict:
                        negcount *= 3.0
                    elif w in moredict:
                        negcount *= 2.0
                    elif w in ishdict:
                        negcount *= 0.5
                    elif w in degree_word:
                        d += 1
                if judgeodd(d) == 'odd':
                    negcount *= -1.0
                    negcount2 += negcount
                    negcount = 0
                    negcount3 = negcount + negcount2 + negcount3
                    negcount2 = 0
                else:
                    negcount3 = negcount + negcount2 + negcount3
                    negcount = 0
                a = i + 1
            elif word == '！' or word == '!':  ##判断句子是否有感叹号
                for w2 in segtmp[::-1]:  # 扫描感叹号前的情感词，发现后权值+2，然后退出循环
                    if w2 in posdict or negdict:
                        poscount3 += 2
                        negcount3 += 2
                        break
            i += 1 # 扫描词位置前移

            # 以下是防止出现负数的情况
            pos_count = 0
            neg_count = 0
            if poscount3 < 0 and negcount3 > 0:
                neg_count += negcount3 - poscount3
                pos_count = 0
            elif negcount3 < 0 and poscount3 > 0:
                pos_count = poscount3 - negcount3
                neg_count = 0
            elif poscount3 < 0 and negcount3 < 0:
                neg_count = -poscount3
                pos_count = -negcount3
            else:
                pos_count = poscount3
                neg_count = negcount3

            count1.append([pos_count, neg_count])
        count2.append(count1)
        count1 = []
    return count2

def sentiment_score(senti_score_list):
    score = []
    for review in senti_score_list:
        score_array = np.array(review)
        Pos = np.sum(score_array[:, 0])
        Neg = np.sum(score_array[:, 1])
        AvgPos = np.mean(score_array[:, 0])
        AvgPos = float('%.1f'%AvgPos)
        AvgNeg = np.mean(score_array[:, 1])
        AvgNeg = float('%.1f'%AvgNeg)
        StdPos = np.std(score_array[:, 0])
        StdPos = float('%.1f'%StdPos)
        StdNeg = np.std(score_array[:, 1])
        StdNeg = float('%.1f'%StdNeg)
        #score.append([Pos, Neg, AvgPos, AvgNeg, StdPos, StdNeg])
        score.append([Pos, Neg])

    return score


# In[38]:

danmulist[0:2]


# In[92]:

# data = '你就是个王八蛋，混账玩意!你们的手机真不好用！非常生气，我非常郁闷！！！！'
# data2= '我好开心啊，非常非常非常高兴！今天我得了一百分，我很兴奋开心，愉快，开心'
# sli = []
# for da in [data,data2]:
#     daa = sentiment_score(sentiment_score_list(da))[0]
#     print(sentiment_score(sentiment_score_list(da))[0])
#     sli.append(daa)
# sentilist2 = []
# for da in danmulist[3]:
#     daa = sentiment_score(sentiment_score_list(da))[0]
#     sentilist2.append(daa)
#     print(daa)


# In[44]:

sentilist2 = []
for da in danmulist[0:2]:
    daa = sentiment_score(sentiment_score_list(da))[0]
    sentilist2.append(daa)


# In[26]:

def dict_classif_list(data):
    scli = sentiment_score(sentiment_score_list(data))
    return scli
# vdict_classif_list = np.vectorize(dict_classif_list)
def dict_classif(data):
    scli = sentiment_score(sentiment_score_list(data))
    if scli[0][0] > scli[0][1]:
        return 1
    else:
        return 0
# dict_classif(data2)
# vdict_classif = np.vectorize(dict_classif)


# In[95]:

#generate sentilist2
start = time.clock()
#long running
# sentilist2 = vdict_classif_list(danmulist)  #fail to do this
sentilist2 = []
for da in danmulist:
    try:
        daa = sentiment_score(sentiment_score_list(da))[0]
        sentilist2.append(daa)
    except:
        daa = [0.0,0.0]
        sentilist2.append(daa)
print(len(sentilist2))
print(sentilist2[0:10])
end = time.clock()
print("runtime: %f s" % (end - start))
#would take 481s to run


# In[96]:

#generate senti2
start = time.clock()
#long running
# senti2 = vdict_classif(danmulist)
senti2 = []
for da in danmulist:
    try:
        daa = dict_classif(da)
        senti2.append(daa)
    except:
        daa = 0
        senti2.append(daa)
print(len(senti2))
print(senti2[0:10])
end = time.clock()
print("runtime: %f s" % (end - start))
#would take 676s to run


# In[97]:

#观察1
da = '我好开心啊，非常非常非常高兴！今天我得了一百分，我很兴奋开心，愉快，开心'
# da = danmulist[24]
print(da)
daa = sentiment_score(sentiment_score_list(da))[0]
print(daa)
s = SnowNLP(da)
print(s.words)
print(s.sentiments)


# In[90]:

#观察2
# da = danmulist[23]
# print(da)
# daa = sentiment_score(sentiment_score_list(da))[0]
# print(daa)
# s = SnowNLP(da)
# print(s.words)
# print(s.sentiments)


# In[84]:

#观察3
# da = danmulist[19]
# print(da)
# daa = sentiment_score(sentiment_score_list(da))[0]
# print(daa)
# s = SnowNLP(da)
# print(s.words)
# print(s.sentiments)


# # 三、

# In[15]:

#videocode, danmulist, sentitlist, sentilist2, senti, senti2
print(len(danmulist))
print(len(sentilist))
print(len(senti))
print(len(sentilist2))
print(len(senti2))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



