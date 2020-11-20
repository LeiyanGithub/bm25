#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/5/23 16:22
#@Author: leiyan
#@File  : dbconnect.py
#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/5/23 16:03
#@Author: leiyan
#@File  : dbconnect.py

import json
import math
import nltk
import math
import numpy as np
import sqlite3
from nltk.stem import WordNetLemmatizer

class Doc: #文档
    docid = 0 #文档编
    tf = 0 #词频
    len=0 #文档长度
    tf_idf=0.0
    def __init__(self, docid,tf,length,tf_idf):
        self.docid = docid
        self.tf = tf
        self.len = length
        self.tf_idf=tf_idf
    def __repr__(self):
        return(str(self.docid)  +"\t" + str(self.tf)+"\t"+str(self.len) +"\t"+str(self.tf_idf))
    def __str__(self):
        return(str(self.docid)  + "\t" + str(self.tf) +"\t"+str(self.len)+"\t"+str(self.tf_idf))

class store_data:
    def __init__(self):
        self.D = 0 #文档长度
        self.avgdl = 0 #平均长度
        self.num=[] #单词数目
        self.docid={} #文档对应的编号
        self.docs = [] #文档列表
        self.f=[] #频率
        self.tfidf=[] #tf_idf值
        self.df = {}  # 存储每个词及出现了该词的文档数量
        self.idf = {}  # 存储每个词的idf值
        self.postings_lists={}
        self.stop_words=set()
        self.init()

    def init(self):
        f_d = json.load(open("documents.json", 'r', encoding="utf-8", errors="ignore"))
        num = 0
        for key in f_d:
            words=nltk.word_tokenize(f_d[key])
            words = [word.lower() for word in words if word.isalpha()]
            self.docs.append(words)
            self.docid[key] = num
            num += 1
        self.D=num
        self.avgdl=sum([len(doc)+0.0 for doc in self.docs]) / self.D #平均长度
        f = open("stopword.txt","r")
        words = f.read()
        self.stop_words = set(words.split('\n'))
        #print (self.stop_words)

    def compute(self):
        wlem = WordNetLemmatizer()
        doc_id=-1 #文档编号
        for doc in self.docs:
            doc_id=doc_id+1
            tmp = {} #文档中单词对应词频
            number=0 #单词编号
            for word in doc:
                word = word.strip().lower() #变为小写
                word = wlem.lemmatize(word) #词形还原
                if(word not in self.stop_words):
                    #print (word)
                    number=number+1
                    tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.num.append(number) #单词个数
            self.f.append(tmp) #词频
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)
        for index in range(self.D):
            self.tfidf.append({})
            # print (self.f[index])
            for word in self.f[index]:
                tf=self.f[index][word]
                #print(self.f[index][word])
                self.tfidf[index][word]=(1+math.log(tf))*self.idf[word]
            #print (self.tfidf[index])
            sum=0
            for word in self.f[index]:
                sum+=math.pow(self.tfidf[index][word],2)
            sum=math.sqrt(sum)
            for word in self.f[index]:
                self.tfidf[index][word]=self.tfidf[index][word]/sum #归一化

            for key, value in self.f[index].items():
                d = Doc(index,value,self.num[index],self.tfidf[index][word])
                if key in self.postings_lists:  # 该词语已经在其他文档中出现，文档id附加在后面
                    self.postings_lists[key][0]= self.postings_lists[key][0]+1
                    self.postings_lists[key][1].append(d)
                else:
                    self.postings_lists[key] = [1, [d]]


    def store(self, db_path):
        js=json.dumps(self.docid)
        file=open("docid.txt","w")
        file.write(js)
        file.close()
        #print (self.docid)
        f=open("average.txt","w")
        f.write(str(self.D))
        f.write("\n")
        f.write(str(self.avgdl))
        f.close()
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''DROP TABLE IF EXISTS postings''')
        c.execute('''CREATE TABLE postings
                                        (term TEXT PRIMARY KEY NOT NULL ,
                                        idf REAL ,
                                        docs TEXT);''')
        print ("insert")
        for key, value in self.postings_lists.items():
            doc_list = "\n".join(map(str, value[1]))  # 以换行符为分解符,将value中的内容变为字符串
            t = (key, value[0], doc_list)
            c.execute("INSERT INTO postings VALUES (?, ?, ?)", t)
        conn.commit()
        c.execute("SELECT * FROM postings")
        # for row in c.fetchall():
        #     print("word:" + row[0])
        #     print ("df:")
        #     print ((row[1]))
        #     print ("doc_list:\n" + row[2])
        conn.close()

class BM25(object):
    def __init__(self,db_path):
        self.db_path=db_path
        self.D=0
        self.avgdl=0
        self.docid={}
        self.queries=[]
        self.labels=[]
        self.scores=[]
        self.k1 =0.8
        self.b = 0.75
        self.stop_words=set()
        self.init()

    def init(self):
        #获得文档长度、平均长度、停用词
        with open("docid.txt", 'r', encoding='UTF-8') as f:
           self.docid = json.load(f)
        f=open("average.txt","r")
        self.D=int(f.readline())
        self.avgdl=float(f.readline())
        f = open("stopword.txt", "r")
        words = f.read()
        self.stop_words = set(words.split('\n'))

    def read_label(self):
        f = json.load(open("currentdoc.json", 'r'))
        for key in f["queries"].keys():
            words = nltk.word_tokenize(f["queries"][key])
            words = [word.lower() for word in words if word.isalpha()]
            self.queries.append(words)
            label = f["labels"][key]
            label_ = []
            for i in range(len(label)): #得到一个query的所有相关文档
                label_.append(self.docid[str(label[i])])
            self.labels.append(label_) #得到n个query的相关文档
        print("num_que:"+str(len(self.queries)))

    def read_no_label(self):
        f = json.load(open("testset_no_label.json", 'r'))
        for key in f["queries"].keys():
            words = nltk.word_tokenize(f["queries"][key])
            words = [word.lower() for word in words if word.isalpha()]
            self.queries.append(words)
        print("num_que:" + str(len(self.queries)))

    def fetch_from_db(self, term):  # 从数据库中获得term对应的内容
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM postings WHERE term=?', (term,))
        return (c.fetchone())


    def cosin(self,sentence):
        wlem = WordNetLemmatizer()
        score = np.zeros(self.D)
        for word in sentence:  # 在query中的每一个词
            word = word.strip().lower()
            word = wlem.lemmatize(word)
            if (word not in self.stop_words):
                r = self.fetch_from_db(word)  # 得到每个词的表
                if r is None:
                    continue
                terms = r[2].split("\n")
                for term in terms:
                    docid, tf, ld , tf_idf = term.split('\t')
                    docid = int(docid)
                    tf_idf=float(tf_idf)
                    score[docid] = score[docid] + tf_idf
        return score

    def que(self, sentence):  # 得到query
        wlem = WordNetLemmatizer()
        score = np.zeros(self.D)
        for word in sentence: #在query中的每一个词
            word = word.strip().lower()
            word = wlem.lemmatize(word)
            if (word not in self.stop_words):
                r = self.fetch_from_db(word)  # 得到每个词的表
                if r is None:
                    continue
                df = r[1]  # 得到df
                idf=math.log(self.D-df+0.5)-math.log(df+0.5) #计算每个词的idf
                #print("word:"+str(word)+"   df:"+str(df) + "idf:"+str(idf))
                terms = r[2].split("\n")
                for term in terms:
                    docid, tf, ld ,tf_idf = term.split('\t')
                    docid = int(docid)
                    tf = int(tf)
                    ld = int(ld)
                    # print ("docid:"+str(docid))
                     # print ("tf:"+str(tf))
                    # print ("ld:"+str(ld))
                    #计算每个词的分数
                    w= (idf * tf * (self.k1 + 1)
                          / (tf + self.k1 * (1 - self.b + self.b * ld
                                                              / self.avgdl)))
                    # print ("分数："+str(w))
                    score [docid] = score [docid] + w
        return score

    def result_by_bm25(self):
        self.scores=[]
        for query in self.queries:
            score =self.que(query)
            #score = self.cosin(query)
            self.scores.append(score)
        return self.scores
        #print (self.scores)

    def result_by_cosin(self):
        for query in self.queries:
            score = self.cosin(query)
            self.scores.append(score)
        #print (self.scores)

    def MRR(self,k):
        logits=np.array(self.scores)
        target=np.array(self.labels)
        print ("logits:")
        #print (logits)
        print ("target:")
        #print (target)
        #assert logits.shape == target.shape
        print ("indiced_k:")
        indices_k = np.argsort(-logits, 1)[:, :k]  # 取topK 的index   [n, k]
        #print (indices_k)
        reciprocal_rank = 0
        for i in range(indices_k.shape[0]):
          for j in range(indices_k.shape[1]):
              if indices_k[i,j] in target[i]:
                    reciprocal_rank += 1.0 / (j + 1)
                    #print(j)
                    break
        return reciprocal_rank / indices_k.shape[0]

    def NDCG(self,k):
        logits = np.array(self.scores)
        target = np.array(self.labels)
        assert logits.shape[1] >= k, 'NDCG@K cannot be computed, invalid value of K.'

        indices = np.argsort(-logits, 1)  # 沿着文档排序，从大到小
        NDCG = 0
        for i in range(indices.shape[0]):
            DCG_ref = 0
            num_rel_docs = len((target[i])) # target非0数目
            for j in range(indices.shape[1]):
                if j == k:
                    break
                if indices[i, j] in target[i]:
                    DCG_ref += 1 / np.log2(j + 2)
            DCG_gt = 0
            for j in range(num_rel_docs):
                if j == k:
                    break
                DCG_gt += 1 / np.log2(j + 2)
            NDCG += DCG_ref / DCG_gt

        return NDCG / indices.shape[0]


if __name__ == '__main__':
    # s=store_data()
    # s.compute()
    # s.store("stand.db")
    # num = 0.4
    # for i in range(6):
    #     num = num + 0.1
    #     print ("k1:", num)
    #     ss = BM25("stand.db",num)
    #     ss.read_label()
    #     ss.result_by_bm25()
    #     mrr = ss.MRR(10)
    #     print('MRR@10 - ', mrr)
    ss=BM25("stand.db") #读取数据分为两种方法，一种有label，一种没有
                        #计算分数
    # ss.read_label()
    ss.read_no_label()
    # #ss.result_by_cosin()
    scores=ss.result_by_bm25()
    print("111")
    logits = np.array(scores)
    indices = np.argsort(-logits, 1)[:, :10]
    np.save("201700130086.npy", indices)  # 最终提交文件2017xxx.npy（学号命名）
    # mrr = ss.MRR(10)
    # print('MRR@10 - ', mrr)
    # ndcg_10 = ss.NDCG(10)
    # print('NDCG@10 - ', ndcg_10)


