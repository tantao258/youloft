import os
import re
import jieba
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing


process_data=False
train_word2vector=True

#========================================================================文本批处理==========================================================
if process_data:
    rootDir="corpus"
    filepath="/data/books/"
    for dir in os.listdir(filepath):
        if not os.path.exists(os.path.join(rootDir, dir)):
            os.makedirs(os.path.join(rootDir, dir))
        if dir=='女频':
            continue
        for item in os.listdir(os.path.join(filepath, dir)):
            if not os.path.exists(os.path.join(rootDir, dir, item)):
                os.makedirs(os.path.join(rootDir, dir, item))
            with open(os.path.join(rootDir, dir, item, item+".txt"), 'w', encoding='utf-8') as ff:
                counter=0
                for fpath, dirs, fs in os.walk(os.path.join(filepath, dir, item)):
                    for f in fs:
                        try:
                            with open(os.path.join(fpath,f),"r",encoding="utf-8") as f:
                                for line in f:
                                    if len(line)<=15 or len(line)>120:
                                        continue
                                    line = re.sub("\s", "", line)  # 去掉空格(也去掉了换行符)
                                    line="  ".join(jieba.cut(line))
                                    ff.write(line)
                                    ff.write("\n")
                                    counter+=1
                                    if counter %10000==0:
                                        print(dir, item, counter)
                        except:
                            print(fpath, f)
                                
#========================================================================词向量批量训练=======================================================
if train_word2vector:
    filepath="./corpus"
    for dir in os.listdir(filepath):
        if dir=="total":
            continue
        if not os.path.exists(os.path.join("word2vector", dir)):
            os.makedirs(os.path.join("word2vector", dir))
        
        for item in os.listdir(os.path.join(filepath, dir)):
            
            if not os.path.exists(os.path.join("word2vector", dir, item)):
                os.makedirs(os.path.join("word2vector", dir, item))
            else:
                continue
            print(item, "开始训练词向量......")
            model = Word2Vec(LineSentence(os.path.join(filepath, dir, item, item+".txt")), 
                             size=256, 
                             window=5, 
                             min_count=1,
                             workers=50, 
                             iter=10)
            model.save(os.path.join("word2vector", dir, item, "word2vector.model"))
            print(item,"词向量训练完成，模型保存到：","/word2vector/",dir,"/",item,"/","word2vector.model")