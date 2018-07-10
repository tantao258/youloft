import os
import re
import jieba
import argparse
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

process_data = False
train_word2vector = False
# ========================================================================文本批处理==========================================================
if process_data:
    with open("./corpus/clean-八零年代纯女户奋斗史.txt", 'w', encoding='utf-8') as f:
        counter = 0
        with open("./corpus/八零年代纯女户奋斗史.txt", "r", encoding="utf-8") as ff:
            for line in ff:
                # if len(line) <= 15 or len(line) > 120:
                #     continue
                line = re.sub("\s", "", line)  # 去掉空格(也去掉了换行符)
                line="  ".join(jieba.cut(line))
                f.write(line)
                f.write("\n")
                counter += 1
                if counter % 10000 == 0:
                    print(counter)
                                
# ========================================================================词向量批量训练=======================================================
if train_word2vector:
        path = "./corpus/clean-八零年代纯女户奋斗史.txt"
        print("Begin train wordvector from {}".format(path))
        model = Word2Vec(LineSentence(path),
                         size=256,
                         window=5,
                         min_count=1,
                         workers=50,
                         iter=10)
        model.save(os.path.join("./word2vector", "word2vector.model"))
        print("wordvector training completed, and model have been saved to: {}"
              .format(os.path.join("./word2vector", "word2vector.model")))

if __name__ == "__main__":

    from gensim.models import Word2Vec
    model_path = "./word2vector/word2vector.model"
    print("="*60)
    print("Load word2vector model from: {}".format(model_path))
    model = Word2Vec.load(model_path)
    print(model)
    print("=" * 60)
    # ----------------------------相关词---------------------------------
    query = "车祸"
    try:
        print(model.most_similar(query))
    except:
        print("该词不存在")


