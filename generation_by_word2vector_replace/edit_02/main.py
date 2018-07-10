import os
import json
import pickle
import itertools
import argparse
import numpy as np
from gensim.models import Word2Vec


def parse_args():
    # 创建参数解析对象
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_keywords', type=int, default=2, help='the number of keywords')
    parser.add_argument('--most_similarity', type=int, default=5, help='most_similarity')
    parser.add_argument('--file_path', type=str, default="./corpus/total/total_segment.txt", help='filepath of corpus')
    return parser.parse_args()


def main(args):
    # 选择生成文本类型
    print("请选择生成文本的类型：")
    print("女频：")
    print("0.农家种田    1.古言穿越    2.女尊天下    3.幻想言情    4.总裁霸爱    5.情深虐恋    6.浪漫青春    7.现代言情    8.存爱同人    9.蜜恋宠文    10.重生虐渣")
    print("男频：")
    print(
        "11.修真异能  12.修真狂人   13.兵王传奇   14.异世奇遇   15.悬疑灵异   16.摸骨神医   17.架空历史   18.武侠仙侠   19.游戏竞技   20.热血爽文   21.玄幻魔幻   22.科幻末日   23.都市小说")

    # ---------------------------------------------------------------------
    with open("./category.json", "r", encoding="utf-8") as f:
        category = json.load(f)
    category_choose = input("请输入类别编号：")

    # word2vector model list
    word2vector_dir = []
    for root, dirs, files in os.walk("./word2vector"):
        for name in files:
            if ".npy" not in os.path.join(root, name):
                word2vector_dir.append(os.path.join(root, name))

    # corpus list
    corpus_dir = []
    for root, dirs, files in os.walk("./corpus"):
        for name in files:
            corpus_dir.append(os.path.join(root, name))

    for item in word2vector_dir:
        if category[category_choose] in item:
            word2vector_path = item

    for item in corpus_dir:
        if category[category_choose] in item:
            corpus_path = item

    # 加载word2vector模型
    print("模型加载中......")
    model = Word2Vec.load(word2vector_path)

    # ----------------------------------------------------------------------
    # 输入关键词
    string = []
    for i in range(args.num_keywords):
        temp = input("输入第" + str(i + 1) + "个关键词：")
        while temp not in model.wv.vocab:
            print("请换个关键词试试.....")
            temp = input("输入第" + str(i + 1) + "个关键词：")
        string.append(temp)

    # 通过word2vector寻找相关检索词
    def find_similaritr_words(string):
        index_words = []
        for i in range(len(string)):
            try:
                similarity_words = model.most_similar([string[i]])[:args.most_similarity]
                temp = [item[0] for item in similarity_words]
                temp.append(string[i])
                index_words.append(temp)
            except:
                print("抱歉，没有关键词：", string[i])
        return index_words

    index_words = find_similaritr_words(string)

    # print('相关检索词： ',index_words)

    # 相关句子检索
    def find_similarity_sentence(corpus_path, index_words):
        for ii in range(len(index_words)):
            for temp in itertools.combinations(index_words, len(index_words) - ii):
                print("------------------------------------------------------------------------------------")
                print('相关检索词： ', temp)
                print("------------------------------------------------------------------------------------")
                # 相关句子检索
                matched_sentence = []
                with open(corpus_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = list(line.strip().split(" "))
                        if len(line) <= 100:  # 去掉长度大于100的句子
                            match_num = 0
                            for item in temp:
                                if len(set(item) & set(line)) > 0:
                                    match_num += 1
                            if match_num == len(index_words) - ii:
                                for i in index_words:
                                    for j in i:
                                        if j in line:
                                            line[line.index(j)] = i[-1]
                                print("".join(line))
                                if line not in matched_sentence:
                                    matched_sentence.append(line)
                if len(matched_sentence) != 0:
                    return matched_sentence

    matched_sentence = find_similarity_sentence(corpus_path, index_words)

    # 按照相似度排序
    print("-------------------------------------------------------------------------------------------")
    print("相关度排序中......")
    print("-------------------------------------------------------------------------------------------")

    # top_n 排序
    def top_n(matched_sentence, n=5):
        matched_similarity = []
        for i in range(len(matched_sentence)):
            temp = []
            for w in matched_sentence[i]:
                if w in model:
                    temp.append(w)

            matched_similarity.append(model.n_similarity(string, temp))
        ordered_matched = [matched_sentence[i] for i in np.argsort(-np.array(matched_similarity))]
        if len(ordered_matched) > n:
            ordered_matched_top_n = ordered_matched[0:n]
        else:
            ordered_matched_top_n = ordered_matched
        return ordered_matched_top_n

    ordered_matched_top_n = top_n(matched_sentence, 5)
    for item in ordered_matched_top_n:
        print("".join(item))
    print("===============================================================================================")


# ===========================================================================================================

if __name__ == '__main__':
    main(parse_args())