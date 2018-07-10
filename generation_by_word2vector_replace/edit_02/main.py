import os
import json
import pickle
import jieba
import itertools
import argparse
from random import sample
import numpy as np
from gensim.models import Word2Vec


def parse_args():
    # 创建参数解析对象
    parser = argparse.ArgumentParser()
    parser.add_argument('--most_similarity', type=int,
                        default=5,
                        help='most_similarity')
    parser.add_argument('--sentence', type=str,
                        default="杨桃儿握着柿饼子拉着她就往后院去，躲过院子里人的视线，拿起柿饼子踮着脚尖就往杨杏儿嘴巴里塞：“姐姐吃。”",
                        help='replace sentence')
    parser.add_argument('--word2vector_model_path', type=str,
                        default="./word2vector/word2vec_wx",
                        help='word2vector_model_path')
    return parser.parse_args()


def main(args):

    # define dot set
    dot = {"，", "。", "？", "！", "：", "：“", "。”", "？”", "！”", "；", "”", "“"}
    # 加载word2vector模型
    print("Loaded word2vector model from：{}".format(args.word2vector_model_path))
    model = Word2Vec.load(args.word2vector_model_path)

    # ---------------- get similar word by word2vec for each word  ---------------
    print("Please input sentence you wan to transform!"), exit()if len(args.sentence) == 0 else print("object is generating......")

    string = [i for i in jieba.cut(args.sentence)]
    similarity_words_list = []
    for word in string:
        if word not in dot:
            try:
                similarity_words = model.most_similar([word])[:args.most_similarity]
                similarity_words_list.append(similarity_words)
            except:
                similarity_words_list.append([word])
        else:
            similarity_words_list.append([word])

    # ----------------  create new sentence by similar words  ----------------------
    new_sentence_list = []
    for i in range(len(similarity_words_list)):
        temp = sample(similarity_words_list[i], 1)[0][0]
        if temp in dot:
            new_sentence_list.append(string[i])
        else:
            new_sentence_list.append(temp)
    # trans list to string
    new_sentence = "".join(new_sentence_list)
    print("="*200)
    print("|".join(string))
    print("|".join(new_sentence_list))
    print("=" * 200)
    print(new_sentence)


if __name__ == '__main__':
    main(parse_args())