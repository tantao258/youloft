import argparse
from pyhanlp import *
from random import sample
from gensim.models import Word2Vec


def parse_args():
    # 创建参数解析对象
    parser = argparse.ArgumentParser()
    parser.add_argument('--most_similarity', type=int,
                        default=5,
                        help='most_similarity')
    parser.add_argument('--sentence', type=str,
                        default="它是国内首家专业会员制电商平台，获得了全球顶级投资机构的投资。开通会员后，全部享受成本价购物",
                        help='replace sentence')
    parser.add_argument('--sentence_method', type=str,
                        default="HanLP_segment",
                        help='sentence_method:NLP_segment or HanLP_segment')
    parser.add_argument('--word2vector_model_path', type=str,
                        default="E:/Pycharm/wiki_word2vecter/model/wiki_word2vectoe.model",
                        help='word2vector_model_path')
    return parser.parse_args()


def main(args):

    # =========================================加载word2vector模型===========================================
    print("Loading word2vector model from：{}".format(args.word2vector_model_path))
    model = Word2Vec.load(args.word2vector_model_path)
    # model = gensim.models.KeyedVectors.load_word2vec_format(args.word2vector_model_path, binary=True)
    print("Please input sentence you wan to transform!"), exit() if len(args.sentence) == 0 else print("object is generating......")

    # =======================================分词 并 进行词性标注==============================================
    string = []    # [('杨杏儿', 'nr'), ('含', 'v'), ('着', 'uzhe'), ('眼泪', 'n'), ('咬了一口', 'i'), ('，', 'w'), ('往', 'p')]

    if args.sentence_method == "HanLP_segment":
        for term in HanLP.segment(args.sentence):
            string.append((term.word, str(term.nature)))
        print(string)

    elif args.sentence_method == "NLP_segment":
        NLP_Tokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
        str1 = str(NLP_Tokenizer.analyze(args.sentence))
        for item in str1.split():
            string.append((item.split("/")[0], item.split("/")[1]))
        print(string)

# =========================================文本生成================================================\
    replace = {
        "a": "形容词",
        "nr": "人名",
        "ns": "地名",
        "n": "名词",
        "z": "状态词",
        "nnt": "",
        "nv": "",
        "nf": "",
        "t": "时间",
    }

    replaced_sentence = []
    for item in string:
        if item[1] not in replace or "n" not in item[1]:   # 包含上述词性  或者 词性包含"n"
            replaced_sentence.append(item[0])

        else:
            # 词典中有这个词
            # ----------------------------------------------------------------------------
            if item[0] in model.wv.vocab:
                candidate = []
                for words in model.most_similar([item[0]]):
                    for seg in HanLP.segment(words[0]):
                        if str(seg.nature) == "item[1]":
                            candidate.append(seg.word)
                        else:
                            candidate.append(seg.word)
                # print(candidate)
                # 从候选列表中随机选择词性相同的词语
                replaced_sentence.append(sample(candidate, 1)[0])
            # ----------------------------------------------------------------------------

            else:
                # 词典中没有的词语 --> 不变
                replaced_sentence.append(item[0])

    # trans list to string
    new_sentence = "".join(replaced_sentence)
    print("="*200)
    print("|".join([item[0] for item in string]))   # 原始句子分词
    print("|".join(replaced_sentence))
    print("=" * 200)
    print(new_sentence)


if __name__ == '__main__':
    main(parse_args())