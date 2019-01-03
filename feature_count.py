# -*- coding: utf-8 -*
import codecs
import numpy as np
import jieba
import pickle


def get_feature_word():
    # 切分feature文件用于生成关键字
    feature_file = codecs.open('feature200.txt', 'r', 'utf-8')
    content = feature_file.read()
    feature_file.close()

    segments = []
    segs = jieba.cut(content)

    # 保存feature关键字
    feature_word = []
    for seg in segs:
        if seg != '\n' and seg != '\t' and seg != ' ':
            feature_word.append(seg)

    return feature_word


class FeatureCount:
    feature_count = {}

    def __init__(self, feature_word):
        # 初始化每个关键字出现次数为0
        print(len(feature_word))
        for i in range(len(feature_word)):
            self.feature_count[feature_word[i]] = 0
        print(len(self.feature_count))

    def clear(self):
        for feature in self.feature_count:
            self.feature_count[feature] = 0


def get_input_feature_from_one_section(section_name, fc):
    # 切分一部书的内容，同时统计上面的每个关键字出现的次数
    section_file = codecs.open('GCD_TXT\\' + section_name + '.txt', 'r', 'utf-8')
    content = section_file.read()
    section_file.close()

    segments = []
    segs = jieba.cut(content)

    # 统计每五千个字中每个关键字出现次数
    input_feature = []
    # 用于计算是否到达一万字
    i = 0
    j = 0
    # 用于保存每五千字文本
    c = ''
    for seg in segs:
        c += seg
        j += 1
        if seg != '\n' and seg != '\t' and seg != ' ':
            if seg in fc.feature_count:
                fc.feature_count[seg] += 1
            i += 1

        if i % 5000 == 0 or seg == object():
            input_feature.append(list(fc.feature_count.values()))

            # 保存这五千字中关键词出现次数
            output = codecs.open('GCD_count\\'+section_name+'_'+str(i/5000)+'_feature_count.txt', 'w', 'utf-8')
            for feature in fc.feature_count:
                output.write(feature+'\t'+str(fc.feature_count[feature])+'\n')
            fc.clear()

            # 保持一万字文本，这里制表符和换行符没有计算在内
            output = codecs.open('GCD_segment\\'+section_name+'_'+str(i/5000)+'.txt', 'w', 'utf-8')
            output.write(c)
            c = ''

    print(section_name, i, j, len(input_feature))

    return input_feature


feature_word = get_feature_word()
fc = FeatureCount(feature_word)
sections = ['1_1', '1_2', '1_3', '1_4',
            '2_1', '2_2', '2_3', '2_4',
            '3_1', '3_2', '3_3', '3_4',
    '4_1', '4_2', '4_3', '4_4', '4_5', '4_6',
    '5_1', '5_2']

others = [
    ]




input_features = {}
for i in range(len(sections)):
    input_features[i] = get_input_feature_from_one_section(sections[i], fc)


# 将逻辑回归输入先保存下来
input_file = open('input_features.bin', 'wb')
s = pickle.dumps(input_features)
input_file.write(s)
input_file.close()
