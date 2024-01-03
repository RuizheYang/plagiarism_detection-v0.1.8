# encoding: utf-8
"""
@author: Shen Xinyi
@contact: xinyi_shen@shannonai.com
@file: match_plagiarism.py
@time: 12/27/19 7:26 PM
@desc:
"""

import numpy as np
from typing import List


class SentenceInfo(object):
    """
    记录句子特征的类
    """

    def __init__(self, start: int, end: int, sentence_idx: int, para_idx: int, tfidf_feature: np.ndarray = None,
                 raw_text: str = ""):
        """
        :param start: 句子开头在文档中的位置
        :param end: 句子结尾在文档中的位置,左闭右开
        :param sentence_idx: 句子在文档中的索引(第几个句子)
        :param para_idx: 句子在文档中的第几个段落
        :param tfidf_feature: 句子的 tfidf feature
        raw_text: str，原句
        """
        self.start = start
        self.end = end
        self.sentence_idx = sentence_idx
        self.para_idx = para_idx
        self.tfidf_feature = tfidf_feature
        self.raw_text = raw_text

    def set_tfidf_feature(self, tfidf_feature: np.ndarray):
        self.tfidf_feature = tfidf_feature


class PlagiarismMatcher(object):
    """
    记录每一组匹配结果的类
    """

    def __init__(self, src_sentence_idxs: List[int], sus_sentence_idxs: List[int], src_start: int, src_end: int,
                 sus_start: int, sus_end: int, src_tfidf_feature: np.ndarray, sus_tfidf_feature: np.ndarray):
        """
        :param src_sentence_idxs: source document 中 match 的 span 中所有句子的索引集
        :param sus_sentence_idxs: suspicious document 中 match 的 span 中所有句子的索引集
        :param src_start: source document 中 match 的 span 在文档中的开始位置
        :param src_end: source document 中 match 的 span 在文档中的结束位置,左闭右开
        :param sus_start: suspicious document 中 match 的 span 在文档中的开始位置
        :param sus_end: suspicious document 中 match 的 span 在文档中的结束位置,左闭右开
        :param src_tfidf_feature: source span 的 tfidf feature
        :param sus_tfidf_feature: suspicious span 的 tfidf feature
        """
        self.src_sentence_idxs = src_sentence_idxs
        self.sus_sentence_idxs = sus_sentence_idxs
        self.src_start = src_start
        self.src_end = src_end
        self.sus_start = sus_start
        self.sus_end = sus_end
        self.src_tfidf_feature = src_tfidf_feature
        self.sus_tfidf_feature = sus_tfidf_feature
        self.aligns = None  # 埋了个坑，在lcs_filter函数调用时被赋值 (src:List[int], sus:List[int])

    def __eq__(self, other):
        if type(other) != PlagiarismMatcher:
            return False
        return (self.src_start, self.src_end, self.sus_start, self.sus_end) == (other.src_start, other.src_end, other.sus_start, other.sus_end)

    def __hash__(self):
        return hash((self.src_start, self.src_end, self.sus_start, self.sus_end))

    def __repr__(self):
        return (f"src_sentence_idxs: {self.src_sentence_idxs}, sus_sentence_idxs: {self.sus_sentence_idxs}, "
                f"src_start: {self.src_start}, src_end: {self.src_end}, "
                f"sus_start: {self.sus_start}, sus_end: {self.sus_end}")
