# encoding: utf-8
"""
@author: Shen Xinyi
@contact: xinyi_shen@shannonai.com
@file: seeding.py
@time: 12/25/19 5:40 PM
@desc:
"""

import re
import numpy as np
from typing import List, Tuple
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from plagiarism_detection.match_plagiarism import SentenceInfo, PlagiarismMatcher


class SeedLocator(object):
    """
    查重 pipeline 第一步,定位 source document 和 suspicious document 中相似的句子,作为 seeds
    """

    def __init__(self, cos_threshold=0.45, dice_threshold=0.33):
        # TODO tfidf_model 中的默认 tokenizer 替换成自己写的 tokenizer,这样就不用传入处理过的句子了
        self.tfidf_model = CountVectorizer(token_pattern=r"\b\w+\b")
        self.cos_threshold = cos_threshold
        self.dice_threshold = dice_threshold

    def split_sentence_with_offset_and_para(self, document: str) -> Tuple[List[SentenceInfo], List[str]]:
        """
        分句,同时记录句子在文档中的位置,以及在文本中的第几个段落
        认为段落直接以 \n 隔开, "a\nb"被认为是两个段落, "a\n\nb被认为是三个段落"
        """
        # 分段
        paras = []
        para_offsets = []  # 每个段落开始在文档中的位置
        start = 0
        doc_len = len(document)
        for match in re.finditer(re.compile("\n"), document):
            end = match.span()[0] + 1
            para = document[start: end]
            paras.append(para)
            para_offsets.append(start)
            start = end
        if start < doc_len:
            paras.append(document[start:])
            para_offsets.append(start)

        # 分句
        sentence_infos = []
        sentences = []
        sentence_idx = 0
        delimiter_pattern = re.compile('([。！\!？\?])')
        for para_idx, (para, para_offset) in enumerate(zip(paras, para_offsets)):
            # print('para:', para_idx + 1)
            start = 0
            para_len = len(para)
            for match in re.finditer(delimiter_pattern, para):
                end = match.span()[0] + 1
                sentence = para[start: end]
                if not sentence.strip():  # 移除只含whitespace characters的句子
                    continue
                sentences.append(sentence)
                # print(sentence_idx, sentence)
                sentence_infos.append(SentenceInfo(start=start + para_offset, end=end + para_offset,
                                                   raw_text=sentence,
                                                   sentence_idx=sentence_idx, para_idx=para_idx))
                start = end
                sentence_idx += 1
            if start < para_len:
                sentence = para[start:]
                if sentence.strip():
                    sentence_infos.append(SentenceInfo(start=start + para_offset, end=para_len + para_offset,
                                                       raw_text=sentence,
                                                       sentence_idx=sentence_idx, para_idx=para_idx))
                    sentences.append(para[start:])
                    sentence_idx += 1
        return sentence_infos, sentences

    def tokenize(self, sentences: List[str]) -> List[str]:
        # TODO shenxiyi 用词表过滤低频词和停用词
        # 字之间空格隔开
        tokenized_sentences = [" ".join(list("".join(sentence.split()))) for sentence in sentences]
        return tokenized_sentences

    def calculate_tf_idf(self, documents: List[str]):
        """计算 tf-idf 特征"""
        self.tfidf_model.fit(documents)
        # print(self.tfidf_model.vocabulary_)
        # print(f"vocabulary size: {len(self.tfidf_model.vocabulary_)}")
        documents_tfidf = self.tfidf_model.transform(documents)
        # print(documents_tfidf.todense())
        return documents_tfidf.todense()

    def similarity_filter(self, src_tfidf, sus_tfidf) -> List[List[int]]:
        """
        计算 src 文档 和 suspicious 文档句子之间的 cosine similarity 和 dice coefficient, 根据给定的阈值选出相似的句子对,返回的
        结果是 src sentence 和 suspicious sentence 在相应文档中的 sentence idx
        dice coefficient 的部分暂时未使用
        """
        # 计算 cosine similarity
        # cos(sus_i, src_j) = \frac{sus_i \cdot src_j}{|sus_i||src_j|}
        src_sent_num = src_tfidf.shape[0]
        sus_sent_num = sus_tfidf.shape[0]
        src_norm = np.linalg.norm(src_tfidf, axis=1, keepdims=True)
        sus_norm = np.linalg.norm(sus_tfidf, axis=1, keepdims=True)
        cos_similarity = np.dot(src_tfidf, sus_tfidf.T) / np.dot(src_norm, sus_norm.T)
        # cos_similarity = np.dot(src_tfidf, sus_tfidf.T)
        # # 计算 dice threshold
        # # Dice(sus_i, src_j) = \frac{2|\delta(sus_i)\delta(src_j)|}{|\delta(sus_i)|^2 + |\delta(src_j)|^2}
        # delta_src = src_tfidf > 0
        # delta_sus = sus_tfidf > 0
        # delta_src_norm_power = np.power(np.linalg.norm(delta_src, axis=1, keepdims=True), 2)
        # delta_sus_norm_power = np.power(np.linalg.norm(delta_sus, axis=1, keepdims=True), 2)
        # dice_coefficient = 2 * np.dot(delta_src, delta_sus.T) / (
        #             np.repeat(delta_src_norm_power, sus_sent_num, axis=1) + np.repeat(delta_sus_norm_power.T,
        #                                                                               src_sent_num, axis=0))
        # threshold_select = np.multiply((cos_similarity > self.cos_threshold),
        #                                (dice_coefficient > self.dice_threshold))
        threshold_select = cos_similarity > self.cos_threshold
        select_indices = np.argwhere(threshold_select).tolist()
        return select_indices

    def test_sentence_infos(self, document: str, sentences: List[str], sentence_infos: List[SentenceInfo]):
        """检查 sentence_info 储存的信息是否正确"""
        for sentence_idx, (sentence, sentence_info) in enumerate(zip(sentences, sentence_infos)):
            assert sentence == document[sentence_info.start: sentence_info.end]
            assert sentence_info.sentence_idx == sentence_idx

    def find_all_seeds(self, source_document: str, suspicious_document: str) -> Tuple[
            List[SentenceInfo], List[SentenceInfo], List[PlagiarismMatcher]]:
        """
        core function
        :param source_document: 语料库中的一篇文档
        :param suspicious_document: 用户上传的待查重文档
        :return:
        """
        # 分句
        # print('split source sentences:')
        src_sentence_infos, src_sentences = self.split_sentence_with_offset_and_para(document=source_document)
        # print('split suspicious sentences:')
        sus_sentence_infos, sus_sentences = self.split_sentence_with_offset_and_para(document=suspicious_document)
        # 检查 sentence_info 的内容是否正确
        # self.test_sentence_infos(document=source_document, sentences=src_sentences, sentence_infos=src_sentence_infos)
        # self.test_sentence_infos(document=suspicious_document, sentences=sus_sentences, sentence_infos=sus_sentence_infos)
        # 分词,目前采用的分词方法是以字为单位
        src_tokenized_sentences = self.tokenize(sentences=src_sentences)
        sus_tokenized_sentences = self.tokenize(sentences=sus_sentences)
        src_sent_num = len(src_tokenized_sentences)
        sus_sent_num = len(sus_tokenized_sentences)

        # 计算 tf-idf
        documents_tfidf = self.calculate_tf_idf(documents=src_tokenized_sentences + sus_tokenized_sentences)

        src_tfidf = documents_tfidf[: src_sent_num]  # np.array, shape: [src_sent_num, feature_dim]
        sus_tfidf = documents_tfidf[src_sent_num:]  # np.array, shape: [sus_sent_num, feature_dim]

        # 将 tfidf feature 加入 sentence_info
        for idx, sentence_info in enumerate(src_sentence_infos):
            sentence_info.set_tfidf_feature(src_tfidf[idx])
        for idx, sentence_info in enumerate(sus_sentence_infos):
            sentence_info.set_tfidf_feature(sus_tfidf[idx])

        select_indices = self.similarity_filter(src_tfidf=src_tfidf, sus_tfidf=sus_tfidf)

        # 过滤 seeds,并生成 PlagiarismMatcher 类型的匹配结果
        match_seeds = []
        for src_idx, sus_idx in select_indices:
            src_sent_info = src_sentence_infos[src_idx]
            sus_sent_info = sus_sentence_infos[sus_idx]
            src_sent_len = src_sent_info.end - src_sent_info.start
            sus_sent_len = sus_sent_info.end - sus_sent_info.start
            # 根据句子长度过滤 seed
            if src_sent_len < max(5, sus_sent_len / 3) or sus_sent_len < max(5, sus_sent_len / 3):
                continue
            match_seeds.append(PlagiarismMatcher(src_sentence_idxs=[src_idx], sus_sentence_idxs=[sus_idx],
                                                 src_start=src_sent_info.start, src_end=src_sent_info.end,
                                                 sus_start=sus_sent_info.start, sus_end=sus_sent_info.end,
                                                 src_tfidf_feature=src_sent_info.tfidf_feature,
                                                 sus_tfidf_feature=sus_sent_info.tfidf_feature))

        return src_sentence_infos, sus_sentence_infos, match_seeds
