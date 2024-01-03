# encoding: utf-8
"""
@author: Shen Xinyi
@contact: xinyi_shen@shannonai.com
@file: filter.py
@time: 12/30/19 3:11 PM
@desc:
"""

import numpy as np
from typing import List, Dict
from plagiarism_detection.match_plagiarism import SentenceInfo, PlagiarismMatcher


class FragmentFilter(object):
    """
    查重 pipeline 第三步,根据 overlap 和 span 长度来过滤候选 spans
    """

    def __init__(self, min_plag_length=20, cos_threshold=0.9):
        self.min_plag_length = min_plag_length
        self.cos_threshold = cos_threshold  # 对于overlap的span，若quality一致，对候选者设的

    def extract_overlapping_spans(self, pivot_idx: int, sorted_match_spans: List[PlagiarismMatcher],
                                  keep_tag: List[int]) -> List[int]:
        """找出跟 pivot span overlap 的所有 match span,返回这些 match span 在 sorted_match_spans 中的 idx"""
        overlap_list = []
        pivot_sus_start = sorted_match_spans[pivot_idx].sus_start
        pivot_sus_end = sorted_match_spans[pivot_idx].sus_end
        for candidate_idx in range(len(sorted_match_spans)):
            if candidate_idx == pivot_idx:
                continue
            if keep_tag[candidate_idx] == 0:
                continue
            candidate_sus_start = sorted_match_spans[candidate_idx].sus_start
            candidate_sus_end = sorted_match_spans[candidate_idx].sus_end
            if candidate_sus_start >= pivot_sus_end:
                break  # 因为 sorted_match_spans 是排过序的,后面不会再有满足的 span
            if pivot_sus_start >= candidate_sus_end:  # 经过上面的判断后,不满足 overlap 的唯一庆幸
                continue
            overlap_list.append(candidate_idx)
        return overlap_list

    def f_sim(self, reference_fragment: List[int], sus_part_fragment: List[int],
              src_sentence_infos: List[SentenceInfo], sus_sentence_infos: List[SentenceInfo]) -> np.ndarray:
        reference_features = np.concatenate([src_sentence_infos[idx].tfidf_feature for idx in reference_fragment],
                                            axis=0)
        sus_part_features = np.concatenate(
            [sus_sentence_infos[idx].tfidf_feature for idx in sus_part_fragment], axis=0)
        reference_norm = np.linalg.norm(
            reference_features, axis=1, keepdims=True)
        sus_part_norm = np.linalg.norm(
            sus_part_features, axis=1, keepdims=True)
        cos_similarity = np.dot(
            reference_features, sus_part_features.T) / np.dot(reference_norm, sus_part_norm.T)
        sim_value = np.mean(np.max(cos_similarity, axis=0))
        # 发现在计算的时候会出现一点精度上的误差,(src: [1,2], sus: [1]) 和 (src: [1], sus: [1]) (src_1 sus_1 对应)
        # 的计算结果应当相同但有误差,在后续判断等值的时候会有问题,所以做了近似
        return np.float16(sim_value)

    def compute_quailty(self, src_sentence_infos: List[SentenceInfo], sus_sentence_infos: List[SentenceInfo],
                        span_x: PlagiarismMatcher, span_y: PlagiarismMatcher) -> float():
        """计算 q_y(x)
        TODO
        这个部分的计算中存在一个问题:
        假设待匹配 span src:[1, 2, 3], sus: [1, 2, 3]
        src_1 和 sus_1 非常相似, src_2, sus_2 完全一样, src_3 和 sus_3 非常相似
        则最终选择的结果会是 ([1, 2], [1, 2]),而不是 ([1, 2, 3], [1, 2, 3])
        """
        x_src_sent_idxs = span_x.src_sentence_idxs
        x_sus_sent_idxs = span_x.sus_sentence_idxs
        y_src_sent_idxs = span_y.src_sentence_idxs
        y_sus_sent_idxs = span_y.sus_sentence_idxs
        overlap_idxs = list(set(x_sus_sent_idxs) & set(y_sus_sent_idxs))
        non_overlap_idx = list(set(x_sus_sent_idxs) - set(overlap_idxs))
        sim_XO = self.f_sim(reference_fragment=x_src_sent_idxs, sus_part_fragment=overlap_idxs,
                            src_sentence_infos=src_sentence_infos, sus_sentence_infos=sus_sentence_infos)
        if len(non_overlap_idx) > 0:
            sim_XN = self.f_sim(reference_fragment=x_src_sent_idxs, sus_part_fragment=non_overlap_idx,
                                src_sentence_infos=src_sentence_infos, sus_sentence_infos=sus_sentence_infos)
        else:
            sim_XN = 0.  # 0 or 1?
        quality = sim_XO + (1 - sim_XO) * sim_XN
        return quality

    def compute_cos_similarity(self, src_sentence_infos: List[SentenceInfo], sus_sentence_infos: List[SentenceInfo],
                               span: PlagiarismMatcher) -> float:
        src_feature = np.sum(
            [src_sentence_infos[idx].tfidf_feature for idx in span.src_sentence_idxs], axis=0)
        sus_feature = np.sum(
            [sus_sentence_infos[idx].tfidf_feature for idx in span.sus_sentence_idxs], axis=0)
        src_norm = np.linalg.norm(src_feature, axis=1, keepdims=True)
        sus_norm = np.linalg.norm(sus_feature, axis=1, keepdims=True)
        cos_similarity = np.dot(src_feature, sus_feature.T) / \
            np.dot(src_norm, sus_norm.T)
        return cos_similarity.item()

    def find_keep_idx(self, pivot_idx: int, overlap_list: List[int], sorted_match_spans: List[PlagiarismMatcher],
                      src_sentence_infos: List[SentenceInfo], sus_sentence_infos: List[SentenceInfo]) -> int:
        """根据相似性保留 pivot 和 overlap 的 spans 中的唯一一个"""
        pivot_span = sorted_match_spans[pivot_idx]
        ops = []
        pos = []
        for overlap_idx in overlap_list:
            overlap_span = sorted_match_spans[overlap_idx]
            quality_op = self.compute_quailty(src_sentence_infos=src_sentence_infos,
                                              sus_sentence_infos=sus_sentence_infos,
                                              span_x=pivot_span, span_y=overlap_span)
            ops.append(quality_op)

            quality_po = self.compute_quailty(src_sentence_infos=src_sentence_infos,
                                              sus_sentence_infos=sus_sentence_infos,
                                              span_x=overlap_span, span_y=pivot_span)
            pos.append(quality_po)

        candidates = FragmentFilter._max_quality([max(ops)] + pos)
        candi_index = [pivot_idx] + overlap_list
        if len(candidates) == 1:
            keep_idx = candi_index[candidates[0]]
        else:
            # 如果 candidate span 和 keep span 的 suspicious sentences 相同,但 src sentences 是真子集关系,则 quality 的计算
            # 结果会相等, src span 中可能会有无关的句子而难以判断,在这种情况下直接计算整个对应 span 的 cos similarity,取 similarity
            # 较大的那一对 span
            cos_similarity = []
            for idx in candidates:
                similarity = self.compute_cos_similarity(src_sentence_infos=src_sentence_infos,
                                                         sus_sentence_infos=sus_sentence_infos,
                                                         span=sorted_match_spans[candi_index[idx]])
                # src_1 和 sus_1 非常相似, src_2, sus_2 完全一样，仅算cos_sim ([2],[2])会优于([1,2],[1,2]) 故对cos_sim加一阈值，优先选最长的
                if similarity > self.cos_threshold:
                    cos_similarity.append(1.0)
                else:
                    cos_similarity.append(similarity)
            s_candidates = FragmentFilter._max_quality(cos_similarity)
            if len(s_candidates) == 1:
                keep_idx = candi_index[candidates[s_candidates[0]]]
            else:
                # quality和similarity均一致，选最长的吧？
                max_length = 0
                for i in s_candidates:
                    idx = candi_index[candidates[i]]
                    cur_length = sorted_match_spans[idx].sus_end - \
                        sorted_match_spans[idx].sus_start + sorted_match_spans[idx].src_end - sorted_match_spans[idx].src_start
                    if cur_length > max_length:
                        max_length = cur_length
                        keep_idx = idx

        return keep_idx

    @staticmethod
    def _max_quality(nums: List[float]) -> List[int]:
        """类似argmax，但会返回所有候选，最大值为1，由于精度问题大于一的都会被看做1"""
        max_index = []
        max_value = 0.0
        for i, value in enumerate(nums):
            if value >= 1 and max_value >= 1:
                max_index.append(i)
                continue
            if value > max_value:
                max_index = [i]
                max_value = value
            elif value == max_value:
                max_index.append(i)
        return max_index

    def overlap_filter(self, src_sentence_infos: List[SentenceInfo], sus_sentence_infos: List[SentenceInfo],
                       match_spans: List[PlagiarismMatcher]) -> List[PlagiarismMatcher]:
        match_spans_num = len(match_spans)
        keep_tag = [1] * match_spans_num  # 1 表示保留, 0 表示过滤
        sorted_match_spans = sorted(
            match_spans, key=lambda x: (x.sus_start, x.sus_end))
        pivot_idx = 0
        # for i, span in enumerate(sorted_match_spans):
        #     if span.src_sentence_idxs == [1, 2, 3] and span.sus_sentence_idxs == [2, 3, 4]:
        #         print(i)
        #     if span.src_sentence_idxs == [6, 7] and span.sus_sentence_idxs == [5, 6]:
        #         print(i)
        select_idxs = []
        while sum(keep_tag):
            # 找到第一个非0的作为idx
            pivot_idx = 0
            while not keep_tag[pivot_idx]:
                pivot_idx += 1

            # if keep_tag[pivot_idx] == 0:
            #     pivot_idx += 1
            #     continue
            overlap_list = self.extract_overlapping_spans(pivot_idx=pivot_idx, sorted_match_spans=sorted_match_spans,
                                                          keep_tag=keep_tag)
            # for i in [50, 143]:
            #     if i in overlap_list:
            #         print(i, "dd")
            # print("\n")

            if len(overlap_list) == 0:  # ? 不一定包含自己吗？
                select_idxs.append(pivot_idx)
                keep_tag[pivot_idx] = 0
                pivot_idx += 1
                continue

            keep_idx = self.find_keep_idx(pivot_idx=pivot_idx, overlap_list=overlap_list,
                                          sorted_match_spans=sorted_match_spans,
                                          src_sentence_infos=src_sentence_infos,
                                          sus_sentence_infos=sus_sentence_infos)

            overlap_with_keep = self.extract_overlapping_spans(pivot_idx=keep_idx,
                                                               sorted_match_spans=sorted_match_spans,
                                                               keep_tag=keep_tag)

            # for candidate_idx in overlap_list + [pivot_idx]:
            for candidate_idx in overlap_with_keep + [keep_idx]:
                keep_tag[candidate_idx] = 0
            select_idxs.append(keep_idx)
            # keep_tag[keep_idx] = 1
            # pivot_idx += 1
        # return [sorted_match_spans[idx] for idx in range(len(keep_tag)) if keep_tag[idx]]
            # print(sorted_match_spans[keep_idx])
        return [sorted_match_spans[idx] for idx in select_idxs]

    def length_filter(self, match_spans: List[PlagiarismMatcher]) -> List[PlagiarismMatcher]:
        """根据长度过滤,过滤掉 src span 或 sus span 长度小于 min_plag_length 的 case"""
        length_filtered_spans = []
        for match_span in match_spans:
            if match_span.sus_end - match_span.sus_start < self.min_plag_length or \
                    match_span.src_end - match_span.src_start < self.min_plag_length:
                continue
            length_filtered_spans.append(match_span)
        return length_filtered_spans

    def sep_filter(self, match_spans: List[PlagiarismMatcher], source_document: str) -> List[PlagiarismMatcher]:
        """"""
        filtered_spans = []
        for match_span in match_spans:
            if "\n" in source_document[match_span.src_start:match_span.src_end].strip():
                continue
            filtered_spans.append(match_span)
        return filtered_spans

    def filter_all(self, src_sentence_infos: List[SentenceInfo], sus_sentence_infos: List[SentenceInfo],
                   match_spans: List[PlagiarismMatcher]) -> List[PlagiarismMatcher]:
        """
        core function
        根据匹配 fragments 的长度和 fragments 之间的 overlap 过滤
        """
        overlap_filtered_match_spans = self.overlap_filter(src_sentence_infos=src_sentence_infos,
                                                           sus_sentence_infos=sus_sentence_infos,
                                                           match_spans=match_spans)
        length_filtered_spans = self.length_filter(
            match_spans=overlap_filtered_match_spans)
        return length_filtered_spans
