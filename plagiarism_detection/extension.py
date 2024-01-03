# encoding: utf-8
"""
@author: Shen Xinyi
@contact: xinyi_shen@shannonai.com
@file: extension.py
@time: 12/28/19 11:03 AM
@desc:
"""

import numpy as np
from typing import List, Dict
from plagiarism_detection.match_plagiarism import SentenceInfo, PlagiarismMatcher


class SeedExpander(object):
    """
    查重 pipeline 第二步,将上一步得到的 seeds 进行合并组合,得到 source document 和 suspicious document 中相似的候选 spans
    """
    def __init__(self, max_gap=2, expand_threshold=0.45, min_size=1):
        self.max_gap = max_gap
        self.expand_threshold = expand_threshold
        self.min_size = min_size

    def check_gap(self, sentence_idxs: List[int]) -> bool:
        """
        检查候选的 seed 集是否满足 max gap 的限制,任意两个相邻 seed 之间的句子数不超过 max gap
        要求传入的 sentence_idxs 是从小到大排序的
        """
        for idx, sentence_idx in enumerate(sentence_idxs[: -1]):
            if sentence_idxs[idx + 1] - sentence_idx - 1 > self.max_gap:
                return False
        return True

    def check_similarity(self, src_span_feature: np.ndarray, sus_span_feature: np.ndarray) -> bool:
        """检查待匹配的 src_span 和 sus_span 是否满足相似度阈值"""
        src_span_norm = np.linalg.norm(src_span_feature, axis=1, keepdims=True) # 求范数
        sus_span_norm = np.linalg.norm(sus_span_feature, axis=1, keepdims=True)
        cos_similarity = np.dot(src_span_feature, sus_span_feature.T) / np.dot(src_span_norm, sus_span_norm.T)
        if cos_similarity > self.expand_threshold:
            return True
        else:
            return False

    def extract_match_seeds(self, seed_dict: Dict[int, List[int]], seed_list: List[int]) -> List[int]:
        """
        根据字典形式的 match seeds,提取出 src/sus 文档中的 seed list 对应到 sus/src 文档中的 seed list
        例如:
        seed_dict = {1: [1, 2], 3: [2, 3], 4: [7, 8]}
        seed_list = [1, 3]
        则返回; [1, 2, 3]
        """
        match_seed_list = []
        for seed in seed_list:
            match_seed_list.extend(seed_dict[seed])
        match_seed_list = list(set(match_seed_list))
        match_seed_list.sort()
        return match_seed_list

    def expand_seed(self, src_sentence_infos: List[SentenceInfo], sus_sentence_infos: List[SentenceInfo],
                    match_seeds: List[PlagiarismMatcher]) -> List[PlagiarismMatcher]:
        """
        core function
        扩展合并匹配的 seeds,得到匹配的 spans
        """
        match_spans = []
        # 提取出 match seeds 中每个 src_idx 对应的所有 sus_idx, 保存成字典
        # 提取出 match seeds 中每个 sus_idx 对应的所有 src_idx, 保存成字典
        # 例如, match_seeds 中得到的匹配结果为 :
        # (source_idx, suspicious_idx) = (1, 0), (1, 2), (1, 3), (2, 3),
        # 则 src_indices_dict = {1: [0, 2, 3], 2: [3]}
        #    sus_indices_dict = {0: [1], 2: [1], 3: [1, 2]}
        src_indices_dict: Dict[int, List[int]] = dict()
        sus_indices_dict: Dict[int, List[int]] = dict()
        for seed in match_seeds:
            src_idx = seed.src_sentence_idxs[0]
            sus_idx = seed.sus_sentence_idxs[0]
            src_indices_dict[src_idx] = src_indices_dict.get(src_idx, []) + [sus_idx]
            sus_indices_dict[sus_idx] = sus_indices_dict.get(sus_idx, []) + [src_idx]
        src_seed_idxs = list(src_indices_dict.keys())
        src_seed_idxs.sort()
        sus_seed_idxs = list(sus_indices_dict.keys())
        sus_seed_idxs.sort()

        # 这一部分是在实现论文中的伪码部分
        # 遍历 suspicious document 中被 match seeds cover 的 fragment F
        for sus_start_idx_tmp in range(len(sus_seed_idxs)):
            for sus_end_idx_tmp in range(sus_start_idx_tmp, len(sus_seed_idxs)):  # 左闭右闭区间
                fragment_F_seeds = sus_seed_idxs[sus_start_idx_tmp: sus_end_idx_tmp + 1]
                S1_src_seed_idxs = self.extract_match_seeds(seed_dict=sus_indices_dict, seed_list=fragment_F_seeds)
                if len(S1_src_seed_idxs) < self.min_size:
                    continue
                # 遍历 source document 中被 S1 cover 的 fragment F'
                for src_start_idx in range(len(S1_src_seed_idxs)):
                    for src_end_idx in range(src_start_idx, len(S1_src_seed_idxs)):
                        src_start_sent_idx = S1_src_seed_idxs[src_start_idx]
                        src_end_sent_idx = S1_src_seed_idxs[src_end_idx]
                        fragment_F1 = list(range(src_start_sent_idx, src_end_sent_idx + 1))
                        fragment_F1_seeds = S1_src_seed_idxs[src_start_idx: src_end_idx + 1]
                        # 判断是否满足 cover 要求的条件
                        if not self.check_gap(fragment_F1_seeds):
                            continue
                        S2_sus_seed_idxs = self.extract_match_seeds(seed_dict=src_indices_dict,
                                                                    seed_list=fragment_F1_seeds)
                        if len(S2_sus_seed_idxs) < self.min_size:
                            continue
                        # 遍历 suspicious document 中被 S2 cover 的 fragment F''
                        for sus_start_idx in range(len(S2_sus_seed_idxs)):
                            for sus_end_idx in range(sus_start_idx, len(S2_sus_seed_idxs)):
                                sus_start_sent_idx = S2_sus_seed_idxs[sus_start_idx]
                                sus_end_sent_idx = S2_sus_seed_idxs[sus_end_idx]
                                fragment_F2 = list(range(sus_start_sent_idx, sus_end_sent_idx + 1))
                                fragment_F2_seeds = S2_sus_seed_idxs[sus_start_idx: sus_end_idx + 1]
                                # 判断是否满足 cover 要求的条件
                                if not self.check_gap(fragment_F2_seeds):
                                    continue
                                # 计算 fragment 的 tfidf feature
                                src_fragment_feature = np.sum(
                                    [src_sentence_infos[idx].tfidf_feature for idx in fragment_F1], axis=0)
                                sus_fragment_feature = np.sum(
                                    [sus_sentence_infos[idx].tfidf_feature for idx in fragment_F2], axis=0)
                                # 比较 fragment_F1 和 fragment_F2 是否满足相似度阈值
                                if not self.check_similarity(src_span_feature=src_fragment_feature,
                                                             sus_span_feature=sus_fragment_feature):
                                    continue
                                match_spans.append(PlagiarismMatcher(src_sentence_idxs=fragment_F1,
                                                                     sus_sentence_idxs=fragment_F2,
                                                                     src_start=src_sentence_infos[
                                                                         fragment_F1[0]].start,
                                                                     src_end=src_sentence_infos[fragment_F1[-1]].end,
                                                                     sus_start=sus_sentence_infos[
                                                                         fragment_F2[0]].start,
                                                                     sus_end=sus_sentence_infos[fragment_F2[-1]].end,
                                                                     src_tfidf_feature=src_fragment_feature,
                                                                     sus_tfidf_feature=sus_fragment_feature
                                                                     ))
        # match_spans 中有重复,去掉
        match_spans = list(set(match_spans))
        return match_spans