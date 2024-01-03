# encoding: utf-8
"""
@author: Shen Xinyi
@contact: xinyi_shen@shannonai.com
@file: plagiarism_detection.py
@time: 12/28/19 11:07 AM
@desc:
"""

from typing import List, Optional, Dict, Any, Tuple, Set
from collections import defaultdict

import jieba
# from nlpc.edit_locator import locate_list_lcs_aligns

from plagiarism_detection.seeding import SeedLocator
from plagiarism_detection.extension import SeedExpander
from plagiarism_detection.filter import FragmentFilter
from plagiarism_detection.match_plagiarism import SentenceInfo, PlagiarismMatcher
# from plagiarism_detection.utils.debug_util import pretty_print_span, visualize_lcs, to_write_test
from plagiarism_detection.utils.text_utils import is_chalnum_char, count_chalnum_num, locate_list_lcs_aligns


class PlagiarismDetector(object):
    """
    查重 pipeline
    1. 定位 source document 和 suspicious document 中匹配的 seeds
    2. 扩展匹配的 seeds ,得到匹配的候选 spans
    3. 过滤第 2 步中得到的候选 spans
    4. 根据 lcs 字符串匹配的结果,过滤并定位最终的匹配 spans
    复现了这篇文章的做法,加入了一些修改:
    http://ceur-ws.org/Vol-1180/CLEF2014wn-Pan-SanchezPerezEt2014.pdf
    """

    def __init__(self, cos_threshold=0.45, dice_threshold=0.33, max_gap=2, expand_threshold=0.45, min_size=1,
                 min_plag_length=25, lcs_threshold=0.8):
        self.seed_locator = SeedLocator(
            cos_threshold=cos_threshold, dice_threshold=dice_threshold)
        self.seed_expander = SeedExpander(
            max_gap=max_gap, expand_threshold=expand_threshold, min_size=min_size)
        self.fragment_filter = FragmentFilter(min_plag_length=min_plag_length)
        self.lcs_threshold = lcs_threshold
        self.min_plag_length = min_plag_length

    def tokenize(self, doc_span: str) -> (List[str], List[int]):
        """
        返回一个 span 中每个非空白符组成的 list，以及每个非空白符在 span 中的位置
        """
        if doc_span == "":
            return [], []
        tokens = []
        positions = []
        for idx, token in enumerate(doc_span):
            # if token not in [" ", "\n", "\t"]:
            if is_chalnum_char(token):
                tokens.append(token)
                positions.append(idx)
        return tokens, positions

    def lcs_filter(self, spans: List[PlagiarismMatcher], source_document: str, suspicious_document: str) -> List[PlagiarismMatcher]:
        """
        根据 lcs 计算 overlap,过滤掉 overlap 比例小于给定阈值的 case
        (可选)根据 lcs 匹配的结果修改 span 的开头结尾, lcs 匹配的第一个 token 作为开头,最后一个 token 作为结尾
        :param spans:
        :param source_document:
        :param suspicious_document:
        :return:
        """
        lcs_filtered_match_spans = []
        for idx, span in enumerate(spans):
            source_span = source_document[span.src_start: span.src_end]
            suspicious_span = suspicious_document[span.sus_start: span.sus_end]
            source_tokens, source_positions = self.tokenize(
                doc_span=source_span)
            suspicious_tokens, suspicious_positions = self.tokenize(
                doc_span=suspicious_span)
            # lcs 字符串匹配
            aligns = locate_list_lcs_aligns(source_tokens, suspicious_tokens)
            # modify aligns 移除句首句末的孤岛 TODO
            assert len(aligns) > 0
            align_src_start = source_positions[aligns[0][0]]
            align_sus_start = suspicious_positions[aligns[0][1]]
            align_src_end = source_positions[aligns[-1][0]] + 1
            align_sus_end = suspicious_positions[aligns[-1][1]] + 1

            # 过滤source太稀疏的候选
            if len(aligns)/len(source_tokens) < 0.2 and "。" not in suspicious_span[align_sus_start:align_sus_end] and "。" in source_span[align_src_start:align_src_end]:
                continue

            # print(f"source: {source_span[: align_src_start]}【{source_span[align_src_start: align_src_end]}】"
            #       f"{source_span[align_src_end: ]}")
            # print(f"suspicious: {suspicious_span[: align_sus_start]}【{suspicious_span[align_sus_start: align_sus_end]}"
            #       f"】{suspicious_span[align_sus_end: ]}")
            # print(f"overlap: {len(aligns) / len(suspicious_span)}")
            # print(f"cut overlap: {len(aligns) / (align_sus_end - align_sus_start)}")
            # visualize_lcs(suspicious_tokens, source_tokens, aligns)
            if aligns[-1][1]-aligns[0][1] > self.min_plag_length:
                marks = [False] * (aligns[-1][1]-self.min_plag_length+1)
                pool = set([m[1] for m in aligns])
                cur_same = -1
                for i in range(aligns[0][1], aligns[-1][1]-self.min_plag_length+1):
                    if cur_same == -1:
                        j = 0
                        cur_same = 0
                        while j < len(aligns):
                            if i <= aligns[j][1] < i+25:
                                cur_same += 1
                            if aligns[j][1] >= i+25:
                                break
                            j += 1
                    else:
                        if i+24 in pool:
                            cur_same += 1
                        if i-1 in pool:
                            cur_same -= 1
                    if cur_same/self.min_plag_length >= self.lcs_threshold:
                        marks[i] = True
                # texts = []
                # for i in range(len(marks)):
                #     char = suspicious_tokens[i+aligns[0][1]]#+str(i)
                #     if i+aligns[0][1] in pool:
                #         if marks[i]:
                #             texts.append(f'\033[0;35;43m{char}\033[0m')
                #         else:
                #             texts.append(f'\033[0;35m{char}\033[0m')
                #     else:
                #         if marks[i]:
                #             texts.append(f'\033[0;42;43m{char}\033[0m')
                #         else:
                #             texts.append(char)
                # print("".join(texts))
                i = 0
                while i < len(marks):
                    j = i+1
                    if marks[i]:
                        while j < len(marks) and marks[j]:
                            j += 1
                        # 默认modify interval
                        j = j+24
                        src_aligns = [span.src_start+source_positions[ali[0]]
                                      for ali in aligns if i <= ali[1] <= j]
                        sus_aligns = [span.sus_start+suspicious_positions[ali[1]]
                                      for ali in aligns if i <= ali[1] <= j]
                        lcs_filtered_match_spans.append(
                            PlagiarismMatcher(src_sentence_idxs=[],  # todo
                                              sus_sentence_idxs=[],
                                              src_start=src_aligns[0],
                                              src_end=src_aligns[-1]+1,
                                              sus_start=sus_aligns[0],
                                              sus_end=sus_aligns[-1]+1,
                                              src_tfidf_feature=span.src_tfidf_feature,
                                              sus_tfidf_feature=span.sus_tfidf_feature))

                        lcs_filtered_match_spans[-1].aligns = (
                            src_aligns, sus_aligns)
                    i = j

            # src_aligns = [span.src_start+source_positions[ali[0]]
            #               for ali in aligns]
            # sus_aligns = [span.sus_start+suspicious_positions[ali[1]]
            #               for ali in aligns]

            # if modify_interval:
            #     # 根据 lcs 匹配的结果修改 span 起始位置
            #     lcs_filtered_match_spans.append(PlagiarismMatcher(src_sentence_idxs=span.src_sentence_idxs,
            #                                                       sus_sentence_idxs=span.sus_sentence_idxs,
            #                                                       src_start=span.src_start + align_src_start,
            #                                                       src_end=span.src_start + align_src_end,
            #                                                       sus_start=span.sus_start + align_sus_start,
            #                                                       sus_end=span.sus_start + align_sus_end,
            #                                                       src_tfidf_feature=span.src_tfidf_feature,
            #                                                       sus_tfidf_feature=span.sus_tfidf_feature))
            # else:
            #     lcs_filtered_match_spans.append(span)
            # lcs_filtered_match_spans[-1].aligns = (src_aligns, sus_aligns)
        return lcs_filtered_match_spans

    @staticmethod
    def ignore_filter(spans: List[PlagiarismMatcher],
                      suspicious_document: str,
                      ignore: Optional[List[str]]):
        """针对用户设置的ignore，不进行查重
        """
        ignore_filtered_spans = []
        if ignore:
            for span in spans:
                match = False
                for ignore_str in ignore:
                    if suspicious_document[span.sus_start: span.sus_end] == ignore_str:
                        match = True
                        break
                if not match:
                    ignore_filtered_spans.append(span)
        else:
            ignore_filtered_spans = spans

        return ignore_filtered_spans

    def detect_plagiarism(self,
                          source_document: str,
                          suspicious_document: str,
                          ignore: Optional[List[str]] = None):
        """
        core function
        输入两个文档(source document and suspicious document),检测疑似抄袭的所有片段
        """
        if not source_document or not suspicious_document:
            return []

        if source_document.strip() == suspicious_document.strip() and len(suspicious_document) > self.min_plag_length:
            merged_spans = [PlagiarismMatcher(src_sentence_idxs=[], sus_sentence_idxs=[],
                                              src_start=0, src_end=len(source_document),
                                              sus_start=0, sus_end=len(suspicious_document),
                                              src_tfidf_feature=None,
                                              sus_tfidf_feature=None)]
        else:
            # seeding
            src_sentence_infos, sus_sentence_infos, match_seeds = self.seed_locator.find_all_seeds(
                source_document=source_document,
                suspicious_document=suspicious_document)
            # print("----src----")
            # for idx, sent_info in enumerate(src_sentence_infos):
            #     print(idx, sent_info.raw_text)
            # print("----sus----")
            # for idx, sent_info in enumerate(sus_sentence_infos):
            #     print(idx, sent_info.raw_text)
            # match_seeds.sort(key=lambda x: x.sus_sentence_idxs[0])
            # pretty_print_span(match_seeds, suspicious_document, source_document)
            # print("==================================")
            # sus_sents, src_sents = {}, {}
            # for item in sus_sentence_infos:
            #     sus_sents[item.sentence_idx] = suspicious_document[item.start: item.end]
            # for item in src_sentence_infos:
            #     src_sents[item.sentence_idx] = source_document[item.start: item.end]

            # extension
            match_spans = self.seed_expander.expand_seed(src_sentence_infos=src_sentence_infos,
                                                         sus_sentence_infos=sus_sentence_infos,
                                                         match_seeds=match_seeds)
            # print(len(match_spans))

            # filter
            match_spans = self.fragment_filter.sep_filter(
                match_spans, source_document)
            filtered_spans = self.fragment_filter.overlap_filter(src_sentence_infos=src_sentence_infos,
                                                                 sus_sentence_infos=sus_sentence_infos,
                                                                 match_spans=match_spans)
            merged_spans = self.merge_span(filtered_spans)
        # lcs filter
        # 发现得到的结果中,有词汇比较接近,但意思完全不匹配的 span,所以添加了 lcs 的过滤,用字符串匹配的方法筛掉这一部分 bad case, 并根据
        # lcs 的匹配来修改匹配 span 的起始位置
        lcs_filtered_spans = self.lcs_filter(spans=merged_spans, source_document=source_document,
                                             suspicious_document=suspicious_document)

        length_filtered_spans = self.fragment_filter.length_filter(
            match_spans=lcs_filtered_spans)

        ignore_filtered_spans = PlagiarismDetector.ignore_filter(spans=length_filtered_spans,
                                                                 suspicious_document=suspicious_document,
                                                                 ignore=ignore)

        return ignore_filtered_spans

    @staticmethod
    def merge_span(filtered_spans: List):
        """
        将([0],[0])([1],[1])合成([0,1],[0,1])
        """
        filtered_spans.sort(key=lambda x: x.sus_sentence_idxs[0])
        merged_spans = []
        i = 0
        while i < len(filtered_spans):
            if i+1 < len(filtered_spans) and filtered_spans[i].sus_sentence_idxs[-1]+1 == filtered_spans[i+1].sus_sentence_idxs[0] and \
                    filtered_spans[i].src_sentence_idxs[-1]+1 >= filtered_spans[i+1].src_sentence_idxs[0] and \
                    filtered_spans[i].src_sentence_idxs[0] <= filtered_spans[i+1].src_sentence_idxs[-1]:
                src_idxs = list(set(
                    filtered_spans[i].src_sentence_idxs + filtered_spans[i + 1].src_sentence_idxs))
                src_idxs.sort()
                merged_spans.append(
                    PlagiarismMatcher(src_sentence_idxs=src_idxs,
                                      sus_sentence_idxs=filtered_spans[i].sus_sentence_idxs +
                                      filtered_spans[i +
                                                     1].sus_sentence_idxs,
                                      src_start=min(filtered_spans[i].src_start, filtered_spans[i+1].src_start),
                                      src_end=max(
                                          filtered_spans[i + 1].src_end, filtered_spans[i].src_end),
                                      sus_start=filtered_spans[i].sus_start,
                                      sus_end=filtered_spans[i +
                                                             1].sus_end,
                                      src_tfidf_feature=None,  # TODO
                                      sus_tfidf_feature=None))
                i += 1
            else:
                merged_spans.append(filtered_spans[i])
            i += 1
        return merged_spans

    def detect_plagiarism_api(self, paragraphs: List[str],
                              files: List[Dict], ignore: List[str], title: str):
        """wrap for api
        """
        def _format_precent(num: float) -> str:
            if num == 0:
                return "0"
            return format(num*100, '.2f')

        def _highlight_same(text: str, points: List[int],
                            sus_points: List[int], offset: int) -> str:
            left_sign, right_sign = "<span class=\"hc-rpt-hlt\">", "</span>"
            highlighted = []
            t_idx, p_idx = 0, 0
            while t_idx < len(text):
                if p_idx >= len(points):
                    highlighted.append(text[t_idx:])
                    t_idx = len(text)
                    continue
                start = t_idx
                while t_idx < points[p_idx]-offset:
                    t_idx += 1
                if t_idx > start:
                    highlighted.append(text[start:t_idx])
                start = t_idx
                if t_idx == points[p_idx]-offset:
                    highlighted.append(left_sign)
                    while p_idx+1 < len(points) and points[p_idx]+1 == points[p_idx+1] \
                            and sus_points[p_idx]+1 == sus_points[p_idx+1]:
                        p_idx += 1
                        t_idx += 1
                    highlighted.append(text[start:t_idx+1])
                    highlighted.append(right_sign)
                    p_idx += 1
                    t_idx += 1
            return "".join(highlighted)

        def _expand(preview: str, source_doc: str, src_start: int, src_end: int) -> str:
            # 往前多加一个词
            if src_start > 0:
                add_start = list(
                    jieba.cut(source_doc[max(src_start-10, 0):src_start]))[-1]
                preview = add_start + preview
            # 如果preview不到三行（字符数小于50）重复的end延长到碰到的第一个标点符号
            if src_end - src_start < 50 and src_end < len(source_doc):
                new_end = src_end
                while new_end < len(source_doc) and new_end < src_start+50:
                    if source_doc[new_end] in ",.?!;，。？！；":
                        preview = preview + source_doc[src_end:new_end+1]
                        return preview
                    new_end += 1
                preview = preview + source_doc[src_end:new_end]
            return preview

        result: Dict[str, Any] = {"title": {"text": title, "result": []},
                                  "article": [],
                                  "fileList": []}

        source_nums: Dict[str, int] = defaultdict(int)  # sus中被查重的字数，key为source
        suspicious_nums: List[int] = []  # 第i段中被查重出的字数（不包括标点符号）
        article_length = sum([count_chalnum_num(para) for para in paragraphs])

        # test_result = []
        for para in paragraphs:
            para_result: Dict[str, Any] = {"text": para, "result": []}
            cur_result: Dict[Tuple, List] = defaultdict(list)
            sus_aligns_all: Set[int] = set()
            for other_file in files:
                spans = self.detect_plagiarism(source_document=other_file["content"],
                                               suspicious_document=para,
                                               ignore=ignore)
                # if spans:
                #     test_result.append(to_write_test(para, other_file["content"], spans))
                for span in spans:
                    # TODO 前端暂不支持重复，保留第一个碰到的
                    overlap = False
                    for key in cur_result:
                        if not (span.sus_end <= key[0] or span.sus_start >= key[1]+key[0]):
                            overlap = True
                            break
                    if overlap:
                        continue

                    sus_aligns_all.update(span.aligns[1])
                    preview = _highlight_same(other_file["content"][span.src_start:span.src_end],
                                              span.aligns[0], span.aligns[1], span.src_start)
                    preview = _expand(
                        preview, other_file["content"], span.src_start, span.src_end)

                    source_nums[other_file["id"]] += len(span.aligns[1])

                    sus_length = span.sus_end-span.sus_start
                    item = {"name": other_file["name"],
                            "percent": _format_precent(len(span.aligns[1])/article_length),
                            "preview": preview}
                    cur_result[(span.sus_start, sus_length)].append(item)
            for key in cur_result:
                para_result["result"].append({"pos": key[0],
                                              "len": key[1],
                                              "files": cur_result[key]})
            result["article"].append(para_result)

            suspicious_nums.append(len(sus_aligns_all))

        if article_length:
            sus_rate = sum(suspicious_nums)/article_length
        else:
            sus_rate = 0
        result["percent"] = _format_precent(sus_rate)

        for other_file in files:
            src_rate = source_nums[other_file["id"]]/article_length
            item = {
                "id": other_file["id"],
                "name": other_file["name"],
                "percent": _format_precent(src_rate),
                "isUserFile": other_file["isUserFile"],
                "url": other_file["url"],
            }
            result["fileList"].append(item)

        return result
