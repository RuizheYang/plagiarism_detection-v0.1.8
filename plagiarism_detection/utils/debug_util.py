'''
@Author: Chen Wenjing
@Date: 2020-01-09
@LastEditor: Chen Wenjing
@LastEditTime: 2020-02-13 11:16:57
@Description: TODO
'''

import json
import os
import uuid
from time import strftime, localtime


def save_file(save_folder: str, content, max_file_num: int = 30):
    """将content写入文件（名字由当前时间和随机数组成），会自动删除掉老文件，维持save_folder下的文件数不超过max_file_num
    """
    files = os.listdir(save_folder)
    to_delete = len(files) - max_file_num
    files.sort()
    while to_delete >= 0:
        os.remove(save_folder+files[0])
        files = files[1:]
        to_delete -= 1
    file_name = strftime("%Y%m%d_%H_%M_%S", localtime()) + \
        str(uuid.uuid1()) + ".json"
    with open(save_folder + file_name, 'w', encoding="utf8") as f_w:
        json.dump(content, f_w, ensure_ascii=False)


def pretty_print_span(spans, suspicious_document: str, source_document: str):
    """pretty_print_span"""
    for idx, match_span in enumerate(spans):
        source_span = source_document[match_span.src_start: match_span.src_end]
        suspicious_span = suspicious_document[match_span.sus_start: match_span.sus_end]
        print(f"match span {idx}: sentence idxs: suspicious {match_span.sus_sentence_idxs} "
              f"source {match_span.src_sentence_idxs} ")
        print(f"sus: {suspicious_span}")
        print(f"src: {source_span}")


def visualize_lcs(suspicious_tokens, source_tokens, aligns):
    src_text = []
    sus_text = []
    src_pool = set(a[0] for a in aligns)
    sus_pool = set(a[1] for a in aligns)
    for i, char in enumerate(suspicious_tokens):
        if i in sus_pool:
            sus_text.append(f'\033[0;32m{char}\033[0m')
        else:
            sus_text.append(char)
    print("\n")
    for i, char in enumerate(source_tokens):
        if i in src_pool:
            src_text.append(f'\033[1;35m{char}\033[0m')
        else:
            src_text.append(char)
    print("".join(sus_text))
    print("".join(src_text))
    print("\n")


def to_write_test(sus_text: str, src_text: str, spans):
    """将结果保存"""
    data = {}
    data["sus_text"] = sus_text
    data["src_text"] = src_text
    data["spans"] = []
    for span in spans:
        item = {}
        item["sus_pos"] = (span.sus_start, span.sus_end)
        item["src_pos"] = (span.src_start, span.src_end)
        item["aligns"] = span.aligns
        data["spans"].append(item)
    return data
