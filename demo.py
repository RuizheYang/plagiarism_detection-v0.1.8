# encoding: utf-8
"""
@author: Shen Xinyi
@contact: xinyi_shen@shannonai.com
@file: testanchor.py
@time: 12/25/19 5:46 PM
@desc:
"""

from plagiarism_detection.plagiarism_detection import PlagiarismDetector

if __name__ == "__main__":
    src_file_path = "data/source.txt"
    sus_file_path = "data/suspicious.txt"

    with open(src_file_path) as fin:
        source_document = fin.read()
    with open(sus_file_path) as fin:
        suspicious_document = fin.read()

    plagiarism_detector = PlagiarismDetector()
    filtered_spans = plagiarism_detector.detect_plagiarism(source_document=source_document,
                                                           suspicious_document=suspicious_document)

    for idx, match_span in enumerate(filtered_spans):
        source_span = source_document[match_span.src_start: match_span.src_end]
        suspicious_span = suspicious_document[match_span.sus_start: match_span.sus_end]
        print(f"match span {idx}: sentence idxs: source {match_span.src_sentence_idxs}, "
              f"suspicious {match_span.sus_sentence_idxs}")
        print(f"src: {source_span}")
        print(f"sus: {suspicious_span}")

    print(f"total match spans: {len(filtered_spans)}")
