# encoding: utf-8

import time

from flask import Flask, request, jsonify

from plagiarism_detection.plagiarism_detection import PlagiarismDetector
from plagiarism_detection.utils.debug_util import save_file

app = Flask(__name__)

SAVE_FOLDER = ""  # debug时存下请求用


@app.route("/api/duplicate/check", methods=["POST"])
def detect():
    """查重api"""
    result = {"code": 0, "msg": "", "data": {}}

    if request.method == "POST":
        request_json = request.get_json()

        if SAVE_FOLDER:
            save_file(SAVE_FOLDER, request_json)

        paragraphs = request_json['article']
        files = request_json['files']
        ignore = request_json["ignore"]
        title = request_json["title"]
    else:
        result["code"] = 1
        result["msg"] = "wrong request method"
        return jsonify(result)

    st = time.time()
    detect_result = plagiarism_detector.detect_plagiarism_api(paragraphs, files, ignore, title)
    et = time.time()
    print("infer time:", et - st)
    result["data"] = detect_result
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """health探针 服务异常重启"""
    try:
        detect_result = plagiarism_detector.detect_plagiarism_api([], [], [], "")
        if "percent" in detect_result:
            return "ok"
        else:
            raise ValueError
    except Exception:
        raise RuntimeError


if __name__ == '__main__':

    plagiarism_detector = PlagiarismDetector()

    app.run(
        host="0.0.0.0",
        port=1212
    )
