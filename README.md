# plagiarism_detection

sub-sentence-level plagiarism detection

## 说明
给定一篇语料库中的文档 `data/source.txt` 和一篇用户上传的文档 `data/suspicious.txt` ,检测 `data/suspicious.txt` 中所有疑似抄袭 `data/source.txt` 的片段,返回匹配片段在 `data/source.txt` 和 `data/suspicious.txt` 中的位置.

主要是复现了
http://ceur-ws.org/Vol-1180/CLEF2014wn-Pan-SanchezPerezEt2014.pdf
的做法,并进行了一些修改.

## Install
python==3.6.8

`pip install --no-cache-dir -i https://pypi.shannonai.com/root/stable/+simple/ --trusted-host pypi.shannonai.com shannon-preprocessor nlpc service-streamer`

`pip install -r requirements.txt`


## Usage
参考 `demo.py` 

## 算法流程
对原始文件中的每一个段落，将其与所有援引文件进行查重。
对每一个段落具体查重步骤为
1. 定位 source document 和 suspicious document 中匹配的 seeds（即根据相似度高的句子，相似度为句子tfidf feature的cosine similarity）
2. 扩展匹配的 seeds ,得到匹配的候选 spans
3. 过滤第 2 步中得到的候选 spans
4. 根据 lcs 字符串匹配的结果,过滤并定位最终的匹配 spans


