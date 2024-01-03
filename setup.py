'''
@Author: Chen Wenjing
@Date: 2019-11-28
@LastEditor: Chen Wenjing
@LastEditTime: 2020-01-12 19:19:01
@Description: TODO
'''
import os
from setuptools import setup, find_packages

# 这一部分不能修改，否则无法通过相关检查
# NAME = os.environ["CI_PROJECT_NAME"]
# VERSION = os.environ["CI_COMMIT_TAG"]
# URL = os.environ["CI_PROJECT_URL"]
# AUTHOR = os.environ["GITLAB_USER_NAME"]
# AUTHOR_EMAIL = os.environ["GITLAB_USER_EMAIL"]
ZIP_SAFE = False

NAME = "plagiarism_detection"
VERSION = "0.1.9"
URL = ""
AUTHOR = ""
AUTHOR_EMAIL = ""

# 项目需要提供requirements.txt和README.md
with open('requirements.txt') as fp:
    REQUIREMENTS = fp.read().splitlines()

with open("README.md", "r",encoding='utf-8')as fp:
    LONG_DESCRIPTION = fp.read()

# 这些变量需要自行填写值
# string example: description="this is a python package"
DESCRIPTION = "plagiarism_detection"
# string of tuple example: keywords=("test","python_package")
KEYWORDS = ("plagiarism", "detection")
PLATFORMS = ["any"]  # list of string example: ["any"]

setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIREMENTS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    keywords=KEYWORDS,
    url=URL,
    # packages参数需要自行填写
    packages=find_packages(exclude=('tests', 'dataset')),
    platforms=PLATFORMS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    scripts=[],
    # entry_point参数根据需求自行填写
    # entry_points={
    #     # how to write script: https://bit.ly/30pwIyv
    #     'console_scripts': [
    #         'webreader=web_reader.__main__:main',
    #         'web-reader=web_reader.__main__:main',
    #         'web_reader=web_reader.__main__:main',
    #     ],
    # },
    zip_safe=ZIP_SAFE,
    # classifiers参数根据需求自行填写
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Private :: Do Not Upload",
    ),
)
