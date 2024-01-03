'''
@Author: Chen Wenjing
@Date: 2020-01-16
@LastEditor: Chen Wenjing
@LastEditTime: 2020-02-22 17:05:55
@Description: TODO
'''


def is_chinese_char(char: str) -> bool:
    """
    [ref](https://www.qqxiuzi.cn/zh/hanzi-unicode-bianma.php)
    """
    if u"\u4E00" <= char <= u"\u9FA5" or \
        u"\u9FA6" <= char <= u"\u9FEF" or \
            u"\u3400" <= char <= u"\u4DB5":
        return True
    return False


def is_chalnum_char(char: str) -> bool:
    """该字符是否是中文英文或者数字
    """
    if is_chinese_char(char) or char.isalnum():
        return True
    return False


def count_chalnum_num(text: str) -> int:
    """text中包含的中文英文及数字的个数"""
    num = 0
    for char in text:
        if is_chalnum_char(char):
            num += 1
    return num


def locate_list_lcs_aligns(s1, s2):
    """locate_list_lcs_aligns"""
    m = len(s1)
    n = len(s2)

    m += 1
    n += 1

    # // add 1 for empty wstring start

    dp = [[0]*n for _ in range(m)]
    points = [[(0, 0)]*n for _ in range(m)]

    for i in range(m):
        dp[i][0] = 0
        points[i][0] = (i - 1, 0)

    for j in range(n):
        dp[0][j] = 0
        points[0][j] = (0, j - 1)

    for i in range(m-1):
        for j in range(n-1):
            left_top = dp[i][j]
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = left_top+1  # 因为前面多append了一个空位
                prev = (i, j)  # 注意这里的i,j是string_idx + 1, 是对应的dp的idx
            else:
                left = dp[i+1][j]
                top = dp[i][j+1]
                if (top >= left):
                    dp[i+1][j+1] = top
                    prev = (i, j+1)
                else:
                    dp[i+1][j+1] = left
                    prev = (i+1, j)
            points[i+1][j+1] = prev

    cur_point = [m-1, n-1]
    while cur_point[0]-1 >= 0 and dp[cur_point[0]][cur_point[1]] == dp[cur_point[0]-1][cur_point[1]]:
        cur_point[0] -= 1
    while cur_point[1]-1 >= 0 and dp[cur_point[0]][cur_point[1]] == dp[cur_point[0]][cur_point[1]-1]:
        cur_point[1] -= 1

    path = [(m-1, n-1)]

    # for l in dp:
    #     print(l)

    while (cur_point[0] >= 0 and cur_point[1] >= 0):
        prev_point = points[cur_point[0]][cur_point[1]]
        path.append(prev_point)
        cur_point = prev_point

    prev_point = path[-1]
    aligns = []
    for i in range(len(path)-2, -1, -1):
        cur_point = path[i]
        if (cur_point[0] > prev_point[0] and cur_point[1] > prev_point[1]):
            aligns.append(prev_point)

        prev_point = cur_point
    return aligns
