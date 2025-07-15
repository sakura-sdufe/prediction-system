# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/29 15:21
# @Author   : 张浩
# @FileName : cprint.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

def cprint(text="", text_color="红色", background_color="黑色", style="常规", sep='\n', end='\n\n', **kwargs):
    """
    将字符串按照指定颜色输出到控制台，也可以按照指定颜色输出评估结果到控制台。
    :param text: 需要输出的字符串
    :param text_color: 输出的颜色，使用 ANSI 转义序列颜色。支持：黑色、红色、绿色、黄色、蓝色、紫色、青色、白色。
    :param background_color: 输出的背景颜色，支持：黑色、红色、绿色、黄色、蓝色、紫色、青色、白色。
    :param style: 输出的样式，支持：常规、加粗、下划线。
    :param sep: 分隔符，默认为换行符。用于分隔 text 和 kwargs、kwargs 之间的内容。
    :param end: 结束符，默认为两个换行符。每次调用 cprint 函数之后的结束符。
    :param kwargs: 如果是评估结果，可以传入评估结果的参数和值
    :return: None
    """
    text_color_dict = {
        "黑色": "30", "红色": "31", "绿色": "32", "黄色": "33", "蓝色": "34", "紫色": "35", "青色": "36",
        "灰色":"90", "浅红色": "91", "浅绿色": "92", "浅黄色": "93", "浅蓝色": "94", "浅紫色": "95", "浅青色": "96",
        "白色": "97"
    }
    background_color_dict = {
        "黑色": "40", "红色": "41", "绿色": "42", "黄色": "43", "蓝色": "44", "紫色": "45", "青色": "46",
        "灰色": "100", "浅红色": "101", "浅绿色": "102", "浅黄色": "103", "浅蓝色": "104", "浅紫色": "105", "浅青色": "106",
        "白色": "107"
    }
    style_dict = {
        "常规": "0", "加粗": "1", "下划线": "4"
    }

    if kwargs:
        metric_text = sep.join([f"{k}: {v}" for k, v in kwargs.items()])
        text = text + sep + metric_text
        print(f"\033[{style_dict[style]};{text_color_dict[text_color]};{background_color_dict[background_color]}m{text}\033[0m",
              end=end)
    else:
        text = text
        print(f"\033[{style_dict[style]};{text_color_dict[text_color]};{background_color_dict[background_color]}m{text}\033[0m",
              end=end)
