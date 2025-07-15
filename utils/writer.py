# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/29 15:28
# @Author   : 张浩
# @FileName : writer.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import shutil
import traceback
import matplotlib
import numpy as np
import pandas as pd
from typing import Union
from copy import deepcopy
from datetime import datetime
from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline  # 设置图片以SVG格式显示

from data.read import write_result_identify, check_result_identify, write_pickle
from .cprint import cprint

matplotlib.use("TkAgg")
# 设置中文字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False   # 用来正常显示负号


class Writer:
    VERSION = 1.0
    def __init__(self, save_dir, is_delete=False, *, fmts=None, figsize=None, **kwargs):
        """
        写入指标、参数、数据、日志；保存模型
        :param save_dir: 保存的根目录，要求为绝对地址。
        :param is_delete: 是否删除原有的 save_dir 文件夹。
        :param fmts: 图像的格式，例如：['b-', 'm--', 'g-.', 'r:']。
        :param figsize: 图窗大小，例如：(7, 5)。
        :param kwargs: 其他图窗相关参数。
        """
        if not os.path.isabs(save_dir):
            raise ValueError(f"输入的根目录 {save_dir} 不是绝对路径！")

        # 设置保存目录
        self.save_dir = save_dir  # 保存的根目录
        self.is_delete = is_delete  # 是否删除原有的 save_result 文件夹

        # draw 绘图方法参数
        if fmts is None:
            fmts = ['C0-', 'C1--', 'C2-.', 'C3:', 'C4-', 'C5--', 'C6-.', 'C7:', 'C8-', 'C9--']
        if figsize is None:
            figsize = (7, 5)
        self.fmts, self.figsize = fmts, figsize
        self.init_axes_set = kwargs

        # 暂存的内容和文件个数
        self.num_df, self.num_param, self.num_text = 0, 0, 0
        self.storage_df = dict()
        self.storage_param = dict()
        self.storage_text = dict()

        # 删除根目录和创建根目录
        if os.path.exists(self.save_dir) and self.is_delete:
            self._delete_dir()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            write_result_identify(self.save_dir, self.VERSION)

    def _delete_dir(self):
        """以递归的形式删除 self.save_dir 文件夹"""
        raise_info = (f"您当前删除的目录并非来自于 Writer 类自动生成或您对目录内的识别文件进行了修改。已触发保护机制，终止删除操作！\n"
                      f"\t当前的根目录路径为：'{self.save_dir}'，请仔细检查！\n")
        # Step 1: 删除保护机制。如果没有找到识别文件或者识别文件版本与当前版本不一致，则抛出 OSError 错误。
        try:
            IDENTIFY_VERSION, _, _ = check_result_identify(self.save_dir)  # 读取识别文件，IDENTIFY_PATH 已经在函数内检查过了。
            assert IDENTIFY_VERSION == self.VERSION, "识别文件版本不一致，无法删除！"
        except AssertionError:  # 抛出原异常，并且添加 raise_info
            error_message = traceback.format_exc()
            full_message = f"{error_message}\n{raise_info}"
            raise OSError(full_message)

        # Step 2: 弹出资源管理器，并在输出中向用户确认是否删除。
        if os.listdir(self.save_dir):
            os.startfile(self.save_dir)
            verify = input("请在已打开的资源管理器中仔细核验需要删除的根目录路径，删除将无法恢复，是否继续删除（摁下除 'y' 以外所有键取消删除）？")
            if verify.lower() == 'y':
                shutil.rmtree(self.save_dir)
                print(f"根目录 '{self.save_dir}' 删除成功！")
            else:
                print(f"取消删除根目录 '{self.save_dir}'！")
        else:
            os.rmdir(self.save_dir)
            print(f"根目录 '{self.save_dir}' 删除成功！")


    def add_df(self, data_df, axis=0, filename=None, folder=None, suffix='csv', save_mode='none',
               reset_index=False, reset_drop=False, sort_list: Union[str, list]=None, sort_ascending: Union[bool, list]=True):
        """
        添加数据到 self.storage_df 中。
        :param data_df: pd.DataFrame 类型，需要添加的数据。
        :param axis: 指定拼接的方向，0 为纵向拼接，1 为横向拼接。
        :param filename: 保存文件的文件名，无需添加后缀名。如果为 None，则用 "DataFrame_i" 代替。
        :param folder: 保存文件所在的文件夹名称，如果为 None，则保存到根目录下。
        :param suffix: 保存文件的文件后缀名，默认为 'csv'。可接受的后缀名有：'csv', 'xlsx', 'xls'。
        :param save_mode: 保存模式。如果为 'none' 表示暂存不写入文件。如果为 'w+'，即覆盖写入。如果为 'a+'，则为追加写入。
        :param reset_index: 是否重置索引，默认为 False。如果为 True，则在拼接时重置索引。
        :param reset_drop: 是否删除原有索引，默认为 False。如果为 True，则在重置索引时删除原有索引。（仅在 reset_index=True 时生效）
        :param sort_list: 是否对 DataFrame 进行列升序排序。传入一个 可迭代对象，按照优先级对可迭代对象中每个元素进行排序。
        :param sort_ascending: 是否对 DataFrame 进行列升序排序，默认为 True。如果为 False，则为降序排序。（仅在 sort_list 不为 None 时生效）
        Note: 如果确定之前没暂存过相同 os.path.join(folder, filename+'.'+suffix) 的数据，并且本地没有该文件，可以选择不传入 axis 参数；
            否则强烈建议每次都手动传入 axis 参数。
        :return: None
        """
        assert suffix in ['csv', 'xlsx', 'xls'], "不支持的文件后缀名！"
        # 生成保存路径
        if filename is None:
            self.num_df += 1
            filename = f"DataFrame_{self.num_df}"
        if folder is None:
            filepath = os.path.join(self.save_dir, filename + '.' + suffix)
        else:
            filepath = os.path.join(self.save_dir, folder, filename + '.' + suffix)
        # 暂存数据
        if filepath not in self.storage_df:
            new_df = data_df
        else:
            new_df = pd.concat([self.storage_df[filepath][1], data_df], axis=axis)
        if sort_list:  # 对 DataFrame 进行列升序排序
            sort_list = sort_list if isinstance(sort_list, list) else [sort_list,]
            sort_ascending = [sort_ascending,] * len(sort_list) if isinstance(sort_ascending, bool) else sort_ascending
            new_df = new_df.sort_values(by=sort_list, ascending=sort_ascending)
        if reset_index:  # 重置索引
            new_df.reset_index(drop=reset_drop, inplace=True)
        self.storage_df[filepath] = (axis, new_df)
        # 写入数据
        if save_mode in ['w+', 'a+']:
            write_df(filepath, self.storage_df[filepath][1], axis=axis, save_mode=save_mode)
            self.storage_df.pop(filepath)

    def add_text(self, context, end='\n', filename=None, folder=None, suffix='txt', save_mode='none'):
        """
        添加日志到 self.storage_text 中。
        :param context: 文本内容。
        :param end: 每次调用本函数添加文本时，在文本之间添加的分隔符。
        :param filename: 保存文件的文件名，无需添加后缀名。如果为 None，则用 "Text_i" 代替。
        :param folder: 保存文件所在的文件夹名称，如果为 None，则保存到根目录下。
        :param suffix: 保存文件的文件后缀名，默认为 'txt'。可接受任意后缀名，只需使用文本编辑器打开即可。
        :param save_mode: 保存模式。如果为 'none' 表示暂存不写入文件。如果为 'w+'，即覆盖写入。如果为 'a+'，则为追加写入。
        :return: None
        """
        # 生成保存路径
        if filename is None:
            self.num_text += 1
            filename = f"Text_{self.num_text}"
        if folder is None:
            filepath = os.path.join(self.save_dir, filename + '.' + suffix)
        else:
            filepath = os.path.join(self.save_dir, folder, filename + '.' + suffix)
        # 暂存数据
        if filepath not in self.storage_text:
            self.storage_text[filepath] = context
        else:
            self.storage_text[filepath] = self.storage_text[filepath] + end + context
        # 写入数据
        if save_mode in ['w+', 'a+']:
            write_text(filepath, self.storage_text[filepath], save_mode=save_mode)
            self.storage_text.pop(filepath)

    def add_param(self, param_desc=None, param_dict=None, sep='\n', end='\n\n', filename=None, folder=None,
                  suffix='param', save_mode='none'):
        """
        添加参数到 self.parameters 中。
        :param param_desc: 需要保存的参数文本内容。
        :param param_dict: 需要保存的参数字典内容。
        :param sep: 参数文本和参数字典、参数字典之间的分隔符。
        :param end: 每次调用本函数添加参数时，在不同参数之间添加的分隔符。
        :param filename: 保存文件的文件名，无需添加后缀名。如果为 None，则用 "Parameter_i" 代替。
        :param folder: 保存文件所在的文件夹名称，如果为 None，则保存到根目录下。
        :param suffix: 保存文件的文件后缀名，默认为 'param'。可接受任意后缀名，只需使用文本编辑器打开即可。
        :param save_mode: 保存模式。如果为 'none' 表示暂存不写入文件。如果为 'w+'，即覆盖写入。如果为 'a+'，则为追加写入。
        :return: None
        Note: add_param 函数虽然和 add_text 函数类似，但是 add_param 函数对字典形式的参数保存进行了格式化处理。
        """
        # 生成保存路径
        if filename is None:
            self.num_param += 1
            filename = f"Parameter_{self.num_param}"
        if folder is None:
            filepath = os.path.join(self.save_dir, filename + '.' + suffix)
        else:
            filepath = os.path.join(self.save_dir, folder, filename + '.' + suffix)
        # 当前文本情况
        if param_desc and param_dict:
            current_content = param_desc + sep + \
                              sep.join([f"\t{(key+':').ljust(30, ' ')} {value}" for key, value in param_dict.items()])
        elif param_desc and not param_dict:
            current_content = param_desc
        elif not param_desc and param_dict:
            current_content = sep.join([f"\t{(key+':').ljust(30, ' ')} {value}" for key, value in param_dict.items()])
        else:
            raise ValueError("param_desc 和 param_dict 不能同时为空！")
        # 暂存数据
        if filepath not in self.storage_param:
            self.storage_param[filepath] = current_content
        else:
            self.storage_param[filepath] = self.storage_param[filepath] + end + current_content
        # 写入数据
        if save_mode in ['w+', 'a+']:
            write_text(filepath, self.storage_param[filepath], save_mode=save_mode)
            self.storage_param.pop(filepath)

    def write_file(self, data, filename, folder=None, suffix='pkl'):
        """
        使用 pickle 模块以二进制的形式保存文件。
        :param data: 需要保存的内容
        :param filename: 保存文件的文件名，无需添加后缀名。
        :param folder: 保存文件所在的文件夹名称，如果为 None，则保存到根目录下。
        :param suffix: 保存文件的文件后缀名，默认为 'pkl'。可接受任意后缀名，只需使用 pickle.load 打开即可。
        :return: None
        """
        # 生成保存路径
        if folder is None:
            filepath = os.path.join(self.save_dir, filename + '.' + suffix)
        else:
            if not os.path.exists(os.path.join(self.save_dir, folder)):
                os.mkdir(os.path.join(self.save_dir, folder))
            filepath = os.path.join(self.save_dir, folder, filename + '.' + suffix)
        # 使用 pickle 模块以二进制的形式保存文件。
        write_pickle(data=data, file_path=filepath)

    def write(self, save_mode=None):
        """
        将暂存的 DataFrame、文本和参数写到指定的文件夹中。
        :param save_mode: 保存模式。如果为 'w+'，即覆盖写入。如果为 'a+'，则为追加写入（可实现不同 Writer 写入同一个文件）。
                            默认为 None，如果 save_dir 目录在初始化时已经存在，并且 is_delete 为 False，则为 'a+'，否则为 'w+'。
        :return: None
        """
        save_mode = 'a+' if save_mode is None else save_mode
        # 写入 DataFrame 文件。
        for path, (axis, value) in self.storage_df.items():
            write_df(path, value, axis=axis, save_mode=save_mode)
        self.storage_df.clear()  # 删除已写入的数据
        # 写入文本文件 和 参数文件
        assert not set(self.storage_text.keys()) & set(self.storage_param.keys()), "add_text 和 add_param 不能写入同一个文件！"
        storage_string = deepcopy(self.storage_text)
        storage_string.update(self.storage_param)
        for path, value in storage_string.items():
            write_text(path, value, save_mode=save_mode)
        self.storage_text.clear()  # 删除已写入的数据
        self.storage_param.clear()  # 删除已写入的数据

    def draw(self, x, y=None, fmts=None, figsize=None, filename=None, folder=None, suffix='svg', show=True, **kwargs):
        """
        绘制图像
        :param x: 1D 类型的序列数据，例如：list、tuple、ndarray等（要求有 __len__ 魔法方法）；该参数也可以省略，自动匹配长度。
        :param y: 1D 或 2D 类型的序列数据，例如：list、tuple、ndarray等（要求有 __len__ 魔法方法）。
        :param fmts: 图像的格式，例如：['b-', 'm--', 'g-.', 'r:']。
        :param figsize: 图窗大小，例如：(7, 5)。
        :param filename: 图片保存名称，无需添加后缀名。如果为 None，则表示不保存图片。
        :param folder: 保存图像所在的文件夹名称，如果为 None，则保存到根目录下。
        :param suffix: 保存文件的文件后缀名，默认为 'svg'。可接受 matplotlib 支持保存的所有后缀名。
        :param show: 是否展示图像，默认为 True。
        :param kwargs: 其他图窗相关参数。
        :return: None
        """
        def use_svg_display():
            """使用矢量图(SVG)打印图片"""
            backend_inline.set_matplotlib_formats('svg')

        # 生成保存路径
        if (filename is not None) and (folder is None):
            filepath = os.path.join(self.save_dir, filename + '.' + suffix)
        elif (filename is not None) and (folder is not None):
            if not os.path.exists(os.path.join(self.save_dir, folder)):
                os.mkdir(os.path.join(self.save_dir, folder))
            filepath = os.path.join(self.save_dir, folder, filename + '.' + suffix)
        else:
            filepath = None  # 这意味着 filename 为 None，不保存图片。
        # 更新默认配置
        current_axes_set = deepcopy(self.init_axes_set)
        current_axes_set.update(kwargs)
        current_fmts = fmts if fmts is not None else self.fmts
        current_figsize = figsize if figsize is not None else self.figsize
        # 如果 y 是 None，则表示需要把 X 设置为 None，y 设置为纵坐标数据。
        if y is None:
            x, y = None, x
        # 如果 y 是 1D 类型的序列数据，则将其转换为 2D 类型的序列数据。
        if isinstance(y, (list, tuple)) and not isinstance(y[0], (list, tuple, np.ndarray)):
            y = [y]
        elif isinstance(y, np.ndarray) and len(y.shape) == 1:
            y = y.reshape(1, -1)

        # 创建画布和子图
        use_svg_display()  # 使用svg格式的矢量图绘制图片
        nrows, ncols = 1, 1
        fig, axes = plt.subplots(nrows, ncols, figsize=current_figsize)
        # 绘制图像
        for i, seq in enumerate(y):
            if x is None:
                axes.plot(seq, current_fmts[i])
            else:
                axes.plot(x, seq, current_fmts[i])
        # 设置图窗
        for key, value in current_axes_set.items():
            if key == "legend":
                getattr(axes, f"{key}")(value, loc='upper right')
            else:
                getattr(axes, f"set_{key}")(value)
        axes.grid()
        # 展示图像 和 保存图像
        if filename and show:
            plt.savefig(filepath, dpi=None, facecolor='w', edgecolor='w')
            plt.show()
            cprint(f"绘制图像，图片已保存至 {filepath}。", text_color="白色", end='\n')
            self.add_text(f"绘制图像，图片已保存至 {filepath}。",
                          filename="Logs", folder="documents", suffix="log")
        elif filename and not show:
            plt.savefig(filepath, dpi=None, facecolor='w', edgecolor='w')
            plt.close()
            cprint(f"未绘制图像，图片已保存至 {filepath}", text_color="红色", end='\n')
            self.add_text(f"未绘制图像，图片已保存至 {filepath}",
                          filename="Logs", folder="documents", suffix="log")
        elif not filename and show:
            plt.show()
            cprint("图像绘制完成，但未保存图像！", text_color="红色", end='\n')
            self.add_text("图像绘制完成，但未保存图像！",
                          filename="Logs", folder="documents", suffix="log")
        elif not filename and not show:
            plt.close()
            cprint("未绘制图像，也未保存图像！", text_color="红色", end='\n')
            self.add_text("未绘制图像，也未保存图像！",
                          filename="Logs", folder="documents", suffix="log", save_mode='a+')
        else:
            raise ValueError("出现未预知的错误！")


def write_df(path, value, axis=0, save_mode='a+'):
    assert save_mode in ['w+', 'a+'], "保存模式只能为 'w+' 或 'a+'！"
    if not os.path.exists(os.path.dirname(path)):  # 如果文件夹不存在，则创建文件夹
        os.mkdir(os.path.dirname(path))
    if (save_mode == 'a+') and os.path.exists(path):  # 追加写入
        if path.endswith('.csv'):  # csv 格式数据
            exist_file = pd.read_csv(path, index_col=0)
        elif path.endswith('.xlsx') or path.endswith('.xls'):  # excel 格式数据
            exist_file = pd.read_excel(path, index_col=0)
        else:
            raise ValueError(f"不支持的文件格式：{os.path.basename(path)}")
        new_file = pd.concat([exist_file, value], axis=axis)
        new_file.to_csv(path, index=True) if path.endswith('.csv') else new_file.to_excel(path, index=True)
    else:  # 覆盖写入
        if path.endswith('.csv'):
            value.to_csv(path, index=True)
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            value.to_excel(path, index=True)
        else:
            raise ValueError(f"不支持的文件格式：{os.path.basename(path)}")


def write_text(path, value, save_mode='a+'):
    assert save_mode in ['w+', 'a+'], "保存模式只能为 'w+' 或 'a+'！"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 当前时间
    current_time = '-' * 100 + '\n' + current_time + '\n' + '-' * 100 + '\n'
    if not os.path.exists(os.path.dirname(path)):  # 如果文件夹不存在，则创建文件夹
        os.mkdir(os.path.dirname(path))
    if (save_mode == 'a+') and os.path.exists(path):  # 追加写入
        with open(path, 'a+', encoding='utf-8') as f:
            f.write('\n\n' + current_time + value)
    else:  # 覆盖写入
        with open(path, 'w+', encoding='utf-8') as f:
            f.write(current_time + value)
