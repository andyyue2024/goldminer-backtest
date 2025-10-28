# -*- coding: utf-8 -*-
import time
from datetime import datetime, timedelta
import os
from gm.api import *
import multiprocessing as mp
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import sqlite3
import warnings

warnings.filterwarnings('ignore')

"""
ai labx test2_3 - Optuna 版本

使用 Optuna 进行超参数优化，替换原来的多进程网格搜索

基本思想：设定所需优化的参数数值范围及步长，将参数数值循环输入进策略，进行遍历回测，
        记录每次回测结果和参数，根据某种规则将回测结果排序，找到最好的参数。
1、定义策略函数  AILabxStrategy
2、多进程循环输入参数数值
3、获取回测报告，生成DataFrame格式
4、排序
"""


class AILabxTool:
    def __init__(self, now=None, w_aa=0.1, w_bb=0.2, w_cc=1,
                 win_trend_score=25,
                 win_roc_score1=5, win_roc_score2=10,
                 win_ma_score1=5, win_ma_score2=18):
        self.now = now
        self.last_day = AILabxTool.get_time_Ymd(self.now - timedelta(days=1)) if self.now else None
        self.all_data = None
        self.w_aa = w_aa
        self.w_bb = w_bb
        self.w_cc = w_cc
        self.win_trend_score = win_trend_score
        self.win_roc_score1 = win_roc_score1
        self.win_roc_score2 = win_roc_score2
        self.win_ma_score1 = win_ma_score1
        self.win_ma_score2 = win_ma_score2

    def set_parameter(self, w_aa=0.4, w_bb=0.2, w_cc=1):
        self.w_aa = w_aa
        self.w_bb = w_bb
        self.w_cc = w_cc

    @property
    def now(self):
        return self._now

    @now.setter
    def now(self, new_value):
        self._now = new_value
        self.last_day = AILabxTool.get_time_Ymd(self.now - timedelta(days=1)) if self.now else None

    def get_score(self, symbol):
        # trend_score2 = self.trend_score2(symbol, "close", 25)
        trend_score = self.trend_score(symbol, "close", self.win_trend_score)
        roc_score1 = self.roc(symbol, "close", self.win_roc_score1)
        roc_score2 = self.roc(symbol, "close", self.win_roc_score2)
        ma_score1 = self.ma(symbol, "volume", self.win_ma_score1)
        ma_score2 = self.ma(symbol, "volume", self.win_ma_score2)
        aa = trend_score
        bb = roc_score1 + roc_score2
        cc = ma_score1 / ma_score2
        score = aa * self.w_aa + bb * self.w_bb + cc * self.w_cc
        return score, aa, bb, cc

    def trend_score(self, symbol, field='close', window=25):
        """
        向量化计算趋势评分：年化收益率 × R平方
        :param symbol:
        :param field:收盘价序列（np.array或pd.Series）
        :param window: 计算窗口长度，默认25天
        :return: 趋势评分数组，长度与输入相同，前period-1位为NaN
        """
        # 确保窗口大小合适
        if window < 2:
            raise ValueError("窗口大小至少为2")
        history_data = None
        if self.all_data is None:
            history_data = history_n(
                symbol=symbol,
                frequency='1d',
                count=window,
                end_time=self.last_day,
                fields=field,
                fill_missing="last",
                adjust=ADJUST_PREV,
                df=True
            )
        else:
            # filtered_df = df[(df['A'] > 2) & (df['B'] < 8)]
            history_data = self.all_data[(self.all_data['bob'] <= self.last_day) & (self.all_data['symbol'] == symbol)]
            history_data = history_data.tail(window)

        # 提取收盘价序列
        data = np.asarray(history_data[field].values)
        # close = history_data[field].values
        if len(data) < window:
            return np.nan  # 数据不足时返回 NaN

        y = np.log(data)
        windows = np.lib.stride_tricks.sliding_window_view(y, window_shape=window)
        x = np.arange(window)

        # 预计算固定值
        n = window
        sum_x = x.sum()
        sum_x2 = (x ** 2).sum()
        denominator = n * sum_x2 - sum_x ** 2

        # 滑动窗口统计量
        sum_y = windows.sum(axis=1)
        sum_xy = (windows * x).sum(axis=1)

        # 回归系数
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # 年化收益率
        annualized_returns = np.exp(slope * 250) - 1

        # R平方计算
        y_pred = slope[:, None] * x + intercept[:, None]
        residuals = windows - y_pred
        ss_res = np.sum(residuals ** 2, axis=1)

        sum_y2 = np.sum(windows ** 2, axis=1)
        ss_tot = sum_y2 - (sum_y ** 2) / n
        r_squared = 1 - (ss_res / ss_tot)
        r_squared = np.nan_to_num(r_squared, nan=0.0)  # 处理零方差情况

        # 综合评分
        score = annualized_returns * r_squared

        # 对齐原始序列长度
        # full_score = np.full_like(y, np.nan)
        # full_score = pd.Series(index=data.index, dtype='float64')
        # full_score[window - 1:] = score
        return score[0]

    def trend_score1(self, symbol, field='close', window=20):
        """
        计算价格序列的趋势评分（斜率 × 拟合度 R²）
        参数:
            window (int): 计算窗口长度
        返回:
            float: 趋势评分值
        """
        # 确保窗口大小合适
        if window < 2:
            raise ValueError("窗口大小至少为2")
        history_data = None
        if self.all_data is None:
            history_data = history_n(
                symbol=symbol,
                frequency='1d',
                count=window,
                end_time=self.last_day,
                fields=field,
                fill_missing="last",
                adjust=ADJUST_PREV,
                df=True
            )
        else:
            # filtered_df = df[(df['A'] > 2) & (df['B'] < 8)]
            history_data = self.all_data[(self.all_data['bob'] <= self.last_day) & (self.all_data['symbol'] == symbol)]
            history_data = history_data.tail(window)

        # 提取收盘价序列
        data = np.asarray(history_data[field].values)
        # close = history_data[field].values
        if len(data) < window:
            return np.nan  # 数据不足时返回 NaN

        y = np.array(data[-window:])  # 取最近 window 期数据
        x = np.arange(1, window + 1)  # 时间序列

        # 计算斜率
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        slope = numerator / denominator if denominator != 0 else 0

        # 计算 R² 拟合度
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2

        return slope * r_squared

    def trend_score2(self, symbol, field='close', window=20):
        """
        计算价格序列的趋势评分（斜率 × 拟合度 R²）
        参数:
            window (int): 计算窗口长度
        返回:
            float: 趋势评分值
        """
        # 确保窗口大小合适
        if window < 2:
            raise ValueError("窗口大小至少为2")
        history_data = None
        if self.all_data is None:
            history_data = history_n(
                symbol=symbol,
                frequency='1d',
                count=window,
                end_time=self.last_day,
                fields=field,
                fill_missing="last",
                adjust=ADJUST_PREV,
                df=True
            )
        else:
            # filtered_df = df[(df['A'] > 2) & (df['B'] < 8)]
            history_data = self.all_data[(self.all_data['bob'] <= self.last_day) & (self.all_data['symbol'] == symbol)]
            history_data = history_data.tail(window)

        # 提取收盘价序列
        data = np.asarray(history_data[field].values)
        # close = history_data[field].values
        if len(data) < window:
            return np.nan  # 数据不足时返回 NaN

        # 提取最近window个收盘价
        data_values = data[-window:]

        # 计算时间序列（作为x轴）
        time = np.arange(window)

        # 使用numpy的polyfit函数进行线性回归
        coefficients = np.polyfit(time, data_values, 1)
        slope = coefficients[0]  # 斜率
        intercept = coefficients[1]  # 截距

        # 计算拟合值
        fit_values = np.polyval(coefficients, time)

        # 计算拟合度（R^2）
        ss_res = np.sum((data_values - fit_values) ** 2)
        ss_tot = np.sum((data_values - np.mean(data_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # 返回斜率乘以拟合度
        return slope * r_squared

    def roc(self, symbol, field='close', window=20):
        history_data = None
        if self.all_data is None:
            history_data = history_n(
                symbol=symbol,
                frequency='1d',
                count=window + 1,  # 获取n+1根K线以确保有n期的差值
                end_time=self.last_day,
                fields=field,
                fill_missing="last",
                adjust=ADJUST_PREV,
                df=True
            )
        else:
            # filtered_df = df[(df['A'] > 2) & (df['B'] < 8)]
            history_data = self.all_data[(self.all_data['bob'] <= self.last_day) & (self.all_data['symbol'] == symbol)]
            history_data = history_data.tail(window + 1)

        # 提取收盘价序列
        data = np.asarray(history_data[field].values)

        # 检查数据是否足够
        if len(data) < window + 1:
            return np.nan

        # 计算ROC
        # se = history_data[field]
        # ret = se / se.shift(window) - 1
        roc_value = (data[-1] - data[0]) / data[0]  # 最新价 vs n天前价
        return roc_value

    def ma(self, symbol, field='close', window=20):
        """
        计算指定窗口大小的移动平均值
        :param symbol:
        :param field:
        :param window: int，移动平均窗口大小
        :return: float，移动平均值
        """
        history_data = None
        if self.all_data is None:
            history_data = history_n(
                symbol=symbol,
                frequency='1d',
                count=window,  # 获取n根K线
                end_time=self.last_day,
                fields=field,
                fill_missing="last",
                adjust=ADJUST_PREV,
                df=True
            )
        else:
            # filtered_df = df[(df['A'] > 2) & (df['B'] < 8)]
            history_data = self.all_data[(self.all_data['bob'] <= self.last_day) & (self.all_data['symbol'] == symbol)]
            history_data = history_data.tail(window)

        # 提取收盘价序列
        data = np.asarray(history_data[field].values)
        # X = history_data[field]
        # X.ffill(inplace=True)
        # y = X.rolling(window=window).mean()

        if len(data) < window:
            return np.nan  # 或 raise ValueError("数据长度不足")
        # ma_value = np.mean(data)
        return np.mean(data)

    def get_all_data(self, symbol_list, start_time, end_time):
        # original_datetime_str = "2024-07-31 09:31:00+08:00"
        dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        pre_start_time = str(dt - timedelta(days=100))[:19]
        all_data = history(symbol_list, frequency='1d', start_time=pre_start_time, end_time=end_time,
                           fill_missing='last', adjust=ADJUST_PREV, df=True)

        all_data['bob'] = all_data['bob'].apply(lambda x: AILabxTool.get_time_Ymd(x))
        all_data['eob'] = all_data['eob'].apply(lambda x: AILabxTool.get_time_Ymd(x))
        self.all_data = all_data

        # # write file
        # scores = []
        # aas = []
        # bbs = []
        # ccs = []
        # for row in all_data.itertuples():
        #     self.last_day = row.bob
        #     score, aa, bb, cc = self.get_score(row.symbol)
        #     scores.append(score)
        #     aas.append(aa)
        #     bbs.append(bb)
        #     ccs.append(cc)
        #
        # all_data["score"] = scores
        # all_data["aa"] = aas
        # all_data["bb"] = bbs
        # all_data["cc"] = ccs
        # input_file_path = f'./data/ai_labx_tools_input_data.xlsx'
        # all_data.to_excel(input_file_path, index=False)
        #
        # description = all_data.describe()
        # print(description)
        # description_str = description.to_string()
        # # 将字符串写入到文件
        # output_description = f'./data/ai_labx_tools_describe_output.txt'
        # with open(output_description, 'w', encoding='utf-8') as file:
        #     file.write(description_str)

    @staticmethod
    def get_time_Ymd(dt: datetime, pre_format: str = "%Y-%m-%d"):
        """
        transfer "2024-07-31 09:31:00+08:00" to "2024-07-31"
        :param pre_format:
        :param dt:
        :return:
        """
        # original_datetime_str = "2024-07-31 09:31:00+08:00"
        # dt = datetime.strptime(original_datetime_str, "%Y-%m-%d %H:%M:%S%z")
        # remove timezone
        dt_no_timezone = dt.replace(tzinfo=None)
        # dt_no_timezone_str = dt_no_timezone.strftime("%Y-%m-%d %H:%M:%S")
        dt_no_timezone_str = dt_no_timezone.strftime(pre_format)
        return dt_no_timezone_str


class AILabxStrategy:
    def __init__(self, context, white_list: list = None, max_count: int = 1,
                 w_aa=0.2, w_bb=1.5, w_cc=1,
                 win_trend_score=25,
                 win_roc_score1=5, win_roc_score2=10,
                 win_ma_score1=5, win_ma_score2=18,
                 w_dd=0.16, w_fd=20):
        self.now = None
        self.context = context
        self.white_list = list(white_list)
        self.max_count = max_count
        self.ailabx = AILabxTool(w_aa=w_aa, w_bb=w_bb, w_cc=w_cc,
                                 win_trend_score=win_trend_score,
                                 win_roc_score1=win_roc_score1,
                                 win_roc_score2=win_roc_score2,
                                 win_ma_score1=win_ma_score1,
                                 win_ma_score2=win_ma_score2)
        if context.mode == MODE_BACKTEST:
            self.ailabx.get_all_data(self.white_list, context.backtest_start_time, context.backtest_end_time)
        self.w_dd = w_dd
        self.w_fd = w_fd
        self.last_symbol = ""

    def filter(self, in_list: list = None):
        if in_list is None:
            in_list = []
        return in_list + [item for item in self.white_list if item not in in_list]

    def filter_for_selling(self, in_list: list = None):
        # filter symbols that should be selling before sorting
        if in_list is None:
            in_list = []
        return [item for item in in_list if not self.should_sell(item)]

    def sort(self, in_list: list, ascending=False) -> list:
        symbol_list = list(in_list)
        scores = []
        for symbol in symbol_list:
            score, _, _, _ = self.ailabx.get_score(symbol)
            scores.append(score)

        df = pd.DataFrame([])
        df["symbol"] = symbol_list
        df["score"] = scores
        df = df.dropna()
        df = df.sort_values(["score"], ascending=ascending)
        return list(df["symbol"].to_list())

    def filter_top(self, in_list: list, top_count=1):
        if not in_list:
            return []
        return in_list[0:top_count]

    def try_to_order(self, in_list: list) -> list:
        positions = self.context.account().positions(side=PositionSide_Long)
        hold_symbol_list = [p.symbol for p in positions]
        if len(in_list) > 0:
            # print("target: ", in_list, "; already hold: ", hold_symbol_list)
            pass
        if len(in_list) == 1 and self.last_symbol == in_list[0]:
            return []
        for position in positions:
            self.sell_target_position(position)

        hold_target_list = []
        for target in in_list:
            self.buy_target(target)
            hold_target_list.append(target)
        return hold_target_list

    def try_to_order1(self, in_list: list) -> list:
        positions = self.context.account().positions(side=PositionSide_Long)
        hold_symbol_list = [p.symbol for p in positions]
        if len(in_list) > 0:
            print("target: ", in_list, "; already hold: ", hold_symbol_list)
        if len(in_list) == 1 and self.last_symbol == in_list[0]:
            if self.should_sell(self.last_symbol):
                self.sell_target(self.last_symbol)
                # self.last_symbol = ""
            return []
        if len(positions) > 0:
            # print("order_close_all: ", hold_symbol_list)
            order_close_all()

        hold_target_list = []
        for target in in_list:
            if not self.should_sell(target):
                self.buy_target(target)
                hold_target_list.append(target)
        return hold_target_list

    def try_to_order2(self, in_list: list) -> list:
        to_buy_list = []
        positions = self.context.account().positions(side=PositionSide_Long)
        hold_symbol_list = [p.symbol for p in positions]
        if len(in_list) > 0:
            print("target: ", in_list, "; already hold: ", hold_symbol_list)
        for hold_symbol in hold_symbol_list:
            if (hold_symbol not in in_list or
                    (hold_symbol in in_list and self.should_sell(hold_symbol))):  # 命中强制卖出条件
                self.sell_target(hold_symbol)

        for target_symbol in in_list:
            if (not self.should_sell(target_symbol)) and (target_symbol not in hold_symbol_list):
                self.buy_target(target_symbol)
                to_buy_list.append(target_symbol)
        return to_buy_list

    def sell_target(self, target: str):
        # print("sell_target: ", target)
        # order_target_percent(symbol=target, percent=0, order_type=OrderType_Limit,
        #                      position_side=PositionSide_Long, price=self.latest_price(target))
        order_percent(symbol=target, percent=1. / self.max_count, side=OrderSide_Sell, order_type=OrderType_Limit,
                      position_effect=PositionEffect_Close, price=self.latest_price(target))

    def sell_target_position(self, p):
        target = p.symbol
        # print("sell_target: ", target)
        order_volume(symbol=target, volume=p.volume, side=OrderSide_Sell, order_type=OrderType_Market,
                     position_effect=PositionEffect_Close, price=self.latest_price(target))

    def buy_target(self, target: str):
        # print("buy_target: ", target)
        # self.last_symbol = target
        # order_target_percent(symbol=target, percent=1. / self.max_count, order_type=OrderType_Limit,
        #                      position_side=PositionSide_Long, price=self.latest_price(target))
        order_percent(symbol=target, percent=1. / self.max_count, side=OrderSide_Buy, order_type=OrderType_Market,
                      position_effect=PositionEffect_Open, price=self.latest_price(target))

    def should_sell(self, target: str):
        return self.ailabx.roc(target, "close", self.w_fd) > self.w_dd
        # return False

    @staticmethod
    def latest_price(symbol):
        current_data = current(symbols=symbol)
        return current_data[0]["price"]

    def execute(self, now):
        order_cancel_all()

        # update info
        self.now = now
        self.ailabx.now = self.now
        positions = self.context.account().positions(side=PositionSide_Long)
        if len(positions) > 0:
            self.last_symbol = positions[0].symbol
        else:
            self.last_symbol = ""

        ret_list = self.filter()
        ret_list = self.filter_for_selling(ret_list)
        ret_list = self.sort(ret_list)
        # print("sort: ", ret_list)
        ret_list = self.filter_top(ret_list)
        ret_list = self.try_to_order(ret_list)
        return ret_list


def init(context):
    context.start_time = time.time()
    print("Start...")
    context.num = 1
    context.symbol = "AAA"
    context.ai_labx_strategy = AILabxStrategy(context=context, white_list=list(index_list.keys()),
                                              w_aa=context.paras['w_aa'],
                                              w_bb=context.paras['w_bb'],
                                              w_cc=context.paras['w_cc'],
                                              win_trend_score=context.paras['win_trend_score'],
                                              win_roc_score1=context.paras['win_roc_score1'],
                                              win_roc_score2=context.paras['win_roc_score2'],
                                              win_ma_score1=context.paras['win_ma_score1'],
                                              win_ma_score2=context.paras['win_ma_score2'],
                                              w_dd=context.paras['w_dd'], w_fd=context.paras['w_fd']
                                              )

    schedule(schedule_func=algo, date_rule='1d', time_rule='14:52:00')
    # subscribe(symbols=list(index_list.keys()), frequency="60s")


def algo(context):
    # print(context.now, " algo...")
    try:
        context.ai_labx_strategy.execute(context.now)
    except Exception as ee:
        print(f"Trial {context.trial_number} failed, algo: {ee}")


def on_order_status(context, order):
    # 订单状态更新处理
    if order.status == OrderStatus_Filled:  # 完全成交
        # print(f"order OrderStatus_Filled, {order}")
        pass
    elif order.status in [OrderStatus_Canceled, OrderStatus_PartiallyFilled]:  # 已撤单/部分成交撤单
        print(f"order OrderStatus_Canceled or OrderStatus_PartiallyFilled, {order.order_id}")
        # self.handle_order_retry(order)
    elif order.status in [OrderStatus_Rejected]:  # 已拒绝
        # print(f"order OrderStatus_Rejected, {order.ord_rej_reason}")
        pass


def on_backtest_finished(context, indicator):
    print(f"Trial  {context.trial_number}: Done! From start, time elapsed: {time.time() - context.start_time} seconds")
    # print(f"{context.symbol} backtest finished: ", indicator)
    # 回测业绩指标数据
    data = [
        indicator['pnl_ratio'], indicator['pnl_ratio_annual'], indicator['sharp_ratio'], indicator['max_drawdown'],
        context.paras['w_aa'], context.paras['w_bb'], context.paras['w_cc'],
        context.paras['win_trend_score'],
        context.paras['win_roc_score1'],
        context.paras['win_roc_score2'],
        context.paras['win_ma_score1'],
        context.paras['win_ma_score2'],
        context.paras['w_dd'], context.paras['w_fd']
    ]


    # 将超参加入context.result
    context.result.append(data)


index_list = {
    # List
    "SHSE.513520": "日经ETF",
    "SHSE.513290": "纳指生物科技ETF",
    "SZSE.159509": "纳指科技ETF",
    "SHSE.513030": "德国ETF",
    "SHSE.513100": "纳指ETF",
    "SZSE.159915": "创业板ETF",
    "SHSE.512100": "中证1000ETF",
    "SHSE.563300": "中证2000ETF",
    "SHSE.560800": "数字经济ETF",
    "SHSE.513040": "港股通互联网ETF",
    "SHSE.518880": "黄金ETF",
    "SZSE.159560": "芯片50ETF",
    "SZSE.159819": "人工智能ETF",
    "SZSE.162719": "石油LOF",
    "SHSE.513330": "恒生互联网ETF",
    "SHSE.513090": "香港证券ETF",
    "SHSE.513380": "恒生科技ETF龙头",
    "SHSE.561600": "消费电子ETF",
    "SHSE.512480": "半导体ETF",
    "SZSE.159752": "新能源龙头ETF",
    "SZSE.159761": "新材料50ETF",
    "SHSE.588000": "科创50ETF",
    "SHSE.513500": "标普500ETF",
    "SHSE.588100": "科创信息技术ETF",
    "SHSE.515030": "新能源车ETF",
    "SHSE.515880": "通信ETF",
    "SHSE.515790": "光伏ETF",

}

index_list1 = {
    # List
    "SHSE.513290": "纳指生物科技ETF",
    "SHSE.513520": "日经ETF",
    "SZSE.159509": "纳指科技ETF",
    "SHSE.513030": "德国ETF",
    "SZSE.159915": "创业板ETF",
    "SHSE.512100": "中证1000ETF",
    "SHSE.563300": "中证2000ETF",
    "SHSE.588100": "科创信息技术ETF",  # little
    "SHSE.513040": "港股通互联网ETF",
    "SHSE.563000": "中国A50ETF",
    "SZSE.159560": "芯片50ETF",
    "SZSE.159819": "人工智能ETF",
    "SZSE.162719": "石油LOF",
    "SHSE.518880": "黄金ETF",
    "SHSE.513330": "恒生互联网ETF",
    "SHSE.513090": "香港证券ETF",
    # "SZSE.159505": "国证2000指数ETF",  # very little
    "SHSE.513180": "恒生科技指数ETF",
    "SHSE.513130": "恒生科技ETF",
    "SZSE.159857": "光伏ETF",
    "SHSE.512480": "半导体ETF",
    "SHSE.561600": "消费电子ETF",  # little
    "SHSE.513100": "纳指ETF",
    "SHSE.588000": "科创50ETF",
    "SHSE.513500": "标普500ETF",
    "SZSE.159619": "基建ETF",  # little
    "SHSE.515880": "通信ETF",
    "SHSE.513380": "恒生科技ETF龙头",
    "SHSE.510300": "沪深300ETF",
    # "SHSE.510050": "上证50ETF",
    "SHSE.510500": "中证500ETF",
    "SHSE.588080": "科创板50ETF",
    # "SHSE.512890": "红利低波ETF",
    "SHSE.513120": "港股创新药ETF",
    # "SHSE.511380": "可转债ETF",
    # "SHSE.562500": "机器人ETF",
    # "SHSE.512690": "酒ETF",
    "SZSE.159920": "恒生ETF",
    # "SZSE.159928": "消费ETF",

}


def run_strategy(paras: dict, trial_number: int):
    # 导入上下文
    from gm.model.storage import context
    # 用context传入参数
    context.paras = paras
    context.trial_number = trial_number
    # context.result用以存储超参
    context.result = []

    run(
        # strategy_id='fc2bd0ce-b24b-11f0-a692-206a8a674f71',
        strategy_id='5f98da58-a9b8-11f0-9808-84a938b5ecc6',  # gfgm
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        token='6860051c58995ae01c30a27d5b72000bababa8e6',  # gfgm
        # token='c8bd4de742240da9483aecd05a2f5e52900786eb',
        backtest_start_time="2023-11-21 09:30:00",
        backtest_end_time='2025-10-28 15:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0005,
        backtest_commission_unit=1,
        backtest_slippage_ratio=0.0001,
        backtest_marginfloat_ratio1=0.2,
        backtest_marginfloat_ratio2=0.4,
        backtest_match_mode=1)
    return context.result


def write_to_file(results, output_file_path=f'./data/AILabxOptunaTest2_info.xlsx'):
    info = pd.DataFrame(results,
                        columns=[
                            'objective_value', 'pnl_ratio', 'pnl_ratio_annual',
                            'sharp_ratio', 'max_drawdown',
                            'w_aa', 'w_bb','w_cc',
                            'win_trend_score',
                            'win_roc_score1',
                            'win_roc_score2',
                            'win_ma_score1',
                            'win_ma_score2',
                            'w_dd', 'w_fd'
                        ])
    print(f"row length: {len(info)}")
    # info.to_csv('./data/info.csv', index=False)
    info.to_excel(output_file_path, index=False)


def format_trial_result(results, trial):
    results.append([
        trial.values[0] if trial.values else -float('inf'),  # 目标值
        trial.user_attrs.get('pnl_ratio', 0),
        trial.user_attrs.get('pnl_ratio_annual', 0),
        trial.user_attrs.get('sharp_ratio', 0),
        trial.user_attrs.get('max_drawdown', 0),
        trial.params.get('w_aa', 0),
        trial.params.get('w_bb', 0),
        1,  # w_cc 固定为1
        trial.params.get('win_trend_score', 0),
        trial.params.get('win_roc_score1', 0),
        trial.params.get('win_roc_score2', 0),
        trial.params.get('win_ma_score1', 0),
        trial.params.get('win_ma_score2', 0),
        trial.params.get('w_dd', 0),
        trial.params.get('w_fd', 0)
    ])

    return results


def objective(trial):
    """
    Optuna 目标函数
    返回需要最大化的指标（年化收益率）
    """
    # 定义搜索空间
    w_aa = trial.suggest_float('w_aa', -0.20, 0.60)   # default: 0.1
    w_bb = trial.suggest_float('w_bb', 0.10, 2.00)   # default: 1.6

    win_trend_score = trial.suggest_int('win_trend_score', 15, 35) # default: 25
    win_roc_score1 = trial.suggest_int('win_roc_score1', 3, 7) # default: 5
    win_roc_score2 = trial.suggest_int('win_roc_score2', 8, 20) # default: 10
    win_ma_score1 = trial.suggest_int('win_ma_score1', 3, 7) # default: 5
    win_ma_score2 = trial.suggest_int('win_ma_score2', 10, 35) # default: 18

    w_dd = trial.suggest_float('w_dd', 0.10, 0.21)      # default: 0.2
    # w_fd = trial.suggest_categorical('w_fd', [18])  # 固定值，但保持参数形式
    w_fd = trial.suggest_int('w_fd', 15, 21)     # default: 18

    paras = {
        "w_aa": w_aa,
        "w_bb": w_bb,
        "w_cc": 1,
        "win_trend_score": win_trend_score,
        "win_roc_score1": win_roc_score1,
        "win_roc_score2": win_roc_score2,
        "win_ma_score1": win_ma_score1,
        "win_ma_score2": win_ma_score2,
        "w_dd": w_dd,
        "w_fd": w_fd
    }

    print(f"试验 {trial.number}: 测试参数 w_aa={w_aa:.2f}, w_bb={w_bb:.2f}, w_cc=1, "
          f"win_trend_score={win_trend_score}, "
          f"win_roc_score1={win_roc_score1}, "
          f"win_roc_score2={win_roc_score2}, "
          f"win_ma_score1={win_ma_score1}, "
          f"win_ma_score2={win_ma_score2}， "
          f"w_dd={w_dd:.2f}, w_fd={w_fd}")

    try:
        # 运行策略
        result =  None
        parameters = [(paras, trial.number)]
        with mp.Pool(processes=1) as pool:
            result = pool.starmap(run_strategy, parameters)

        # processes = mp.cpu_count()
        # result = run_strategy(paras, trial.number)
        print(f"试验 {trial.number} result: {result}")
        if result and len(result) > 0:
            metrics = result[0][0]  # 获取回测指标
            pnl_ratio_annual = metrics[1]  # 年化收益率

            # 设置用户属性，用于后续分析
            trial.set_user_attr("pnl_ratio", metrics[0])
            trial.set_user_attr("pnl_ratio_annual", metrics[1])
            trial.set_user_attr("sharp_ratio", metrics[2])
            trial.set_user_attr("max_drawdown", metrics[3])

            print(f"试验 {trial.number} 完成: 年化收益率 = {pnl_ratio_annual:.4f}")
            return pnl_ratio_annual
        else:
            print(f"试验 {trial.number} 无结果")
            return -float('inf')

    except Exception as e:
        print(f"试验 {trial.number} 失败: {e}")
        return -float('inf')


class SaveResultsCallback:
    """回调函数用于定期保存结果"""

    def __init__(self, save_interval=10):
        self.save_interval = save_interval
        self.results = []

    def __call__(self, study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.results = format_trial_result(self.results, trial)
            # 定期保存
            if len(self.results) % self.save_interval == 0:
                write_to_file(self.results, f'./data/AILabxOptunaTest2_interim_{len(self.results)}.xlsx')
                print(f"已保存 {len(self.results)} 个试验的中间结果")


def create_or_load_study(storage_url, study_name):
    """创建或加载study，确保进程安全"""
    try:
        # 尝试加载已存在的study
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
            sampler=TPESampler(seed=42)
        )
        print(f"加载已存在的study: {study_name}")
    except KeyError:
        # 如果study不存在，创建新的
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        print(f"创建新的study: {study_name}")

    return study


if __name__ == '__main__':
    # 使用数据库存储确保进程安全
    storage_url = "sqlite:///AILabxOptunaStudy3_001.db"
    study_name = "AILabxOptunaStudy"

    # 确保数据目录存在
    os.makedirs('./data', exist_ok=True)

    # 创建保存结果的回调实例
    save_callback = SaveResultsCallback(save_interval=10)

    # 创建或加载study
    study = create_or_load_study(storage_url, study_name)

    print('开始 Optuna 超参数优化...')
    print(f"当前study包含 {len(study.trials)} 个试验")

    # 运行优化
    study.optimize(
        objective,
        n_trials=20,  # 试验次数，可以根据需要调整
        n_jobs=16,  # 并行任务数，根据CPU核心数调整
        callbacks=[save_callback],
        show_progress_bar=True,
        gc_after_trial=True  # 每次试验后执行垃圾回收
    )

    print('优化完成!')

    # 输出最佳结果
    print(f"最佳试验: {study.best_trial.number}")
    print(f"最佳参数: {study.best_trial.params}")
    print(f"最佳年化收益率: {study.best_value:.4f}")

    # 保存最终结果
    final_results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            final_results = format_trial_result(final_results, trial)

    # 保存到文件
    write_to_file(final_results, f'./data/AILabxOptunaTest2_final_{len(final_results)}.xlsx')

    # 保存最佳参数
    best_params_df = pd.DataFrame([study.best_trial.params])
    best_params_df['best_value'] = study.best_value
    best_params_df.to_excel('./data/AILabxOptunaTest3_best_params.xlsx', index=False)

    print(f"总共完成 {len(study.trials)} 个试验")
    print("结果已保存到文件")

    # 可视化结果（可选）
    # 只有在有足够试验时才绘制参数重要性
    if len(study.trials) > 1:
        try:
            import optuna.visualization as vis
            from datetime import datetime

            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 优化历史
            fig1 = vis.plot_optimization_history(study)
            fig1.write_image(f'./data/optimization_history_{timestamp}.png')
            print("优化历史图已保存")

            # 参数重要性
            fig2 = vis.plot_param_importances(study)
            fig2.write_image(f'./data/param_importances_{timestamp}.png')
            print("参数重要性图已保存")

            # 平行坐标图
            fig3 = vis.plot_parallel_coordinate(study)
            fig3.write_image(f'./data/parallel_coordinate_{timestamp}.png')
            print("平行坐标图已保存")

        except (ValueError, ImportError) as e:
            print(f"可视化错误: {e}")
    else:
        print("试验数量不足，无法生成参数重要性图")