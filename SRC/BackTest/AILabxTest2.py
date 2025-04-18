# -*- coding: utf-8 -*-
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from gm.api import *

"""
ai labx test2

"""


class AILabxTool:
    def __init__(self, now=None, w_aa=0.1, w_bb=0.2, w_cc=1):
        self.now = now
        self.last_day = AILabxTool.get_time_Ymd(self.now - timedelta(days=1)) if self.now else None
        self.all_data = None
        self.w_aa = w_aa
        self.w_bb = w_bb
        self.w_cc = w_cc

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
        trend_score = self.trend_score(symbol, "close", 26)
        roc_score1 = self.roc(symbol, "close", 5)
        roc_score2 = self.roc(symbol, "close", 10)
        ma_score1 = self.ma(symbol, "volume", 5)
        ma_score2 = self.ma(symbol, "volume", 20)
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
    def __init__(self, context, white_list: list = None, max_count: int = 1, w_aa=0.1, w_bb=0.1, w_cc=1, w_dd=0.20):
        self.now = None
        self.context = context
        self.white_list = list(white_list)
        self.max_count = max_count
        self.ailabx = AILabxTool(w_aa=w_aa, w_bb=w_bb, w_cc=w_cc)
        if context.mode == MODE_BACKTEST:
            self.ailabx.get_all_data(self.white_list, context.backtest_start_time, context.backtest_end_time)
        self.w_dd = w_dd
        self.last_symbol = ""

    def filter(self, in_list: list=None):
        if in_list is None:
            in_list = []
        return in_list + [item for item in self.white_list if item not in in_list]

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
        if len(in_list) == 1 and self.last_symbol == in_list[0]:
            if self.should_sell(self.last_symbol):
                self.sell_target(self.last_symbol)
                self.last_symbol = ""
            return []
        order_close_all()
        print("order_close_all: ")
        hold_target_list = []
        for target in in_list:
            if not self.should_sell(target):
                self.buy_target(target)
                hold_target_list.append(target)
        return hold_target_list

    def sell_target(self, target: str):
        print("sell_target: ", target)
        order_target_percent(symbol=target, percent=0, order_type=OrderType_Market,
                             position_side=PositionSide_Long, price=self.latest_price(target))

    def buy_target(self, target: str):
        print("buy_target: ", target)
        self.last_symbol = target
        order_target_percent(symbol=target, percent=1. / self.max_count, order_type=OrderType_Market,
                             position_side=PositionSide_Long, price=self.latest_price(target))

    def should_sell(self, target: str):
        return self.ailabx.roc(target, "close", 18) > self.w_dd
        # return False

    @staticmethod
    def latest_price(symbol):
        current_data = current(symbols=symbol)
        return current_data[0]["price"]

    def execute(self, now):
        self.now = now
        self.ailabx.now = self.now
        ret_list = self.filter()
        ret_list = self.sort(ret_list)
        ret_list = self.filter_top(ret_list)
        ret_list = self.try_to_order(ret_list)
        return ret_list


def init(context):
    context.start_time = time.time()
    print("Start...")
    context.num = 1
    context.symbol = "AAA"
    context.ai_labx_strategy = AILabxStrategy(context=context, white_list=list(index_list.keys()))

    schedule(schedule_func=algo, date_rule='1d', time_rule='09:31:00')


def algo(context):
    print(context.now, " algo...")
    context.ai_labx_strategy.execute(context.now)


def on_order_status(context, order):
    # 订单状态更新处理
    if order.status == OrderStatus_Filled:  # 完全成交
        print(f"order OrderStatus_Filled, {order}")
    elif order.status in [OrderStatus_Canceled, OrderStatus_PartiallyFilled]:  # 已撤单/部分成交撤单
        print(f"order OrderStatus_Canceled or OrderStatus_PartiallyFilled, {order.order_id}")
        # self.handle_order_retry(order)
    elif order.status in [OrderStatus_Rejected]:  # 已拒绝
        print(f"order OrderStatus_Rejected, {order.ord_rej_reason}")


def on_backtest_finished(context, indicator):
    print(f"Done! From start, time elapsed: {time.time() - context.start_time} seconds")
    print(f"{context.symbol} backtest finished: ", indicator)


index_list = {
    # List
    "SZSE.159509": "纳指科技ETF",
    "SHSE.518880": "黄金ETF",
    "SHSE.512480": "半导体ETF",
    "SZSE.159531": "中证2000ETF",
    "SHSE.513100": "纳指ETF",
    "SHSE.513520": "日经ETF",
    "SZSE.159857": "光伏ETF",
    "SHSE.512100": "中证1000ETF",
    "SHSE.510180": "上证180ETF",
    "SHSE.588000": "科创50ETF",
    "SHSE.513330": "恒生互联网ETF",
    "SZSE.162719": "石油LOF",
    "SHSE.513500": "标普500ETF",
    "SZSE.159915": "创业板ETF",

}

if __name__ == "__main__":
    run(
        # strategy_id='19236129-09e5-11f0-99ab-00155dd6c843',  # gfgm
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        # token='6860051c58995ae01c30a27d5b72000bababa8e6',  # gfgm
        strategy_id='630ce8b7-0c6d-11f0-a2bc-00155dd6c843',  # ydgm 回测
        token='c8bd4de742240da9483aecd05a2f5e52900786eb',  # ydgm
        backtest_start_time="2023-09-19 09:30:00",
        backtest_end_time='2025-03-27 15:00:00',
        # backtest_end_time='2023-10-20 15:00:00',
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0000,  # 0.0005
        backtest_commission_unit=1,
        backtest_slippage_ratio=0.0001,
        backtest_marginfloat_ratio1=0.2,
        backtest_marginfloat_ratio2=0.4,
        backtest_match_mode=0)

