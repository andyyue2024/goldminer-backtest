# coding=utf-8
from __future__ import print_function, absolute_import
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from gm.api import *



"""
ai labx test3

- support ai-labx sort
- support splitting orders
- speed by local cache

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
        trend_score = self.trend_score(symbol, "close", 25)
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
    def __init__(self, context, white_list: list = None, max_count: int = 1, w_aa=0.1, w_bb=0.2, w_cc=1, w_dd=0.18):
        self.now = None
        self.context = context
        self.white_list = list(white_list)
        self.max_count = max_count
        self.ailabx = AILabxTool(w_aa=w_aa, w_bb=w_bb, w_cc=w_cc)
        if context.mode == MODE_BACKTEST:
            self.ailabx.get_all_data(self.white_list, context.backtest_start_time, context.backtest_end_time)
        self.w_dd = w_dd

        self.enable_split_order = True
        self.active_orders = {}  # 活跃订单跟踪
        self.pending_orders = {}  # 待处理订单跟踪
        self.max_retry = 3  # 最大撤单重试次数
        self.order_size_limit = 20000  # 单笔订单数量限制（根据风控要求调整）

    def filter(self, in_list: list = None):
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

    @staticmethod
    def filter_top(in_list: list, top_count=1):
        if not in_list:
            return []
        return in_list[0:top_count]

    def should_sell(self, target: str):
        return self.ailabx.roc(target, "close", 21) > self.w_dd
        # return False

    def split_order(self, symbol, target_volume):
        # 拆单逻辑（时间加权）
        orders = []
        if not self.enable_split_order:
            orders.append({'symbol': symbol, 'volume': target_volume})
            return orders
        remaining = target_volume
        while remaining > 0:
            size = min(remaining, self.order_size_limit)
            orders.append({'symbol': symbol, 'volume': size})
            remaining -= size
        return orders

    def execute_sell(self, symbol, volume):
        # 获取当前持仓
        if volume <= 0:
            return True

        # 生成拆单指令
        sub_orders = self.split_order(symbol, volume)
        for order in sub_orders:
            order_id = order_volume(symbol=order['symbol'],
                                    volume=order['volume'],
                                    side=OrderSide_Sell,
                                    order_type=OrderType_Limit,
                                    position_effect=PositionEffect_Close,
                                    price=self.latest_price(symbol))[0]["order_id"]
            self.active_orders[order_id] = {
                'symbol': symbol,
                'create_time': time.time(),
                'retry_count': 0,
                'original_volume': volume,  # 总需要平仓量
                'filled_total': 0,  # 新增累计成交量字段
                'side': OrderSide_Sell,
                'position_effect': PositionEffect_Close
            }
            print(f"execute_sell: {order['volume']}, order_id:{order_id}")
        return False

    def execute_buy(self, symbol):
        # 计算可用资金
        account = self.context.account()
        # positions = account.positions()
        available_cash = account.cash['available']

        # 获取最新价格计算可买数量
        latest = self.latest_price(symbol)
        max_volume = int(available_cash / (latest * 100)) * 100  # 按整手计算

        # 生成拆单指令
        sub_orders = self.split_order(symbol, max_volume)
        for order in sub_orders:
            order_id = order_volume(symbol=order['symbol'],
                                    volume=order['volume'],
                                    side=OrderSide_Buy,
                                    order_type=OrderType_Limit,
                                    position_effect=PositionEffect_Open,
                                    price=latest)[0]["order_id"]
            self.active_orders[order_id] = {
                'symbol': symbol,
                'create_time': time.time(),
                'retry_count': 0,
                'original_volume': max_volume,  # 总需要建仓量
                'filled_total': 0,  # 新增累计成交量字段
                'side': OrderSide_Buy,
                'position_effect': PositionEffect_Open

            }
            print(f"execute_buy: {order['volume']} * {latest} = {order['volume'] * latest} vs {available_cash}, order_id:{order_id}")

    def handle_order_retry(self, order):
        order_info = self.active_orders.get(order.order_id)
        if not order_info:
            return

        # 累计已成交量到总成交
        order_info['filled_total'] += order.filled_volume  # 新增字段记录累计成交量
        remaining = order_info['original_volume'] - order_info['filled_total']

        # 检查完成情况
        if remaining <= 0:
            print(f"订单全部完成 {order_info['symbol']}")
            return

        # 检查重试次数
        if order_info['retry_count'] >= self.max_retry:
            print(f"达到最大重试次数 {order_info['symbol']} 剩余{remaining}未成交")
            return

        # 重新创建订单（带累计成交量跟踪）
        new_order_id = order_volume(
            symbol=order_info['symbol'],
            volume=remaining,
            side=order_info['side'],
            order_type=OrderType_Limit,
            position_effect=order_info['position_effect'],
            price=self.latest_price(order_info['symbol'])
        )[0]["order_id"]
        print(f"Done to order again: {order.order_id} -> {new_order_id}")

        # 更新订单记录（保持原始目标量）
        self.active_orders[new_order_id] = {
            'symbol': order_info['symbol'],
            'create_time': time.time(),
            'retry_count': order_info['retry_count'] + 1,
            'original_volume': order_info['original_volume'],  # 保持原始目标量
            'filled_total': order_info['filled_total'],  # 继承累计成交量
            'side': order_info['side'],
            'position_effect': order_info['position_effect']
        }

        # 删除原订单记录
        del self.active_orders[order.order_id]

    @staticmethod
    def latest_price(symbol):
        current_data = current(symbols=symbol)
        return current_data[0]["price"]

    def rebalance(self, target_symbol):
        # 获取当前持仓
        account = self.context.account()
        positions = account.positions()
        current_pos = {p.symbol: p.volume for p in positions}

        # 选择最优标的
        # target_symbol = self.select_optimal_symbol()
        # self.counter += 1
        # target_symbol = self.target_list[self.counter % 2]

        # 卖出非目标持仓
        for symbol, volume in list(current_pos.items()):
            if (symbol != target_symbol or
                    (symbol == target_symbol and self.should_sell(target_symbol))):  # 命中强制卖出条件
                print(f"Start to sell: {symbol}")
                if not self.execute_sell(symbol, volume):
                    # print(f"Done to order to sell: {symbol}")
                    pass

        # 等待所有卖单完成
        while len(self.active_orders) > 0:
            pending_orders_tmp = {k: v for k, v in self.pending_orders.items() if v != 0}
            for order_id in pending_orders_tmp.keys():
                if order_id in self.active_orders.keys():
                    del self.active_orders[order_id]
                    # self.pending_orders[order_id] = 0
                    del self.pending_orders[order_id]
            time.sleep(0.05)

        if len(current_pos) > 0 and (target_symbol not in current_pos):
            # print(f"Done, sold")
            pass

        # 命中强制卖出条件
        if self.should_sell(target_symbol):
            return

        # 买入目标标的
        if target_symbol not in current_pos:
            print(f"Start to buy: {target_symbol}")
            self.execute_buy(target_symbol)
            # print(f"Done to order to buy: {target_symbol}")

        # 等待所有买单完成
        while len(self.active_orders) > 0:
            pending_orders_tmp = {k: v for k, v in self.pending_orders.items() if v != 0}
            for order_id in pending_orders_tmp.keys():
                if order_id in self.active_orders.keys():
                    del self.active_orders[order_id]
                    # self.pending_orders[order_id] = 0
                    del self.pending_orders[order_id]
            time.sleep(0.05)
        if target_symbol not in current_pos:
            # print(f"Done, bought")
            pass

    def execute(self, now):
        self.now = now
        self.ailabx.now = self.now
        ret_list = self.filter()
        ret_list = self.sort(ret_list)
        ret_list = self.filter_top(ret_list)
        if len(ret_list) == 1:
            self.rebalance(target_symbol=ret_list[0])
        return ret_list

    def on_order_status(self, order):
        # 订单状态更新处理
        if order.status == OrderStatus_Filled:  # 完全成交
            if order.order_id in self.active_orders.keys():
                del self.active_orders[order.order_id]
            else:
                # print(f"pending_orders {order.order_id} = 1")
                self.pending_orders[order.order_id] = 1
                pass
        elif order.status in [OrderStatus_Canceled, OrderStatus_PartiallyFilled]:  # 已撤单/部分成交撤单
            print(f"handle_order_retry {order.order_id}")
            self.handle_order_retry(order)
        elif order.status in [OrderStatus_Rejected]:  # 已拒绝
            if order.ord_rej_reason in [OrderRejectReason_NoEnoughCash]:
                # print(f"pending_orders {order.order_id} = 1")
                self.pending_orders[order.order_id] = 1
            else:
                print(f"handle_order_retry {order.order_id}")
                self.handle_order_retry(order)

    def on_tick(self, tick):
        # 订单价格跟踪（示例：动态更新限价单价格）
        unfinished = get_unfinished_orders()
        for order in unfinished:
            # 下单超过60秒，没有全部成交撤单
            if (abs(self.context.now - order['created_at'])).seconds > 60:
                # 撤单
                print(f"撤单超时订单：{order['order_id']}")
                order_cancel(wait_cancel_orders=[{'cl_ord_id': order['cl_ord_id'], 'account_id': order['account_id']}])


def init(context):
    context.start_time = time.time()
    print("Start...")
    context.num = 1
    context.symbol = "AAA"
    context.target_list = list(index_list.keys())
    context.ai_labx_strategy = AILabxStrategy(context=context, white_list=context.target_list)
    # set_option(backtest_thread_num=5)

    # subscribe(context.target_list, '30s')
    schedule(schedule_func=algo, date_rule='1d', time_rule='09:31:00')


def algo(context):
    print(context.now, " algo...")
    context.ai_labx_strategy.execute(context.now)


def on_order_status(context, order):
    context.ai_labx_strategy.on_order_status(order)


# def on_tick(context, tick):
#     context.ai_labx_strategy.on_tick(tick)


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


if __name__ == '__main__':
    '''
        strategy_id策略ID, 由系统生成
        filename文件名, 请与本文件名保持一致
        mode运行模式, 实时模式:MODE_LIVE回测模式:MODE_BACKTEST
        token绑定计算机的ID, 可在系统设置-密钥管理中生成
        backtest_start_time回测开始时间
        backtest_end_time回测结束时间
        backtest_adjust股票复权方式, 不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
        backtest_initial_cash回测初始资金
        backtest_commission_ratio回测佣金比例
        backtest_slippage_ratio回测滑点比例
        backtest_match_mode市价撮合模式，以下一tick/bar开盘价撮合:0，以当前tick/bar收盘价撮合：1
        '''
    run(strategy_id='99daf740-13a1-11f0-a294-206a8a674f71',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='c8bd4de742240da9483aecd05a2f5e52900786eb',
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

