import time
import os
import numpy as np
import pandas as pd
from gm.api import *
from copy import copy


class SplitOrderStrategy:
    def __init__(self, context, target_list: list):
        self.context = context
        self.target_list = target_list  # 标的池
        self.enable_split_order = True
        self.active_orders = {}  # 活跃订单跟踪
        self.pending_orders = {}  # 待处理订单跟踪
        self.max_retry = 3  # 最大撤单重试次数
        self.order_size_limit = 1000  # 单笔订单数量限制（根据风控要求调整）
        self.history_days = 20  # 指标计算周期
        self.counter = 0

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

    def select_optimal_symbol(self):
        # 计算动量指标和能量均线
        momentum_scores = {}
        energy_scores = {}

        for symbol in self.target_list:
            # 获取历史数据
            bars = history_n(symbol, frequency='1d', count=self.history_days + 1, end_time=self.context.now,
                             fields='close,volume', adjust=ADJUST_PREV, df=True)

            # 动量计算（20日收益率）
            momentum = (bars['close'].iloc[-1] / bars['close'].iloc[0]) - 1

            # 能量均线计算（成交量加权EMA）
            ema_close = bars['close'].ewm(span=10).mean().iloc[-1]
            ema_volume = bars['volume'].ewm(span=10).mean().iloc[-1]
            energy = ema_close * ema_volume

            momentum_scores[symbol] = momentum
            energy_scores[symbol] = energy

        # 标准化评分并综合
        df = pd.DataFrame({
            'momentum': pd.Series(momentum_scores),
            'energy': pd.Series(energy_scores)
        })
        df = df.apply(lambda x: (x - x.mean()) / x.std())  # Z-score标准化
        df['total'] = df['momentum'] + df['energy']

        return df['total'].idxmax()

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
            print(f"execute_buy: {order['volume']}, order_id:{order_id}")

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

    def on_rebalance(self):
        # 获取当前持仓
        account = self.context.account()
        positions = account.positions()
        current_pos = {p.symbol: p.volume for p in positions}

        # 选择最优标的
        # target_symbol = self.select_optimal_symbol()
        self.counter += 1
        target_symbol = self.target_list[self.counter % 2]

        # 卖出非目标持仓
        for symbol, volume in list(current_pos.items()):
            if symbol != target_symbol:
                print(f"Start to sell: {symbol}")
                if not self.execute_sell(symbol, volume):
                    print(f"Done to order to sell: {symbol}")

        # 等待所有卖单完成
        while len(self.active_orders) > 0:
            pending_orders_tmp = {k: v for k, v in self.pending_orders.items() if v != 0}
            for order_id in pending_orders_tmp.keys():
                if order_id in self.active_orders.keys():
                    del self.active_orders[order_id]
                    # self.pending_orders[order_id] = 0
                    del self.pending_orders[order_id]
            time.sleep(0.005)
        print(f"Done, sold")

        # 买入目标标的
        if target_symbol not in current_pos:
            print(f"Start to buy: {target_symbol}")
            self.execute_buy(target_symbol)
            print(f"Done to order to buy: {target_symbol}")

        # 等待所有买单完成
        while len(self.active_orders) > 0:
            pending_orders_tmp = {k: v for k, v in self.pending_orders.items() if v != 0}
            for order_id in pending_orders_tmp.keys():
                if order_id in self.active_orders.keys():
                    del self.active_orders[order_id]
                    # self.pending_orders[order_id] = 0
                    del self.pending_orders[order_id]
            time.sleep(0.005)

        print(f"Done, bought")

    def on_tick(self, tick):
        # 订单价格跟踪（示例：动态更新限价单价格）
        unfinished = get_unfinished_orders()
        for order in unfinished:
            # 下单超过60秒，没有全部成交撤单
            if (abs(self.context.now - order['created_at'])).seconds > 60:
                # 撤单
                print(f"撤单超时订单：{order['order_id']}")
                order_cancel(wait_cancel_orders=[{'cl_ord_id': order['cl_ord_id'], 'account_id': order['account_id']}])

    def latest_price(self, symbol):
        current_data = current(symbols=symbol)
        return current_data[0]["price"]


def init(context):
    context.start_time = time.time()
    print("Start...")
    context.num = 1
    context.symbol = "AAA"
    context.target_list = ['SHSE.513020', 'SZSE.300856']  # 标的池
    context.strategy = SplitOrderStrategy(context, context.target_list)
    set_option(backtest_thread_num=5)

    # 初始化订阅和定时任务
    subscribe(context.target_list, 'tick')
    schedule(schedule_func=algo, date_rule='1d', time_rule='14:55:00')
    # schedule(schedule_func=algo, date_rule='1d', time_rule='09:31:00')


def algo(context):
    print(context.now, " algo...")
    context.strategy.on_rebalance()


def on_order_status(context, order):
    context.strategy.on_order_status(order)


def on_bar(context, bars):
    current_bar = bars[0]
    if current_bar['symbol'] != context.symbol:
        return
    price = current_bar['close']


def on_tick(context, tick):
    context.strategy.on_tick(tick)


def on_backtest_finished(context, indicator):
    print(f"Done! From start, time elapsed: {time.time() - context.start_time} seconds")
    print(f"{context.symbol} backtest finished: ", indicator)


if __name__ == '__main__':
    run(
        # strategy_id='19236129-09e5-11f0-99ab-00155dd6c843',  # gfgm
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        # token='6860051c58995ae01c30a27d5b72000bababa8e6',  # gfgm
        strategy_id='630ce8b7-0c6d-11f0-a2bc-00155dd6c843',  # ydgm 回测
        token='c8bd4de742240da9483aecd05a2f5e52900786eb',  # ydgm
        backtest_start_time="2025-01-01 09:30:00",
        backtest_end_time='2025-02-20 15:00:00',
        # backtest_end_time='2023-10-20 15:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0001,  # 0.0005
        backtest_commission_unit=1,
        backtest_slippage_ratio=0.0001,
        backtest_marginfloat_ratio1=0.2,
        backtest_marginfloat_ratio2=0.4,
        backtest_match_mode=0)
