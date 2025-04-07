from gm.api import *
import os
import time
from datetime import datetime, timedelta
import numpy as np
import math
# 可运行，简单验证过。初步调试过。
'''
网格策略（可运行）
策略逻辑：

'''


def init(context):
    context.start_time = time.time()
    # 策略参数设置
    context.symbol = 'SZSE.159869'  # 交易标的
    context.frequency = '300s'  # 数据频率
    context.grid_levels = 5  # 网格层级数（单边）
    context.order_ratio = 0.2  # 每层仓位比例
    context.stop_loss = 0.05  # 止损比例5%
    context.take_profit = 0.03  # 止盈比例3%
    context.enable_short = True  # 允许做空

    # 初始化网格参数
    context.base_price = None  # 基准价格
    context.grid_prices = []  # 网格价格列表
    context.current_level = 0  # 当前网格层级
    context.position_side = 0  # 持仓方向：0-无，1-多，-1-空
    context.returns = -99999  # 持仓期间盈利率

    # 订阅数据和定时任务
    subscribe(context.symbol, context.frequency)
    schedule(schedule_func=check_grid, date_rule='1d', time_rule='09:31:00')


def check_grid(context):
    # 获取最新价格
    # latest_price = context.now.market('last', context.symbol)
    # latest_price = history_n(symbol=context.symbol, frequency='1d', end_time=context.now,
    #                          count=1, fields='close')[0]['open']
    latest_price = current(context.symbol)[0]['open']
    # 初始化基准价格
    if not context.base_price:
        context.base_price = latest_price
        context.grid_prices = generate_grid_prices(context.base_price, context.grid_levels)
        return

    # 检查持仓状态
    position = context.account().position(symbol=context.symbol, side=PositionSide_Long)
    current_qty = position.volume if position else 0

    # 止盈止损检查
    if current_qty != 0:
        check_risk_control(context, latest_price, position)

    # 生成交易信号
    signal = get_grid_signal(context, latest_price)

    # 执行交易
    if signal != 0:
        execute_trade(context, signal, latest_price)


def generate_grid_prices(base_price, levels):
    # 生成等分网格价格（确保升序排列）
    interval = base_price * 0.01
    prices = [base_price * (1 + i*0.01) for i in range(-levels, levels+1)]
    return sorted(prices)


def get_grid_signal(context, price):
    # 获取当前价格所在网格层级
    current_level = get_current_level(context, price)

    # 计算信号（考虑持仓方向）
    if context.position_side == 1:  # 多头持仓
        if current_level > context.current_level:
            return -1  # 平多
        elif current_level < context.current_level:
            return 1  # 加多
    elif context.position_side == -1:  # 空头持仓
        if current_level < context.current_level:
            return 1  # 平空
        elif current_level > context.current_level:
            return -1  # 加空
    else:  # 无持仓
        if current_level < len(context.grid_prices) // 2:
            return 1  # 开多
        else:
            return -1 if context.enable_short else 0  # 开空

    return 0


def execute_trade(context, signal, price):
    # 计算下单数量
    if price <= 0:
        return
    equity = context.account().cash['available']
    total_cash = equity * context.order_ratio
    qty = int(total_cash / price / 100) * 100  # 整手交易

    # 执行订单
    if signal == 1:
        order_volume(symbol=context.symbol, volume=qty, side=OrderSide_Buy,
                     order_type=OrderType_Market, position_effect=PositionEffect_Open)
        context.position_side = 1
        print("Long/Short，Buy:", context.symbol, ", volume:",  qty, ", ", context.now)
    elif signal == -1:
        if context.position_side == 1:
            order_volume(symbol=context.symbol, volume=qty, side=OrderSide_Sell,
                         order_type=OrderType_Market, position_effect=PositionEffect_Close)
            print("Long，Sell:", context.symbol, ", volume:", qty, ", ", context.now)
        else:
            order_volume(symbol=context.symbol, volume=qty, side=OrderSide_Sell,
                         order_type=OrderType_Market, position_effect=PositionEffect_Open)
            print("Short，Sell:", context.symbol, ", volume:", qty, ", ", context.now)
        context.position_side = -1 if context.enable_short else 0

    # 更新网格状态
    context.current_level = get_current_level(context, price)


def get_current_level(context, price):
    # 假设grid_prices已经是升序排列（在generate_grid_prices中保证）
    grid_prices = context.grid_prices

    # 二分查找优化版本
    left, right = 0, len(grid_prices) - 1
    best_level = 0
    while left <= right:
        mid = (left + right) // 2
        if price >= grid_prices[mid]:
            best_level = mid
            left = mid + 1
        else:
            right = mid - 1
    return best_level


def check_risk_control(context, price, position):
    # 计算盈亏比例
    cost_price = position.vwap
    returns = (price - cost_price) / cost_price if context.position_side == 1 else (cost_price - price) / cost_price
    context.returns = returns
    # 止盈止损检查
    if returns >= context.take_profit:
        print('take profit')
        close_position(context, position)
        reset_strategy(context)

    if returns <= -context.stop_loss:
        print('stop loss')
        close_position(context, position)
        reset_strategy(context)


def close_position(context, position):
    if position.side == PositionSide_Long:
        order_volume(symbol=context.symbol, volume=position.volume,
                     side=OrderSide_Sell, order_type=OrderType_Market,
                     position_effect=PositionEffect_Close)
        print("Long，Empty:", context.symbol, ", returns:",  context.returns, ", ", context.now)
    else:
        order_volume(symbol=context.symbol, volume=position.volume,
                     side=OrderSide_Buy, order_type=OrderType_Market,
                     position_effect=PositionEffect_Close)
        print("Short，Empty:", context.symbol, ", returns:",  context.returns, ", ", context.now)


def reset_strategy(context):
    context.base_price = None
    context.grid_prices = []
    context.current_level = 0
    context.position_side = 0
    context.returns = -99999


def on_order_status(context, order):
    # 订单状态更新处理
    pass


def on_execution_report(context, execrpt):
    # 成交回报处理
    pass


# 其他必要接口保持空实现
def on_tick(context, tick): pass


def on_bar(context, bars):
    # 获取当前K线数据
    current_bar = bars[0]
    symbol = current_bar['symbol']
    latest_price = current_bar['close']

    # 只在交易标的匹配时处理
    if symbol != context.symbol:
        return

    # 初始化基准价格（防止定时任务未执行）
    if not context.base_price:
        context.base_price = latest_price
        context.grid_prices = generate_grid_prices(context.base_price, context.grid_levels)
        return

    # 获取持仓信息（多空分开处理）
    long_position = context.account().position(symbol=context.symbol, side=PositionSide_Long)
    short_position = context.account().position(symbol=context.symbol, side=PositionSide_Short)

    # 风险控制检查（多空分别处理）
    for position in [long_position, short_position]:
        if position and position.volume > 0:
            check_risk_control(context, latest_price, position)

    # 生成交易信号
    signal = get_grid_signal(context, latest_price)


    # 执行交易
    if signal != 0:
        execute_trade(context, signal, latest_price)

    # 记录策略状态（可选）
    # context.write_log(f"当前价格{latest_price}，网格层级{context.current_level}，持仓方向{context.position_side}")

def on_account_status(context, account_status): pass


def on_parameter(context, parameter): pass


def on_error(context, code, info): pass


def on_trade_data_connected(context): pass


def on_market_data_connected(context): pass


def on_market_data_disconnected(context): pass


def on_trade_data_disconnected(context): pass


def on_shutdown(context): pass


def on_backtest_finished(context, indicator):
    print(f"Done! From start, time elapsed: {time.time() - context.start_time} seconds")
    print(f"{context.symbol} backtest finished: ", indicator)


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
    backtest_start_time = str(datetime.now() - timedelta(days=150))[:19]
    backtest_end_time = str(datetime.now())[:19]
    run(strategy_id='07f52dc2-489e-11ef-8e54-00ffb355a5f1',
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        token='42c5e7778813e4a38aae2e65d70eb372ac8f2435',
        # backtest_start_time=backtest_start_time,
        # backtest_end_time=backtest_end_time,
        backtest_start_time="2024-06-24 09:30:00",
        backtest_end_time='2024-12-01 15:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)

