from gm.api import *
import os
import time
from datetime import datetime
# 可运行。已测试。
"""
双均线策略
"""


def init(context):
    context.start_time = time.time()
    # 策略配置
    context.symbol = 'SHSE.510300'  # 沪深300ETF
    context.short_period = 5  # 短期均线周期
    context.long_period = 20  # 长期均线周期
    context.close_prices = []  # 存储收盘价

    # 订阅沪深300ETF的日线数据
    subscribe(symbols=[context.symbol], frequency='1d', count=context.long_period + 1, wait_group=True)


def on_bar(context, bars):
    # 获取最新收盘价
    close_price = bars[-1].close
    context.close_prices.append(close_price)

    # 确保有足够数据计算长周期均线
    if len(context.close_prices) < context.long_period:
        return

    # 计算双均线
    ma_short = sum(context.close_prices[-context.short_period:]) / context.short_period
    ma_long = sum(context.close_prices[-context.long_period:]) / context.long_period

    # 获取当前持仓
    position = context.account().position(symbol=context.symbol, side=PositionSide_Long)

    # 交易信号逻辑
    if cross_over(ma_short, ma_long, context):
        if not position:  # 无持仓时买入
            cash = context.account().cash['available']
            volume = int(cash // (close_price * 100) * 100)  # ETF按100股整数倍交易
            if volume > 0:
                print('Buy, volume:', volume, ', price:', close_price, ', now:', context.now)
                order_volume(
                    symbol=context.symbol,
                    volume=volume,
                    side=OrderSide_Buy,
                    order_type=OrderType_Market,
                    position_effect=PositionEffect_Open
                )
    elif cross_under(ma_short, ma_long, context):
        if position:  # 有持仓时卖出
            print('Sell, volume:', position.volume, ', price:', close_price, ', now:', context.now)
            order_volume(
                symbol=context.symbol,
                volume=position.volume,
                side=OrderSide_Sell,
                order_type=OrderType_Market,
                position_effect=PositionEffect_Close
            )

    # 维护固定长度数据队列
    if len(context.close_prices) > context.long_period * 2:
        context.close_prices.pop(0)


def cross_over(short, long, context):
    """判断金叉：需要前一日短期均线<=长期均线且当日短期均线>长期均线"""
    if len(context.close_prices) < context.long_period + 1:
        return False
    prev_short = sum(context.close_prices[-context.short_period - 1:-1]) / context.short_period
    prev_long = sum(context.close_prices[-context.long_period - 1:-1]) / context.long_period
    return prev_short <= prev_long and short > long


def cross_under(short, long, context):
    """判断死叉：需要前一日短期均线>=长期均线且当日短期均线<长期均线"""
    if len(context.close_prices) < context.long_period + 1:
        return False
    prev_short = sum(context.close_prices[-context.short_period - 1:-1]) / context.short_period
    prev_long = sum(context.close_prices[-context.long_period - 1:-1]) / context.long_period
    return prev_short >= prev_long and short < long


def on_backtest_finished(context, indicator):
    print(f"Done! From start, time elapsed: {time.time() - context.start_time} seconds")
    print(f"{context.symbol} backtest finished: ", indicator)


if __name__ == '__main__':
    run(
        strategy_id='dd5b221e-f18e-11ef-8188-80304917db79',
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        token='6860051c58995ae01c30a27d5b72000bababa8e6',
        backtest_start_time='2016-10-11 08:00:00',
        backtest_end_time='2025-02-20 15:00:00',
        backtest_initial_cash=1000000,
        backtest_transaction_ratio=0.001,  # 交易费率
        backtest_commission_ratio=0.0003,  # 佣金率
        backtest_adjust=ADJUST_PREV  # 前复权
    )

