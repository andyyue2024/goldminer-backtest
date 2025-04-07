from gm.api import *
import os
import time
from datetime import datetime, timedelta
import numpy as np
# 可运行，验证过。按照时间线分割
'''
马丁格尔策略 （可运行）
策略逻辑：
1. 初始开仓1手
2. 当持仓出现亏损时，按倍数加仓（每次加仓手数=前次手数*倍数）
3. 当盈利达到目标或达到最大加仓次数时平仓
4. 交易标的：
5. 使用5分钟K线

'''


def init(context):
    context.start_time = time.time()
    # 策略参数
    context.symbol = 'SZSE.159869'  # 交易标的
    context.initial_volume = 100  # 初始手数
    context.multiplier = 2  # 加仓倍数
    context.max_times = 20  # 最大加仓次数
    context.take_profit = 0.03  # 止盈比例（2%）
    context.trailing_stop = 0.03  # 回撤止损（10%）
    context.frequency = "1740s"

    # 订阅数据
    subscribe(context.symbol, context.frequency, count=21)  # 订阅20根K线（用于计算ATR）
    # subscribe(context.symbol, '60s', count=21)  # 订阅20根K线（用于计算ATR）

    # 初始化变量
    context.entry_price = None  # 开仓均价
    context.position_volume = 0  # 持仓数量
    context.add_times = 0  # 当前加仓次数
    context.direction = 0  # 持仓方向（0无仓，1多，-1空）
    context.highest_profit = -99999  # 持仓期间最高盈利
    context.profit = -99999  # 持仓期间盈利


def on_bar(context, bars):
    # print("on_bar:", bars[0].symbol, context.now)
    bar = bars[0]
    symbol = context.symbol

    # 获取当前仓位
    position = context.account().position(symbol=symbol, side=PositionSide_Long)
    if not position:
        position = context.account().position(symbol=symbol, side=PositionSide_Short)

    # 如果有持仓但context未记录，同步仓位信息
    if position and context.direction == 0:
        context.entry_price = position.vwap
        context.position_volume = position.volume
        context.direction = 1 if position.side == PositionSide_Long else -1

    # 获取历史数据计算波动率
    history_data = history_n(symbol=symbol, frequency=context.frequency, count=21, fields='close', end_time=bar.bob, df=True)
    atr = np.mean(np.abs(history_data['close'].diff().dropna()[-14:]))  # 计算14周期ATR

    # 无持仓时逻辑
    if context.direction == 0:
        # 计算布林带通道
        ma20 = history_data['close'].rolling(20).mean().iloc[-1]
        std = history_data['close'].rolling(20).std().iloc[-1]
        upper = ma20 + 2 * std
        lower = ma20 - 2 * std

        # 开仓条件：价格突破通道
        if 1:  # bar.close > upper:
            order_volume(symbol=symbol, volume=context.initial_volume, side=OrderSide_Buy,
                         order_type=OrderType_Market, position_effect=PositionEffect_Open)
            context.entry_price = bar.close
            context.position_volume = context.initial_volume
            context.direction = 1
            context.add_times = 1
            print("看多，开仓:", symbol, ", initial_volume:", context.initial_volume, ", ", context.now)

        elif bar.close < lower:
            order_volume(symbol=symbol, volume=context.initial_volume, side=OrderSide_Sell,
                         order_type=OrderType_Market, position_effect=PositionEffect_Open)
            context.entry_price = bar.close
            context.position_volume = context.initial_volume
            context.direction = -1
            context.add_times = 1
            print("看空，开仓:", symbol, ", initial_volume:", context.initial_volume, ", ", context.now)

    # 有持仓时逻辑
    else:
        current_price = bar.close
        # 计算浮动盈亏
        if context.direction == 1:
            profit = (current_price - context.entry_price) / context.entry_price
        else:
            profit = (context.entry_price - current_price) / context.entry_price

        context.profit = profit
        # 更新最高盈利
        if profit > context.highest_profit:
            context.highest_profit = profit

        # 止盈止损逻辑
        if profit >= context.take_profit:
            print('止盈')
            close_position(context)
            return
        if (context.highest_profit - profit) >= context.trailing_stop:
            print('止损')
            close_position(context)
            return

        # 加仓逻辑（亏损且未达最大次数）
        if profit < 0 and context.add_times < context.max_times:
            # 计算加仓手数（根据ATR调整）
            add_volume = context.initial_volume * (context.multiplier ** context.add_times)
            add_volume = int(max(1, add_volume * (atr / context.entry_price)))  # 波动率调整
            # add_volume = int(max(1, add_volume * (1 / context.entry_price)))  # 波动率调整

            # 计算新平均价格
            total_volume = context.position_volume + add_volume
            new_avg_price = (context.entry_price * context.position_volume + current_price * add_volume) / total_volume

            # 执行加仓
            if context.direction == 1:
                order_volume(symbol=symbol, volume=add_volume, side=OrderSide_Buy,
                             order_type=OrderType_Market, position_effect=PositionEffect_Open)
                print("看多，加仓:", symbol, ", add_volume:", add_volume, ", ", context.now)
            else:
                order_volume(symbol=symbol, volume=add_volume, side=OrderSide_Sell,
                             order_type=OrderType_Market, position_effect=PositionEffect_Open)
                print("看空，加仓:", symbol, ", add_volume:", add_volume, ", ", context.now)

            # 更新持仓信息
            context.entry_price = new_avg_price
            context.position_volume = total_volume
            context.add_times += 1

        # 达到最大加仓次数后强制平仓
        elif context.add_times >= context.max_times:
            print('达到最大加仓次数后强制平仓')
            close_position(context)


def close_position(context):
    symbol = context.symbol
    if context.direction == 1:
        order_volume(symbol=symbol, volume=context.position_volume, side=OrderSide_Sell,
                     order_type=OrderType_Market, position_effect=PositionEffect_Close)
        print("看多，平仓:", symbol, ", profit:",  context.profit, ", ",context.now)
    elif context.direction == -1:
        order_volume(symbol=symbol, volume=context.position_volume, side=OrderSide_Buy,
                     order_type=OrderType_Market, position_effect=PositionEffect_Close)
        print("看空，平仓:", symbol, ", profit:",  context.profit, ", ",context.now)

    # 重置状态
    context.entry_price = None
    context.position_volume = 0
    context.direction = 0
    context.add_times = 0
    context.highest_profit = -99999
    context.profit = -99999


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

