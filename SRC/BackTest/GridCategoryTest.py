# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from gm.api import *
import os
import time
import datetime
import numpy as np
import pandas as pd
# 可运行，未验证过。未全面调试过。

'''

网格交易法是一种把行情的所有日间上下的波动全部囊括，不会放过任何一次行情上下波动的策略。
本策略标的为：SZSE.159869
价格中枢设定为：每日前一交易日的收盘价，每个网格间距3%；每变动一次，交易一手
'''


def init(context):
    context.start_time = time.time()
    # 策略标的为SZSE.159869
    # context.symbol = 'SHFE.RB'
    context.symbol = 'SZSE.159869'
    context.main_contract = 'SZSE.159869'
    # 设置每变动一格，增减的数量
    context.volume = 10000
    # 储存前一个网格所处区间，用来和最新网格所处区间作比较
    context.last_grid = 0
    # 记录上一次交易时网格范围的变化情况（例如从4区到5区，记为4,5）
    context.grid_change_last = [0, 0]
    # 止损条件:最大持仓
    context.max_volume = 15
    # 数据一次性获取
    if context.mode == MODE_BACKTEST:
        contract_list = fut_get_continuous_contracts(csymbol=context.symbol,
                                                     start_date=context.backtest_start_time[:10],
                                                     end_date=context.backtest_end_time[:10])
        if len(contract_list) > 0:
            context.contract_list = {dic['trade_date']: dic['symbol'] for dic in contract_list}
    # 定时任务，日频，盘前运行
    # schedule(schedule_func=algo, date_rule='1d', time_rule='21:00:00')
    schedule(schedule_func=algo, date_rule='1d', time_rule='09:00:00')


def algo(context):
    now_str = context.now.strftime('%Y-%m-%d')
    # 主力合约
    if context.now.hour > 15:
        date = get_next_n_trading_dates(exchange='SHFE', date=now_str, n=1)[0]
    else:
        date = context.now.strftime('%Y-%m-%d')
    # if context.mode == MODE_BACKTEST and date in context.contract_list:
    #     context.main_contract = context.contract_list[date]
    # else:
    #     context.main_contract = fut_get_continuous_contracts(csymbol=context.symbol, start_date=date, end_date=date)[0][
    #         'symbol']
    # 订阅行情
    subscribe(context.main_contract, '60s', count=1, unsubscribe_previous=True)
    # 有持仓时，检查持仓的合约是否为主力合约,非主力合约则卖出
    Account_positions = get_position()
    if Account_positions:
        for posi in Account_positions:
            if context.main_contract != posi['symbol']:
                print('{}：持仓合约由{}替换为主力合约{}'.format(context.now, posi['symbol'], context.main_contract))
                new_price = current(symbols=posi['symbol'])[0]['price']
                order_target_volume(symbol=posi['symbol'],
                                    volume=0,
                                    position_side=posi['side'],
                                    order_type=OrderType_Limit,
                                    price=new_price)

    # 获取前一交易日的收盘价作为价格中枢
    if context.now.hour >= 20:
        # 当天夜盘和次日日盘属于同一天数据，为此当天夜盘的上一交易日收盘价应调用当天的收盘价
        context.center = \
        history_n(symbol=context.main_contract, frequency='1d', end_time=context.now, count=1, fields='close')[0][
            'close']
    else:
        last_date = get_previous_n_trading_dates(exchange='SHSE', date=now_str, n=1)[0]
        context.center = \
        history_n(symbol=context.main_contract, frequency='1d', end_time=last_date, count=1, fields='close')[0]['close']

    # 设置网格
    context.band = np.array([0.92, 0.94, 0.96, 0.98, 1, 1.02, 1.04, 1.06, 1.08]) * context.center


def on_bar(context, bars):
    bar = bars[0]
    # 获取仓位
    positions = get_position()
    position_long = list(
        filter(lambda x: x['symbol'] == context.main_contract and x['side'] == PositionSide_Long, positions))  # 多头仓位
    position_short = list(
        filter(lambda x: x['symbol'] == context.main_contract and x['side'] == PositionSide_Short, positions))  # 空头仓位

    # 当前价格所处的网格区域
    grid = pd.cut([bar.close], context.band, labels=[1, 2, 3, 4, 5, 6, 7, 8])[
        0]  # 1代表(0.88%,0.91%]区间，2代表(0.91%,0.94%]区间...

    # 如果价格超出网格设置范围，则提示调节网格宽度和数量
    if np.isnan(grid):
        # print('价格波动超过网格范围，可适当调节网格宽度和数量')
        return

        # 如果新的价格所处网格区间和前一个价格所处的网格区间不同，说明触碰到了网格线，需要进行交易
    # 如果新网格大于前一天的网格，做空或平多
    if context.last_grid < grid:
        # 记录新旧格子范围（按照大小排序）
        grid_change_new = [context.last_grid, grid]

        # 当last_grid = 0 时是初始阶段，不构成信号
        if context.last_grid == 0:
            context.last_grid = grid
            return

        # 如果前一次开仓是4-5，这一次是5-4，算是没有突破，不成交
        if grid_change_new != context.grid_change_last:
            # 如果有多仓，平多
            if position_long:
                order_volume(symbol=context.main_contract, volume=context.volume, side=OrderSide_Sell,
                             order_type=OrderType_Market,
                             position_effect=PositionEffect_Close)
                print('{}:从{}区调整至{}区，以市价单平多仓{}手'.format(context.now, context.last_grid, grid,
                                                                      context.volume))

            # # 否则，做空
            # if not position_long:
            #     order_volume(symbol=context.main_contract, volume=context.volume, side=OrderSide_Sell,
            #                  order_type=OrderType_Market,
            #                  position_effect=PositionEffect_Open)
            #     print(
            #         '{}:从{}区调整至{}区，以市价单开空{}手'.format(context.now, context.last_grid, grid, context.volume))

            # 更新前一次的数据
            context.last_grid = grid
            context.grid_change_last = grid_change_new
        else:
            print('{}:从{}区调整至{}区，无交易'.format(context.now, context.last_grid, grid))
            context.last_grid = grid

    # 如果新网格小于前一天的网格，做多或平空
    if context.last_grid > grid:
        # 记录新旧格子范围（按照大小排序）
        grid_change_new = [grid, context.last_grid]

        # 当last_grid = 0 时是初始阶段，不构成信号
        if context.last_grid == 0:
            context.last_grid = grid
            return

        # 如果前一次开仓是4-5，这一次是5-4，算是没有突破，不成交
        if grid_change_new != context.grid_change_last:
            # 如果有空仓，平空
            if position_short:
                order_volume(symbol=context.main_contract, volume=context.volume, side=OrderSide_Buy,
                             order_type=OrderType_Market, position_effect=PositionEffect_Close)
                print('{}:从{}区调整至{}区，以市价单平空仓{}手'.format(context.now, context.last_grid, grid,
                                                                      context.volume))

            # 否则，做多
            if not position_short:
                order_volume(symbol=context.main_contract, volume=context.volume, side=OrderSide_Buy,
                             order_type=OrderType_Market, position_effect=PositionEffect_Open)
                print(
                    '{}:从{}区调整至{}区，以市价单开多{}手'.format(context.now, context.last_grid, grid, context.volume))

            # 更新前一次的数据
            context.last_grid = grid
            context.grid_change_last = grid_change_new
        else:
            print('{}:从{}区调整至{}区，无交易'.format(context.now, context.last_grid, grid))
            context.last_grid = grid

    # 设计一个止损条件：当持仓量达到20手，全部平仓
    if (position_short and position_short[0]['volume'] == context.max_volume) or (
            position_long and position_long[0]['volume'] == context.max_volume):
        order_close_all()
        print('{}:触发止损，全部平仓'.format(context.now))


def on_backtest_finished(context, indicator):
    print(f"Done! From start, time elapsed: {time.time() - context.start_time} seconds")
    print(f"{context.symbol} backtest finished: ", indicator)


if __name__ == '__main__':
    '''
    strategy_id策略ID,由系统生成
    filename文件名,请与本文件名保持一致
    mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
    token绑定计算机的ID,可在系统设置-密钥管理中生成
    backtest_start_time回测开始时间
    backtest_end_time回测结束时间
    backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
    backtest_initial_cash回测初始资金
    backtest_commission_ratio回测佣金比例
    backtest_slippage_ratio回测滑点比例
    backtest_match_mode市价撮合模式，以下一tick/bar开盘价撮合:0，以当前tick/bar收盘价撮合：1
    '''
    backtest_start_time = str(datetime.datetime.now() - datetime.timedelta(days=1400))[:19]
    backtest_end_time = str(datetime.datetime.now())[:19]
    run(strategy_id='a0612bb2-eb4f-11ef-83e4-00ffb355a5f1',
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        token='42c5e7778813e4a38aae2e65d70eb372ac8f2435',
        backtest_start_time=backtest_start_time,
        backtest_end_time=backtest_end_time,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
