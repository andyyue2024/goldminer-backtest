from gm.api import *
import os
import time
from datetime import datetime, timedelta
import numpy as np

# try:
#     import matplotlib.pyplot as plt
# except:
#     print('请安装matplotlib库')
#     import sys
#     sys.exit(-1)

"""
双卖Delta中性策略
注：此策略只用于学习、交流、演示，不构成任何投资建议。
- 设置初始委托量，这里为10张，设置好需要对冲的绝对值阈值abs = 0.5，
  即组合的delta绝对值大于此值就调整一次目标仓位，实现目标仓位组合的delta趋向于0.
- 每一分钟onbar计算greeks并更新，同时判断是否需要调整目标仓位，
  如果需要调整则计算目标仓位（目标仓位的delta在Range（-0.5，0.5）），并调仓至目标仓位。
- 此策略回测标的为下季一档虚值合约，不进行移仓换月，请注意回测时间跨度不要超过一个季度，避免超出合约存续期。
"""


# 策略中必须有init方法
def init(context):
    # 量
    context.volume = 10
    # bars
    context.bars = {}
    # 希腊值，包含iv
    context.greeks = {}
    # 对冲前组合greeks
    context.greeks_before_hedging = {}
    # 对冲后组合greeks
    context.greeks_after_hedging = {}
    # 目前仓位
    context.target_pos = {}
    # 对冲阈值
    context.abs = 0.5

    # 对冲前组合greeks
    context.greeks_before_hedging_series = {}
    # 对冲后组合greeks
    context.greeks_after_hedging_series = {}

    # 标的物
    context.underlying = 'SZSE.159901'
    data = current(context.underlying)
    if data:
        s = data[0]['price']
    else:
        print('{}没有获取到context.underlying的数据'.format(context.now))
        return

    # 获取下季一档虚值合约
    context.c_symbol, context.p_symbol = (option_get_symbols_by_in_at_out(underlying_symbol=context.underlying,
                                                                          trade_date=context.now,
                                                                          execute_month=3,
                                                                          call_or_put=call_or_put,
                                                                          in_at_out=-1,
                                                                          s=s) for call_or_put in ['C', 'P'])
    if not (context.c_symbol or context.p_symbol):
        print('{}没有获取到context.c_symbol或context.p_symbol的数据, 请检查当天是否为交易日'.format(context.now))
        return

    context.c_info = get_instrumentinfos(symbols=context.c_symbol)
    if context.c_info:
        context.c_info = context.c_info[0]
    else:
        print('{}没有获取到context.c_info的数据'.format(context.now))
        return

    context.p_info = get_instrumentinfos(symbols=context.p_symbol)
    if context.p_info:
        context.p_info = context.p_info[0]
    else:
        print('{}没有获取到context.p_info的数据'.format(context.now))
        return

    # 订阅相关数据
    subscribe(symbols=[context.underlying, context.c_info['symbol'], context.p_info['symbol']], frequency='600s')


def on_bar(context, bars):
    # 更新bar
    for bar in bars:
        context.bars.update({bar['symbol']: bar})

    # 计算greeks并更新
    if len(context.bars) == 3:
        # 计算希腊值
        calculate_greeks(context)
        # 计算组合对冲希腊值
        sum_synthesis_greeks(context)
        # 计算对冲后的目标持仓量
        calculate_target_pos(context)
        # 委托
        for symbol, target_volume in context.target_pos.items():
            order_target_pos(context=context, symbol=symbol, target_volume=target_volume)
            print('{}市价调整到目标持仓量{}'.format(symbol, abs(target_volume)))


def order_target_pos(context, symbol, target_volume):
    '''
    市价调整到目标持仓
    '''
    # 目标持仓方向
    target_side = OrderSide_Buy if target_volume > 0 else OrderSide_Sell
    # 当前空头持仓
    current_pos = context.account().position(symbol, side=PositionSide_Short, covered_flag=0)
    if not current_pos is None:
        # 当前持仓方向
        current_pos_side = current_pos.side
        # 当前持仓数量
        current_pos_volume = current_pos.volume
        if current_pos_side == target_side:
            # 仓差
            pos_diff = current_pos_volume - abs(target_volume)
            if pos_diff > 0:
                # 减仓
                order_volume(symbol=symbol, volume=abs(pos_diff),
                             side=OrderSide_Buy if current_pos_side == PositionSide_Short else OrderSide_Sell,
                             order_type=OrderType_Market, position_effect=PositionEffect_Close)
            elif pos_diff < 0:
                # 加仓
                order_volume(symbol=symbol, volume=abs(pos_diff),
                             side=current_pos_side, order_type=OrderType_Market, position_effect=PositionEffect_Open)
        elif current_pos_side != target_side:
            # 当前方向平仓
            order_volume(symbol=symbol, volume=current_pos_volume,
                         side=target_side, order_type=OrderType_Market, position_effect=PositionEffect_Close)
            # 目标方向开仓
            order_volume(symbol=symbol, volume=abs(target_volume),
                         side=target_side, order_type=OrderType_Market, position_effect=PositionEffect_Open)
    else:
        # 无持仓，目标方向开仓
        order_volume(symbol=symbol, volume=abs(target_volume), side=target_side,
                     order_type=OrderType_Market, position_effect=PositionEffect_Open)


def calculate_target_pos(context):
    '''
    计算目标仓位，实现对冲
    '''
    context.target_pos = {context.c_info['symbol']: -context.volume,
                          context.p_info['symbol']: -context.volume}

    if abs(context.greeks_before_hedging['delta']) > context.abs:
        if (context.greeks_before_hedging['delta'] > 0) & (context.greeks[context.c_info['symbol']]['delta'] != 0):
            volume = int(
                abs(context.greeks_before_hedging['delta'] / context.greeks[context.c_info['symbol']]['delta']))
            context.target_pos[context.c_info['symbol']] = - context.volume - volume
        elif (context.greeks_before_hedging['delta'] < 0) & (context.greeks[context.p_info['symbol']]['delta'] != 0):
            volume = int(
                abs(context.greeks_before_hedging['delta'] / context.greeks[context.p_info['symbol']]['delta']))
            context.target_pos[context.p_info['symbol']] = - context.volume - volume


def sum_synthesis_greeks(context):
    '''
    计算组合对冲希腊值
    '''
    # 所有持仓
    Account_positions = {pos['symbol']
                         : pos for pos in context.account().positions()}

    for greek in ['delta', 'vega', 'theta', 'gamma']:
        # 对冲前组合greeks,双卖*-1
        context.greeks_before_hedging[greek] = (context.greeks[context.c_info['symbol']][greek] +
                                                context.greeks[context.p_info['symbol']][
                                                    greek]) * context.volume * (-1)

        if len(Account_positions) == 2:
            # 对冲后组合greeks
            side = 1 if Account_positions[context.c_info['symbol']
                        ]['side'] == 1 else -1
            c_greek = context.greeks[context.c_info['symbol']][greek] * \
                      Account_positions[context.c_info['symbol']]['volume'] * side

            side = 1 if Account_positions[context.p_info['symbol']
                        ]['side'] == 1 else -1
            p_greek = context.greeks[context.p_info['symbol']][greek] * \
                      Account_positions[context.p_info['symbol']]['volume'] * side

            context.greeks_after_hedging[greek] = c_greek + p_greek

    context.greeks_before_hedging_series[context.now] = context.greeks_before_hedging.copy()
    context.greeks_after_hedging_series[context.now] = context.greeks_after_hedging.copy()


def calculate_greeks(context):
    '''
    计算希腊值
    '''
    bars = context.bars

    for call_or_put, info in zip(['C', 'P'], [context.c_info, context.p_info]):
        # 获取相关参数
        s = bars[context.underlying]['close']
        bar = bars[info['symbol']]
        p = bar['close']
        k = info['exercise_price']
        delisted_date = info['delisted_date']
        t = max(option_calculate_t(context.now, delisted_date), 0.01)

        # 计算隐含波动率
        iv = max(option_calculate_iv(p, s, k, t, call_or_put), 0.01)
        # 计算greeks
        greeks = option_calculate_greeks(s, k, iv, t, call_or_put)
        # 更新greeks
        greeks.update({'iv': iv})
        context.greeks.update({bar['symbol']: greeks})


def on_backtest_finished(context, indicator):
    '''
    回测完成，画出对冲前后的风险值，iv可自己拓展画出
    '''
    if not context.greeks_before_hedging_series or not context.greeks_after_hedging_series:
        return

    df_greeks_before_hedging = pd.DataFrame(context.greeks_before_hedging_series).T
    df_greeks_before_hedging.index.name = 'datetime'
    df_greeks_before_hedging = df_greeks_before_hedging.reset_index().drop(columns='datetime')

    df_greeks_after_hedging = pd.DataFrame(context.greeks_after_hedging_series).T
    df_greeks_after_hedging.index.name = 'datetime'
    df_greeks_after_hedging = df_greeks_after_hedging.reset_index().drop(columns='datetime')

    # fig, axs = plt.subplots(2, 1)
    # df_greeks_before_hedging.plot(ax=axs[0], title='greeks_before_hedging', grid=1)
    # df_greeks_after_hedging.plot(ax=axs[1], title='greeks_after_hedging', grid=1)
    # plt.show()


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
        backtest_commission_unit回测单位手续费（元/张或手,此参数对股票不生效）
        backtest_slippage_ratio回测滑点比例
        backtest_marginfloat_ratio1回测保证金上浮比例1（距到期日>2天）
        backtest_marginfloat_ratio2回测保证金上浮比例2（据到期日<=2天）
        backtest_match_mode回测撮合模式（默认0）, 实时撮合=1（当前的bar/tick撮合）, 延时撮合=0（下一bar/tick撮合）
        '''
    run(strategy_id='378911f6-f0c2-11ef-8d98-80304917db79',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='6860051c58995ae01c30a27d5b72000bababa8e6',
        backtest_start_time='2024-10-11 09:31:00',
        backtest_end_time='2024-12-09 15:00:00',
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=400000,
        backtest_commission_unit=1,
        backtest_slippage_ratio=0.0001,
        backtest_marginfloat_ratio1=0.2,
        backtest_marginfloat_ratio2=0.4,
        backtest_match_mode=0)