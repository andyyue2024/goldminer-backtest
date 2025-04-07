# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import os
import time
import pandas as pd
from datetime import timedelta, datetime

"""
长期备兑策略
注：此策略只用于学习、交流、演示，不构成任何投资建议。
- 建立底仓，买入510050etf基金10w份，同时卖出当月虚值3档认购期权合约10张。
- 当行权价与510050的当前价格区间小于0.1时，平仓期权合约，同时卖出新的当月虚值3档认购期权合约。
- 当持有的期权合约到期时间小于等于5个交易日时，进行移仓换月，平仓期权合约，同时卖出新的次月虚值3档认购期权合约。
- 策略采用虚值3档合约，建议回测时间段设置在2018年以后。
"""


# 策略中必须有init方法
def init(context):
    context.start_time = time.time()
    # 标的物
    context.underlying_symbol = 'SHSE.510050'
    # 建立多头底仓
    long_open_underlying(context)

    # 期权symbol
    context.c_symbol = None
    # 当月合约
    context.execute_month = 1
    # 更新间距
    context.grid = 0.1
    # 上一合约到期时间
    context.last_delisted_date = None

    # 获取虚值合约并且订阅
    get_symbol(context)

    # 重新订阅新合约行情
    unsubscribe(symbols='*', frequency='1d')
    subscribe(symbols=[context.c_symbol[0],
                       context.underlying_symbol], frequency='1d')


def get_symbol(context):
    if context.c_symbol:
        positions = context.account().positions()
        if positions:
            for position in positions:
                if position['side'] == 2 and position['covered_flag'] == 1:
                    ## 持仓合约平仓
                    # order_volume(symbol=position['symbol'], volume=position['volume'], side=OrderSide_Buy,
                    #               order_type=OrderType_Market, position_effect=PositionEffect_Close)
                    # 备兑仓可采用“备兑平仓”(若开仓时采用“备兑开仓”下单)
                    option_covered_close(symbol=position['symbol'], volume=position['volume'],
                                         order_type=OrderType_Market)

    # 标的物价格(回测模式为context.now的当日收盘价,实时模式为最新tick的price,可在subscribe中设置日内频度,如'3600s')
    if context.mode == MODE_LIVE:
        data = current(context.underlying_symbol)
        if data:
            s = data[0]['price']
        else:
            print('{}current没有获取到{}数据'.format(context.now, context.underlying_symbol))
            return
    elif context.mode == MODE_BACKTEST:
        data = history_n(symbol=context.underlying_symbol, frequency='1d', count=1, end_time=context.now,
                         fields='close')
        if data:
            s = data[0]['close']
        else:
            print('{}history_n没有获取到{}数据'.format(context.now, context.underlying_symbol))
            return

    # 获取最新的虚值3档认购合约
    context.c_symbol = option_get_symbols_by_in_at_out(underlying_symbol=context.underlying_symbol,
                                                       trade_date=context.now,
                                                       execute_month=context.execute_month,
                                                       call_or_put='C',
                                                       in_at_out=-3,
                                                       s=s,
                                                       adjust_flag='M')
    # 若当日不存在虚值3档认购合约，用虚值2档认购合约代替
    if len(context.c_symbol) == 0:
        context.c_symbol = option_get_symbols_by_in_at_out(underlying_symbol=context.underlying_symbol,
                                                           trade_date=context.now,
                                                           execute_month=context.execute_month,
                                                           call_or_put='C',
                                                           in_at_out=-2,
                                                           s=s,
                                                           adjust_flag='M')
        if not context.c_symbol:
            print('{}没有获取到context.c_symbol的数据, 请检查当天是否为交易日'.format(context.now))
            return

    # 获取合约信息
    context.c_info = get_instrumentinfos(symbols=context.c_symbol, fields='sec_name,delisted_date,exercise_price')
    if context.c_info:
        print(context.now, context.c_symbol[0], context.c_info[0]['sec_name'], context.c_info[0]['delisted_date'])
    else:
        print('{}没有获取到context.c_info的数据'.format(context.now))
        return

    ## 新合约开仓
    # order_volume(symbol=context.c_symbol[0], volume=10, side=OrderSide_Sell,
    #              order_type=OrderType_Market, position_effect=PositionEffect_Open)
    # 可采用“备兑开仓”豁免保证金
    option_covered_open(symbol=context.c_symbol[0], volume=10, order_type=OrderType_Market)




def long_open_underlying(context):
    # 建立底仓：现货多头10w股，实盘运行需要指定现货账户acoount_id
    order_volume(symbol=context.underlying_symbol, volume=100000, side=OrderSide_Buy,
                 order_type=OrderType_Market, position_effect=PositionEffect_Open)


def on_bar(context, bars):
    # 前一合约是否已到期，如果是恢复获取当月合约
    if context.last_delisted_date:
        if (datetime.date(context.last_delisted_date) - datetime.date(context.now)).days < 0:
            context.last_delisted_date = None
            context.execute_month = 1

    for bar in bars:
        # 判断行权价与当前510050价格的间距小于grid，如果小于更新期权合约
        if bar['symbol'] == context.underlying_symbol:
            close = bar['close']
            K = context.c_info[0]['exercise_price']
            if abs(close - K) < context.grid:
                get_symbol(context)
                print(context.now, '小于grid，更新期权合约')

        # 判断到期日是否小于等于5个交易日，如果小于则更新次月合约
        delisted_days = get_trading_dates(exchange='SHSE', start_date=context.now,
                                          end_date=context.c_info[0]['delisted_date'])
        if len(delisted_days) <= 5:
            context.last_delisted_date = context.c_info[0]['delisted_date']
            context.execute_month = 2
            get_symbol(context)
            print(context.now, '移仓换月')


def on_order_status(context, order):
    if order.status != OrderStatus_Filled:
        print('on_order_status, now:', context.now, ', info:', order.status, ', ', order.ord_rej_reason_detail)


def on_backtest_finished(context, indicator):
    print(f"Done! From start, time elapsed: {time.time() - context.start_time} seconds")
    print(f"{context.symbols} backtest finished: ", indicator)


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
    run(strategy_id='ed44b9ac-078a-11f0-a529-00155dd6c843',
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        token='6860051c58995ae01c30a27d5b72000bababa8e6',
        backtest_start_time='2018-03-02 08:00:00',
        backtest_end_time='2021-03-02 16:00:00',
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=300000,
        backtest_commission_ratio=0.0000,
        backtest_commission_unit=1,
        backtest_slippage_ratio=0.0001,
        backtest_marginfloat_ratio1=0.2,
        backtest_marginfloat_ratio2=0.4,
        backtest_match_mode=0)