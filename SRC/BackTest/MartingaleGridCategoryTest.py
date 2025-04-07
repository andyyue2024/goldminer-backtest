from gm.api import *
import os
import time
from datetime import datetime, timedelta
import numpy as np
import math

# 不可运行，未验证过。未全面调试过。
'''
马丁格尔-网格策略

'''




def init(context):
    context.start_time = time.time()
    # 策略参数
    context.symbol = 'SZSE.159869'  # 交易标的
    context.initialized = False  # 网格初始化标志
    context.max_grid = 5  # 最大网格层级
    context.base_invest = 100000  # 基础投资金额（单位：元）
    context.step_pct = 0.02  # 网格步长百分比
    context.current_grid = 0  # 当前持仓网格层级
    context.grid_positions = {}  # 网格持仓记录 {层级: 数量}
    context.pending_orders = set()  # 待成交订单集合
    context.frequency = "1740s"

    # 订阅标的（示例为沪深300股指期货）
    subscribe(context.symbol, context.frequency)


def on_bar(context, bars):
    bar = bars[0]
    if bar.symbol != context.symbol:
        return

    # 初始化网格参数（以第一个BAR的收盘价为基准）
    if not context.initialized:
        context.initial_price = bar.close
        context.step = context.initial_price * context.step_pct
        context.initialized = True
        print(f'网格初始化完成 基准价:{context.initial_price:.2f} 步长:{context.step:.2f}')
        return

    current_price = bar.close

    # 计算当前网格层级
    price_diff = context.initial_price - current_price
    current_level = math.floor(price_diff / context.step)
    current_level = max(0, min(current_level, context.max_grid))

    # 买入逻辑：价格下跌触发新层级
    if current_level > context.current_grid:
        for target_level in range(context.current_grid + 1, current_level + 1):
            # 计算投资金额（马丁格尔加倍）
            invest_amount = context.base_invest * (2 ** (target_level - 1))

            # 获取可用资金
            account = context.account()
            cash_available = account.cash

            # 资金检查
            if cash_available < invest_amount:
                print(f'资金不足，无法买入层级{target_level} 需{invest_amount:.0f}元')
                break

            # 计算可买数量（股指期货1手=1单位）
            quantity = int(invest_amount // (current_price * 300))  # 假设合约乘数300
            if quantity < 1:
                print(f'价格过高无法买入层级{target_level}')
                break

            # 发送买入订单
            order = order_volume(symbol=context.symbol, volume=quantity, side=OrderSide_Buy,
                                 order_type=OrderType_Market, position_effect=PositionEffect_Open,
                                 user_data={'grid_level': target_level})

            if order:
                context.pending_orders.add(order.order_id)
                print(f'买入下单 层级{target_level} 数量{quantity}手')

    # 卖出逻辑：价格上涨触发平仓
    elif current_level < context.current_grid:
        # 计算需要平仓的层级数
        levels_to_close = context.current_grid - current_level
        for _ in range(levels_to_close):
            close_level = context.current_grid
            if close_level not in context.grid_positions:
                break

            # 获取该层持仓数量
            position_qty = context.grid_positions[close_level]
            if position_qty <= 0:
                break

            # 发送卖出订单
            order = order_volume(symbol=context.symbol, volume=position_qty, side=OrderSide_Sell,
                                 order_type=OrderType_Market, position_effect=PositionEffect_Close,
                                 user_data={'grid_level': close_level})

            if order:
                context.pending_orders.add(order.order_id)
                print(f'卖出下单 层级{close_level} 数量{position_qty}手')


def on_execution_report(context, execrpt):
    # 只处理成交事件
    if execrpt.exec_type != ExecType_Trade:
        return

    # 移除已完成订单
    order_id = execrpt.order_id
    if order_id in context.pending_orders:
        context.pending_orders.remove(order_id)

    # 获取网格层级信息
    grid_level = execrpt.user_data.get('grid_level')
    if not grid_level:
        return

    # 处理买入成交
    if execrpt.side == OrderSide_Buy and execrpt.position_effect == PositionEffect_Open:
        context.grid_positions[grid_level] = execrpt.volume
        context.current_grid = max(context.current_grid, grid_level)
        print(f'买入成交 层级{grid_level} 数量{execrpt.volume}手')

    # 处理卖出成交
    elif execrpt.side == OrderSide_Sell and execrpt.position_effect == PositionEffect_Close:
        if grid_level in context.grid_positions:
            del context.grid_positions[grid_level]
            # 更新当前持仓层级
            if context.grid_positions:
                context.current_grid = max(context.grid_positions.keys())
            else:
                context.current_grid = 0
            print(f'卖出成交 层级{grid_level} 数量{execrpt.volume}手')


# 其他回调函数保持空实现
def on_order_status(context, order): pass


def on_account_status(context, account_status): pass


def on_parameter(context, parameter): pass


def on_error(context, code, info): print(f'错误代码:{code}, 信息:{info}')


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

