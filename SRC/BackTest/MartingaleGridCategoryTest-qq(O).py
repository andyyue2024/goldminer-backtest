from gm.api import *
import os
import time
from datetime import datetime, timedelta
import numpy as np
import math
from numpy.lib.scimath import logn
# 可运行，验证过。按照价格分割，逐层倍增
'''
马丁格尔-网格 策略
**策略核心机制说明：**
1. **马丁格尔加仓逻辑**：
   - 每次加仓金额 = 基准金额 × (倍数^当前层数)
   - 价格每下跌/上涨指定间隔（price_interval）触发加仓
   - 最大加仓层数控制（max_layers）

2. **动态成本计算**：
   # 多头平均成本更新公式
   new_avg_cost = (原成本 × 原仓位 + 新价格 × 新仓位) / 总仓位
   # 空头平均成本同理
   
3. **智能平仓机制**：
   - 当价格回升至平均成本时自动止盈
   - 总亏损达到阈值时强制止损
   - 资产保护机制（总资产低于基准金额50%时清仓）

4. **双向交易系统**：
   - 初始开仓方向选择基于价格相对基准位置
   - 多空仓位独立管理
   - 启用空头交易开关（enable_short）

5. **风险控制体系**：
   graph TD
   A[开始交易] --> B{价格变动}
   B -->|下跌| C[检查多头加仓条件]
   B -->|上涨| D[检查空头加仓条件]
   C --> E{满足条件?}
   E -->|是| F[按倍数加仓]
   E -->|否| G[检查止损条件]
   G -->|触发| H[强制平仓]

**策略优势：**
1. 严格遵循"亏损加倍"的马丁格尔核心理念
2. 动态计算持仓成本实现精准止盈
3. 多层风控防止过度亏损
4. 支持双向交易提升市场适应性
5. 参数体系清晰便于优化调整

**使用建议：**
1. 首次运行建议设置较小倍数（1.5-2倍）
2. 价格间隔参数需结合标的波动率设置
3. 最大层数根据资金承受能力调整
4. 建议配合历史极端行情测试
5. 实盘前需充分验证参数组合

此版本实现了经典马丁格尔策略的核心思想，通过亏损加仓、动态止盈、多层风控等机制，在控制风险的前提下追求收益。相比简单网格策略，更符合马丁格尔策略的数学期望特征。
'''


def init(context):
    context.start_time = time.time()
    # 马丁格尔核心参数
    # context.symbol = 'SHSE.10008528'  # 交易标的
    context.symbol = 'SHSE.10008547'  # 交易标的
    context.frequency = "60s"
    context.base_investment = 1000  # 基准投资金额（元）
    context.multiplier = 2  # 加仓倍数
    context.max_layers = 7  # 最大加仓层数
    context.price_interval = 0.14  # 价格间隔3%。当加仓倍数是2时，平均成本下降约为价格间隔的一半。
    context.take_profit_ratio = 0.18  # 止盈比例（2%）
    context.stop_loss_ratio = 0.30  # 总亏损止损20%
    context.base_shares = 1  # 标的下单的最小数量（手或张）
    context.base_volume = 10000  # 标的每手或每张代表的交易数量
    context.enable_short = False  # 是否启用做空

    # 策略状态变量
    context.base_price = None  # 初始基准价格
    context.long_avg_cost = 0.0        # 多头平均持仓成本
    context.long_shares = 0            # 多头总（手或张）数
    context.long_layers = 0            # 多头持仓层数
    context.short_avg_cost = 0.0       # 空头平均持仓成本
    context.short_shares = 0           # 空头总（手或张）数
    context.short_layers = 0           # 空头持仓层数
    context.last_trade_side = 0  # 最后交易方向：1-多，-1-空

    # 动态设置策略状态变量函数
    update_status(context)

    # 设置动态参数函数
    # init_parameter(context)
    # 订阅数据
    subscribe(symbols=[context.symbol], frequency=context.frequency)  # 订阅20根K线（用于计算ATR）


def update_status(context):
    # 1.重新执行时，获取账号信息，动态设置策略状态变量函数。
    # 2.订单执行后，更新设置策略状态变量函数
    account = context.account()
    positions = account.positions(symbol=context.symbol)
    # position = account.position(symbol=context.symbol, side=PositionSide_Long)

    def z_function(sn, multiplier):
        return int(logn(multiplier, sn*(multiplier-1)+1) + 0.5)

    for position in positions:
        if not context.base_price:
            context.base_price = position.vwap  # 初始基准价格
        if position.side == PositionSide_Long:
            context.long_avg_cost = position.vwap  # 多头平均持仓成本
            context.long_shares = position.volume  # 多头总股数
            context.long_layers = z_function(position.amount/context.base_investment, context.multiplier)  # 多头持仓层数
            context.last_trade_side = 1
        elif position.side == PositionSide_Short:
            context.short_avg_cost = position.vwap  # 空头平均持仓成本
            context.short_shares = position.volume  # 空头总股数
            context.short_layers = z_function(position.amount/context.base_investment, context.multiplier)  # 空头持仓层数
            context.last_trade_side = -1

    print('\tposition long. now:', context.now, ", long_shares:", context.long_shares, ", long_avg_cost:", context.long_avg_cost,
          ', final_amount:', "{:.2f}".format(context.long_avg_cost * context.long_shares * context.base_volume))
    if context.enable_short:
        print('\tposition short. now:', context.now, ", short_shares:", context.short_shares, ", short_avg_cost:", context.short_avg_cost,
              ', final_amount:', "{:.2f}".format(context.short_avg_cost * context.short_shares * context.base_volume))
    # print('\n')


def init_parameter(context):
    # add_parameter设置动态参数函数，只支持实时模式，在仿真交易和实盘交易界面查看，重启终端动态参数会被清除，重新运行策略会重新设置
    add_parameter(key='price_interval', value=context.price_interval, min=0, max=1, name='价格间隔', intro='设置价格间隔阀值', group='1',
                  readonly=False)
    # add_parameter(key='d_value', value=context.d_value, min=0, max=100, name='d值阀值', intro='设置d值阀值', group='2',
    #               readonly=False)
    print('All parameter:', context.parameters)


def on_bar(context, bars):
    current_bar = bars[0]
    if current_bar['symbol'] != context.symbol:
        return
    price = current_bar['close']

    # 初始化基准价格（以第一个K线收盘价为基准）
    if not context.base_price:
        context.base_price = price
        print(context.symbol, ", base_price:", context.base_price, ", ", context.now)
        return

    # 检查强制平仓条件
    if check_force_close(context, price):
        return

    # 主交易逻辑
    # 多头持仓处理
    if context.long_layers > 0:
        handle_long_position(context, price)

    # 空头持仓处理
    if context.short_layers > 0:
        handle_short_position(context, price)

    # 开仓逻辑（无持仓时触发）
    if context.long_layers == 0 and context.short_layers == 0:
        open_initial_position(context, price)


def handle_long_position(context, price):
    # 计算浮动盈亏
    # print('handle_long_position')
    # current_value = context.long_layers * context.base_investment
    profit_ratio = (price - context.long_avg_cost) / context.long_avg_cost

    # 止盈逻辑（回升到平均成本）
    if profit_ratio >= context.take_profit_ratio:
        print("Long，take_profit. profit_ratio:", profit_ratio, ', price:', price)
        close_long_position(context)
        return

    # 止损条件：当前价格低于平均成本的止损比例
    loss_ratio = (context.long_avg_cost - price) / context.long_avg_cost
    if loss_ratio >= context.stop_loss_ratio:
        print("Long，stop_loss. loss_ratio:", loss_ratio, ', price:', price)
        close_long_position(context)
        return

    # 加仓条件：价格下跌超过间隔且未达最大层数
    if (price <= context.long_avg_cost * (1 - context.price_interval)) and (
                context.long_layers < context.max_layers):
        add_position(context, price, side=1)
    # 达到最大加仓次数后强制平仓
    elif context.long_layers >= context.max_layers:
        print("Long，beyond max_layers. long_layers:", context.long_layers)
        close_long_position(context)


def handle_short_position(context, price):
    # 计算浮动盈亏
    # current_value = context.short_layers * context.base_investment
    profit_ratio = (context.short_avg_cost - price) / context.short_avg_cost

    # 止盈逻辑
    if profit_ratio >= context.take_profit_ratio:
        print("Short，take_profit. profit_ratio:", profit_ratio, ', price:', price)
        close_short_position(context)
        return

    # 止损条件：当前价格高于平均成本的止损比例
    loss_ratio = (price - context.short_avg_cost) / context.short_avg_cost
    if loss_ratio >= context.stop_loss_ratio:
        print("Short，stop_loss. loss_ratio:", loss_ratio, ', price:', price)
        close_short_position(context)
        return

    # 加仓逻辑
    if (price >= context.short_avg_cost * (1 + context.price_interval)) and (
                context.short_layers < context.max_layers):
        add_position(context, price, side=-1)
    # 达到最大加仓次数后强制平仓
    elif context.short_layers >= context.max_layers:
        print("Short，beyond max_layers. short_layers:", context.short_layers)
        close_short_position(context)


def add_position(context, price, side):
    # 计算加仓金额
    layer = context.long_layers if side == 1 else context.short_layers
    invest_amount = context.base_investment * (context.multiplier ** layer)

    # 计算实际成交数量（按金额计算）
    new_shares = get_shares(context, invest_amount, price)
    if new_shares <= 0:
        return
    # 执行下单
    if side == 1:
        print("Long，Buy more:", context.symbol, ", ", context.now,
              ", new_shares:",  new_shares, ", price:",  price, ", invest_amount:",  price * new_shares)

        # 更新多头平均成本（精确计算）
        if context.long_shares == 0:
            context.long_avg_cost = price
        else:
            total_cost = context.long_avg_cost * context.long_shares + price * new_shares
            context.long_avg_cost = total_cost / (context.long_shares + new_shares)
        context.long_shares += new_shares
        context.long_layers += 1

        # 执行多头加仓
        order_volume(symbol=context.symbol, volume=new_shares, side=OrderSide_Buy,
                     order_type=OrderType_Market, position_effect=PositionEffect_Open)
    else:
        print("Short，Sell more:", context.symbol, ", ", context.now,
              ", new_shares:",  new_shares,  ", price:",  price, ", invest_amount:",  price * new_shares)
        # 更新空头平均成本（精确计算）
        if context.short_shares == 0:
            context.short_avg_cost = price
        else:
            total_cost = context.short_avg_cost * context.short_shares + price * new_shares
            context.short_avg_cost = total_cost / (context.short_shares + new_shares)
        context.short_shares += new_shares
        context.short_layers += 1
        # 执行空头加仓
        order_volume(symbol=context.symbol, volume=new_shares, side=OrderSide_Sell,
                     order_type=OrderType_Market, position_effect=PositionEffect_Open)


def open_initial_position(context, price):
    # 计算初始买卖数量
    new_shares = get_shares(context, context.base_investment, price)
    if new_shares <= 0:
        return
    # 初始开仓方向选择
    if context.last_trade_side == 1 or (context.last_trade_side == 0 and price < context.base_price):
        print("Long，Buy 1st:", context.symbol, ", ", context.now,
              ", new_shares:", new_shares, ", price:", price, ", invest_amount:",  price * new_shares)
        context.long_avg_cost = price
        context.long_shares = new_shares
        context.long_layers = 1
        context.last_trade_side = 1
        order_volume(symbol=context.symbol, volume=new_shares, side=OrderSide_Buy,
                     order_type=OrderType_Market, position_effect=PositionEffect_Open)
    elif context.enable_short:
        print("Short，Sell 1st:", context.symbol, ", ", context.now,
              ", new_shares:",  new_shares, ", price:",  price, ", invest_amount:",  price * new_shares)
        context.short_avg_cost = price
        context.short_shares = new_shares
        context.short_layers = 1
        context.last_trade_side = -1
        order_volume(symbol=context.symbol, volume=new_shares, side=OrderSide_Sell,
                     order_type=OrderType_Market, position_effect=PositionEffect_Open)


def close_long_position(context):
    # account = context.account()
    # positions = account.positions()
    position = context.account().position(symbol=context.symbol, side=PositionSide_Long)
    if position and position.volume > 0:
        print("Long，Close:", context.symbol, ", ", context.now, ", long_shares:",  context.long_shares,
              ", long_avg_cost:",  context.long_avg_cost, ', total_amount:',
              "{:.2f}".format(context.long_avg_cost * context.long_shares * context.base_volume))
        context.long_layers = 0
        context.long_avg_cost = 0.0
        context.long_shares = 0
        context.last_trade_side = 0
        order_target_volume(symbol=context.symbol, volume=0, position_side=PositionSide_Long,
                            order_type=OrderType_Market)


def close_short_position(context):
    position = context.account().position(symbol=context.symbol, side=PositionSide_Short)
    if position and position.volume > 0:
        print("Short，Close:", context.symbol, ", ", context.now, ", short_shares:",  context.short_shares,
              ", short_avg_cost:",  context.short_avg_cost, ', total_amount:',
              "{:.2f}".format(context.short_avg_cost * context.short_shares * context.base_volume))
        context.short_layers = 0
        context.short_avg_cost = 0.0
        context.short_shares = 0
        context.last_trade_side = 0
        order_target_volume(symbol=context.symbol, volume=0, position_side=PositionSide_Short,
                            order_type=OrderType_Market)


def check_force_close(context, price):
    # 总资产风控
    # total_asset = context.account().cash['available']
    # if total_asset < context.base_investment * 0.5:
    #     print("total_asset is to low:", total_asset)
    #     close_long_position(context)
    #     close_short_position(context)
    #     return True
    return False


def get_shares(context, amount, price):
    if amount <= 0 or price <= 0:
        return 0
    return int(amount / price / context.base_shares / context.base_volume) * context.base_shares


# 其他接口保持空实现
def on_order_status(context, order):
    if order.status != OrderStatus_Filled:
        print('order error, now:', context.now, ', info:', order.ord_rej_reason_detail)
    update_status(context)
    # print('now:', context.now, ', order:', order)
    # account = context.account()
    # positions = account.positions()
    # print('now:', context.now, ', positions:', positions)
    # print('now:', context.now, ", long_shares:", context.long_shares, ", long_avg_cost:", context.long_avg_cost,
    #       ', final_amount:', context.long_avg_cost * context.long_shares)
    pass


def on_execution_report(context, execrpt): pass


def on_tick(context, tick): pass


def on_account_status(context, account_status): pass


def on_parameter(context, parameter):
    # print(parameter)
    if parameter['key'] == 'price_interval':
        # 通过全局变量把动态参数值传入别的事件里
        context.price_interval = parameter['value']
    elif parameter['key'] == 'd_value':
        context.d_value = parameter['value']
    else:
        return
    print('{}已经修改为{}'.format(parameter['name'], parameter['value']))


def on_error(context, code, info):
    print("on_error:", info)
    pass


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
    run(strategy_id='95eadb57-f0cb-11ef-b05b-80304917db79',
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        token='6860051c58995ae01c30a27d5b72000bababa8e6',
        backtest_start_time='2025-01-06 09:31:00',
        backtest_end_time='2025-02-28 15:00:00',
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=30000,
        backtest_commission_unit=4.5,
        backtest_slippage_ratio=0.0001,
        backtest_marginfloat_ratio1=0.2,
        backtest_marginfloat_ratio2=0.4,
        backtest_match_mode=0)

