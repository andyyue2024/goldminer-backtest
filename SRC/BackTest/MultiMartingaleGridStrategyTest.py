from typing import Dict

from gm.api import *
import os
import time
from MultiMartingaleGridStrategy import MultiMartingaleGridStrategy

# 可运行，验证过。按照价格分割，逐层倍增。支持多标的。
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
    context.frequency = "60s"
    context.symbol_list = [ # 6
        'SHSE.10008546',
        'SHSE.10008555'
    ]
    # context.symbol_list = [ # 3
    #     'SHSE.10008526',
    #     'SHSE.10008536'
    # ]

    interval_parameters = {'price_interval_min': 0.14,   # 6月0.14
                           'take_profit_ratio_max': 0.23,   # 6月0.23
                           'price_interval_max': 0.14,  # 3月0.25
                           'take_profit_ratio_min': 0.23,  # 3月0.13
                           }
    context.symbol_strategies = {symbol: MultiMartingaleGridStrategy(context, symbol, base_investment=1500,
                                                                     price_interval=0.14, take_profit_ratio=0.23,
                                                                     # enable_interval=True, interval_parameters=interval_parameters
                                                                     ) for symbol in context.symbol_list}

    print('Symbol List:', context.symbol_list, '\ncurrent position:')
    # 动态设置策略状态变量函数
    for symbol, strategy in context.symbol_strategies.items():
        strategy.update_status()

    # 设置动态参数函数
    # init_parameter(context)
    # 订阅数据
    subscribe(symbols=context.symbol_list, frequency=context.frequency)  # 订阅20根K线（用于计算ATR）


def init_parameter(context):
    # add_parameter设置动态参数函数，只支持实时模式，在仿真交易和实盘交易界面查看，重启终端动态参数会被清除，重新运行策略会重新设置
    add_parameter(key='price_interval', value=context.price_interval, min=0, max=1, name='价格间隔', intro='设置价格间隔阀值', group='1',
                  readonly=False)
    # add_parameter(key='d_value', value=context.d_value, min=0, max=100, name='d值阀值', intro='设置d值阀值', group='2',
    #               readonly=False)
    print('All parameter:', context.parameters)


def on_bar(context, bars):
    # print("on_bar:", bars)
    for bar in bars:
        strategy = context.symbol_strategies[bar.symbol]
        if strategy:
            strategy.on_bar(bar)


# 其他接口保持空实现
def on_order_status(context, order):
    if order.status != OrderStatus_Filled:
        print('on_order_status, now:', context.now, ', info:', order.status, ', ', order.ord_rej_reason_detail)

    print('current position:')
    for symbol, strategy in context.symbol_strategies.items():
        strategy.update_status()
    # print('now:', context.now, ', order:', order)
    # account = context.account()
    # positions = account.positions()
    # print('now:', context.now, ', positions:', positions)
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
            backtest_slippage_ratio回测滑点比例
            backtest_match_mode市价撮合模式，以下一tick/bar开盘价撮合:0，以当前tick/bar收盘价撮合：1
        '''
    run(strategy_id='95eadb57-f0cb-11ef-b05b-80304917db79',
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        token='6860051c58995ae01c30a27d5b72000bababa8e6',
        backtest_start_time='2024-12-01 09:31:00',
        backtest_end_time='2025-03-07 15:00:00',
        # backtest_start_time='2024-09-01 09:31:00',
        # backtest_end_time='2025-01-07 15:00:00',
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=45000,
        backtest_commission_unit=4.5,
        backtest_slippage_ratio=0.0001,
        backtest_marginfloat_ratio1=0.2,
        backtest_marginfloat_ratio2=0.4,
        backtest_match_mode=0)

