from typing import Dict

from gm.api import *
import os
import time
from gm.api import *
from numpy.lib.scimath import logn
from datetime import datetime, timedelta

# 可运行，验证过。按照价格分割，逐层倍增。支持多标的。面向实盘。还未完成
# 该文件包含MultiMartingaleGridStrategy类
'''
马丁格尔-网格 多标的组合策略
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


class MultiMartingaleGridStrategy:
    def __init__(self,
                 context,
                 symbol='SHSE.10000',  # 交易标的
                 base_investment=2000,  # 基准投资金额（元）
                 multiplier=2,  # 加仓倍数
                 max_layers=9,  # 最大加仓层数
                 price_interval=0.20,  # 价格间隔3%。当加仓倍数是2时，平均成本下降约为价格间隔的一半。
                 take_profit_ratio=0.15,  # 止盈比例（2%）
                 stop_loss_ratio=0.30,  # 总亏损止损20%
                 base_shares=1,  # 标的下单的最小数量（手或张）
                 base_volume=10000,  # 标的每手或每张代表的交易数量
                 enable_short=False,  # 是否启用做空
                 enable_interval=False,  # 是否使用区间参数
                 interval_parameters=None,
                 ):
        if interval_parameters is None:
            interval_parameters = {'price_interval_min': 0.14,
                                   'take_profit_ratio_max': 0.23,
                                   'price_interval_max': 0.25,
                                   'take_profit_ratio_min': 0.13, }

        self.context = context
        # 马丁格尔核心参数
        self.symbol = symbol
        self.base_investment = base_investment
        self.multiplier = multiplier
        self.max_layers = max_layers
        self.price_interval = price_interval
        self.take_profit_ratio = take_profit_ratio
        self.stop_loss_ratio = stop_loss_ratio
        self.base_shares = base_shares
        self.base_volume = base_volume
        self.enable_short = enable_short

        self.enable_interval = enable_interval
        self.price_interval_min = interval_parameters['price_interval_min']
        self.price_interval_max = interval_parameters['price_interval_max']
        self.take_profit_ratio_min = interval_parameters['take_profit_ratio_min']
        self.take_profit_ratio_max = interval_parameters['take_profit_ratio_max']

        self.symbol_info = get_instrumentinfos(symbols=self.symbol)
        if self.symbol_info:
            self.symbol_info = self.symbol_info[0]
            self.listed_date: datetime = self.symbol_info['listed_date']
            self.delisted_date: datetime = self.symbol_info['delisted_date']
            # self.base_volume = self.symbol_info['multiplier']

        # 策略状态变量
        self.base_price = None  # 初始基准价格
        self.long_avg_cost = 0.0  # 多头平均持仓成本
        self.long_shares = 0  # 多头总（手或张）数
        self.long_layers = 0  # 多头持仓层数
        self.short_avg_cost = 0.0  # 空头平均持仓成本
        self.short_shares = 0  # 空头总（手或张）数
        self.short_layers = 0  # 空头持仓层数
        self.last_trade_side = 0  # 最后交易方向：1-多，-1-空

    def update_status(self):
        # 1.重新执行时，获取账号信息，动态设置策略状态变量函数。
        # 2.订单执行后，更新设置策略状态变量函数
        account = self.context.account()
        positions = account.positions(symbol=self.symbol)

        # position = account.position(symbol=self.symbol, side=PositionSide_Long)

        def z_function(sn, multiplier):
            return int(logn(multiplier, sn * (multiplier - 1) + 1) + 0.5)

        for position in positions:
            if not self.base_price:
                self.base_price = position.vwap  # 初始基准价格
            if position.side == PositionSide_Long:
                self.long_avg_cost = position.vwap  # 多头平均持仓成本
                self.long_shares = position.volume  # 多头总股数
                self.long_layers = z_function(position.amount / self.base_investment, self.multiplier)  # 多头持仓层数
                self.last_trade_side = 1
            elif position.side == PositionSide_Short:
                self.short_avg_cost = position.vwap  # 空头平均持仓成本
                self.short_shares = position.volume  # 空头总股数
                self.short_layers = z_function(position.amount / self.base_investment, self.multiplier)  # 空头持仓层数
                self.last_trade_side = -1

        print('\t\t', self.symbol, ', position long. now:', self.context.now, ", long_shares:", self.long_shares, ", long_avg_cost:",
              self.long_avg_cost,
              ', final_amount:', "{:.2f}".format(self.long_avg_cost * self.long_shares * self.base_volume))
        if self.enable_short:
            print('\t\t', self.symbol, ', position short. now:', self.context.now, ", short_shares:", self.short_shares, ", short_avg_cost:",
                  self.short_avg_cost,
                  ', final_amount:', "{:.2f}".format(self.short_avg_cost * self.short_shares * self.base_volume))
        # print('\n')

    def on_bar(self, bar):
        # current_bar, = (bar for bar in bars if bar['symbol'] == self.symbol)
        if not bar:
            return
        price = bar['close']

        # 初始化基准价格（以第一个K线收盘价为基准）
        if not self.base_price:
            self.base_price = price
            print('Base price: \n\t\t', self.symbol, ", base_price:", self.base_price, ", ", self.context.now)
            return
        if self.enable_interval:
            self.update_input_parameters()

        # 检查强制平仓条件
        if self.check_force_close(price):
            return

        # 主交易逻辑
        # 多头持仓处理
        if self.long_layers > 0:
            self.handle_long_position(price)

        # 空头持仓处理
        if self.short_layers > 0:
            self.handle_short_position(price)

        # 开仓逻辑（无持仓时触发）
        if self.long_layers == 0 and self.short_layers == 0:
            self.open_initial_position(price)

    def handle_long_position(self, price):
        # 计算浮动盈亏
        # print('handle_long_position')
        # current_value = self.long_layers * self.base_investment
        profit_ratio = (price - self.long_avg_cost) / self.long_avg_cost

        # 止盈逻辑（回升到平均成本）
        if profit_ratio >= self.take_profit_ratio:
            print("Long，Take profit: ", self.symbol, ", ", self.context.now, ", profit ratio:", profit_ratio, ', price:', price)
            self.close_long_position()
            return

        # 止损条件：当前价格低于平均成本的止损比例
        loss_ratio = (self.long_avg_cost - price) / self.long_avg_cost
        if loss_ratio >= self.stop_loss_ratio:
            print("Long，stop_loss. loss_ratio:", loss_ratio, ', price:', price)
            self.close_long_position()
            return

        # 加仓条件：价格下跌超过间隔且未达最大层数
        if (price <= self.long_avg_cost * (1 - self.price_interval)) and (
                self.long_layers < self.max_layers):
            self.add_position(price, side=1)
        # 达到最大加仓次数后强制平仓
        elif self.long_layers >= self.max_layers:
            print("Long，beyond max_layers. long_layers:", self.long_layers)
            self.close_long_position()

    def handle_short_position(self, price):
        # 计算浮动盈亏
        # current_value = self.short_layers * self.base_investment
        profit_ratio = (self.short_avg_cost - price) / self.short_avg_cost

        # 止盈逻辑
        if profit_ratio >= self.take_profit_ratio:
            print("Short，Take profit: ", self.symbol, ", ", self.context.now, ", profit ratio:", profit_ratio, ', price:', price)
            self.close_short_position()
            return

        # 止损条件：当前价格高于平均成本的止损比例
        loss_ratio = (price - self.short_avg_cost) / self.short_avg_cost
        if loss_ratio >= self.stop_loss_ratio:
            print("Short，stop_loss. loss_ratio:", loss_ratio, ', price:', price)
            self.close_short_position()
            return

        # 加仓逻辑
        if (price >= self.short_avg_cost * (1 + self.price_interval)) and (
                self.short_layers < self.max_layers):
            self.add_position(price, side=-1)
        # 达到最大加仓次数后强制平仓
        elif self.short_layers >= self.max_layers:
            print("Short，beyond max_layers. short_layers:", self.short_layers)
            self.close_short_position()

    def add_position(self, price, side):
        # 计算加仓金额
        layer = self.long_layers if side == 1 else self.short_layers
        invest_amount = self.base_investment * (self.multiplier ** layer)

        # 计算实际成交数量（按金额计算）
        new_shares = self.get_shares(invest_amount, price)
        if new_shares <= 0:
            return
        # 执行下单
        if side == 1:
            print("Long，Buy more:", self.symbol, ", ", self.context.now,
                  ", new_shares:", new_shares, ", price:", price, ", invest_amount:", price * new_shares)

            # 更新多头平均成本（精确计算）
            if self.long_shares == 0:
                self.long_avg_cost = price
            else:
                total_cost = self.long_avg_cost * self.long_shares + price * new_shares
                self.long_avg_cost = total_cost / (self.long_shares + new_shares)
            self.long_shares += new_shares
            self.long_layers += 1

            # 执行多头加仓
            order_volume(symbol=self.symbol, volume=new_shares, side=OrderSide_Buy,
                         order_type=OrderType_Market, position_effect=PositionEffect_Open)
        else:
            print("Short，Sell more:", self.symbol, ", ", self.context.now,
                  ", new_shares:", new_shares, ", price:", price, ", invest_amount:", price * new_shares)
            # 更新空头平均成本（精确计算）
            if self.short_shares == 0:
                self.short_avg_cost = price
            else:
                total_cost = self.short_avg_cost * self.short_shares + price * new_shares
                self.short_avg_cost = total_cost / (self.short_shares + new_shares)
            self.short_shares += new_shares
            self.short_layers += 1
            # 执行空头加仓
            order_volume(symbol=self.symbol, volume=new_shares, side=OrderSide_Sell,
                         order_type=OrderType_Market, position_effect=PositionEffect_Open)

    def open_initial_position(self, price):
        # 计算初始买卖数量
        new_shares = self.get_shares(self.base_investment, price)
        if new_shares <= 0:
            return
        # 初始开仓方向选择
        if self.last_trade_side == 1 or (self.last_trade_side == 0 and price < self.base_price):
            print("Long，Buy 1st:", self.symbol, ", ", self.context.now,
                  ", new_shares:", new_shares, ", price:", price, ", invest_amount:", price * new_shares)
            self.long_avg_cost = price
            self.long_shares = new_shares
            self.long_layers = 1
            self.last_trade_side = 1
            order_volume(symbol=self.symbol, volume=new_shares, side=OrderSide_Buy,
                         order_type=OrderType_Market, position_effect=PositionEffect_Open)
        elif self.enable_short:
            print("Short，Sell 1st:", self.symbol, ", ", self.context.now,
                  ", new_shares:", new_shares, ", price:", price, ", invest_amount:", price * new_shares)
            self.short_avg_cost = price
            self.short_shares = new_shares
            self.short_layers = 1
            self.last_trade_side = -1
            order_volume(symbol=self.symbol, volume=new_shares, side=OrderSide_Sell,
                         order_type=OrderType_Market, position_effect=PositionEffect_Open)

    def close_long_position(self):
        # account = self.account()
        # positions = account.positions()
        position = self.context.account().position(symbol=self.symbol, side=PositionSide_Long)
        if position and position.volume > 0:
            print("Long，Close:", self.symbol, ", ", self.context.now, ", long_shares:", self.long_shares,
                  ", long_avg_cost:", self.long_avg_cost, ', total_amount:',
                  "{:.2f}".format(self.long_avg_cost * self.long_shares * self.base_volume))
            self.long_layers = 0
            self.long_avg_cost = 0.0
            # self.long_shares = 0
            self.last_trade_side = 0
            # order_target_volume(symbol=self.symbol, volume=0, position_side=PositionSide_Long,
            #                     order_type=OrderType_Market)
            order_volume(symbol=self.symbol, volume=self.long_shares, side=OrderSide_Sell,
                         order_type=OrderType_Market, position_effect=PositionEffect_Close)

    def close_short_position(self):
        position = self.context.account().position(symbol=self.symbol, side=PositionSide_Short)
        if position and position.volume > 0:
            print("Short，Close:", self.symbol, ", ", self.context.now, ", short_shares:", self.short_shares,
                  ", short_avg_cost:", self.short_avg_cost, ', total_amount:',
                  "{:.2f}".format(self.short_avg_cost * self.short_shares * self.base_volume))
            self.short_layers = 0
            self.short_avg_cost = 0.0
            # self.short_shares = 0
            self.last_trade_side = 0
            # order_target_volume(symbol=self.symbol, volume=0, position_side=PositionSide_Short,
            #                     order_type=OrderType_Market)
            order_volume(symbol=self.symbol, volume=self.short_shares, side=OrderSide_Buy,
                         order_type=OrderType_Market, position_effect=PositionEffect_Close)

    def check_force_close(self, price):
        # 总资产风控
        # total_asset = self.account().cash['available']
        # if total_asset < self.base_investment * 0.5:
        #     print("total_asset is to low:", total_asset)
        #     close_long_position(self)
        #     close_short_position(self)
        #     return True
        return False

    def get_shares(self, amount, price):
        if amount <= 0 or price <= 0:
            return 0
        return int(amount / price / self.base_shares / self.base_volume) * self.base_shares

    def get_day_delta(self):
        """
        Calculate day delta between self.delisted_date and self.context.now
        :return:
        """
        time_delta = self.delisted_date - self.context.now
        return time_delta.days

    def update_input_parameters(self):
        """
        Linear adjustment of parameter values
        :return:
        """
        LEFT_DAYS = 203
        RIGHT_DAYS = 113
        day_delta = self.get_day_delta()
        if day_delta >= LEFT_DAYS:
            self.price_interval = self.price_interval_min
            self.take_profit_ratio = self.take_profit_ratio_max
        elif day_delta <= RIGHT_DAYS:
            self.price_interval = self.price_interval_max
            self.take_profit_ratio = self.take_profit_ratio_min
        else:
            ratio = (day_delta - LEFT_DAYS) / (RIGHT_DAYS - LEFT_DAYS)
            self.price_interval = self.price_interval_min + (self.price_interval_max - self.price_interval_min) * ratio
            self.take_profit_ratio = self.take_profit_ratio_max + (self.take_profit_ratio_min - self.take_profit_ratio_max) * ratio


# def order_volume(symbol,
#                  volume,
#                  side,
#                  order_type,
#                  position_effect,
#                  price=0,
#                  order_duration=OrderDuration_Unknown,
#                  order_qualifier=OrderQualifier_Unknown,
#                  account="",
#                  ):
#     # 执行多头加仓
#     # order_volume(symbol=self.symbol, volume=new_shares, side=OrderSide_Buy,
#     #              order_type=OrderType_Market, position_effect=PositionEffect_Open)
#     # 执行空头加仓
#     # order_volume(symbol=self.symbol, volume=new_shares, side=OrderSide_Sell,
#     #              order_type=OrderType_Market, position_effect=PositionEffect_Open)
#
#     pass


def init(context):
    context.start_time = time.time()
    context.frequency = "60s"
    # context.symbol_list = [ # 12
    #     'SHSE.10009222',
    #     'SHSE.10009231',
    # ]
    context.symbol_list = [ # 9
        'SHSE.10008800',
        'SHSE.10008809',
    ]
    # context.symbol_list = [ # 6
    #     # 'SHSE.10008557',
    #     'SHSE.10008546',
    #     'SHSE.10008555'
    # ]
    # context.symbol_list = [ # 3
    #     'SHSE.10008525',
    #     'SHSE.10008526',
    #     'SHSE.10008536'
    #     'SZSE.90004231',
    #     'SZSE.90004232'
    # ]

    # interval_parameters = {'price_interval_min': 0.14,   # 6月0.14
    #                        'take_profit_ratio_max': 0.23,   # 6月0.23
    #                        'price_interval_max': 0.25,  # 3月0.25
    #                        'take_profit_ratio_min': 0.13,  # 3月0.13
    #                        }
    context.symbol_strategies = {symbol: MultiMartingaleGridStrategy(context, symbol, base_investment=1350,
                                                                     price_interval=0.14, take_profit_ratio=0.25,
                                                                     stop_loss_ratio=0.50
                                                                     # enable_interval=True, interval_parameters=interval_parameters
                                                                     ) for symbol in context.symbol_list}

    print('Symbol List:', context.symbol_list, '\ncurrent position:')
    # 动态设置策略状态变量函数
    for symbol, strategy in context.symbol_strategies.items():
        strategy.update_status()

    # 设置动态参数函数
    # init_parameter(context)
    # 订阅数据
    subscribe(symbols=context.symbol_list, frequency=context.frequency)
    subscribe(symbols=context.symbol_list, frequency='tick')


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


def on_tick(context, tick):
    if context.now.hour == 14 and context.now.minute == 0 and context.now.second == 0:
        context.last_tmp_time = context.now
        print('current position in on_tick:')
        for symbol, strategy in context.symbol_strategies.items():
            strategy.update_status()


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
        backtest_start_time='2025-03-01 09:31:00',
        backtest_end_time='2025-06-07 15:00:00',
        # backtest_start_time='2024-09-01 09:31:00',
        # backtest_end_time='2025-01-07 15:00:00',
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=40000,
        backtest_commission_unit=4.5,
        backtest_slippage_ratio=0.0001,
        backtest_marginfloat_ratio1=0.2,
        backtest_marginfloat_ratio2=0.4,
        backtest_match_mode=0)

