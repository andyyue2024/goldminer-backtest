# -*- coding: utf-8 -*-
import time
import os
from gm.api import *
from sq import *
from sq.source import ailabx

"""
ai labx test

"""


def init(context):
    context.start_time = time.time()
    print_log("Start...")
    context.num = 1
    context.symbol = "AAA"
    context.model = create_model(context)
    schedule(schedule_func=algo, date_rule='1d', time_rule='09:31:00')
    # # call PyCallGraph and Graphviz
    # with PyCallGraph(output=GraphvizOutput()):
    #     context.model = test_model(context)


def create_model(context):
    model1 = GMModel()
    # model1.add(AllSymbolSelector(list(index_list.keys())))
    # model1.add(BlackFilter())
    model1.add(WhiteFilter(list(index_list.keys())))
    model1.add(AILabxSorter(context))
    model1.add(TopNFilter(context.num))
    model1.add(AILabxOrder(context.num))

    if context.mode == MODE_BACKTEST:
        ailabx.get_all_data(list(index_list.keys()), context.backtest_start_time, context.backtest_end_time)
    # model1.execute(context.now)
    return model1


def algo(context):
    print_log(context.now, " algo...")
    # last_day = get_previous_trading_date("SHSE", context.now)
    context.model.execute(context.now)


def on_backtest_finished(context, indicator):
    print_log(f"Done! From start, time elapsed: {time.time() - context.start_time} seconds")
    print_log(f"{context.symbol} backtest finished: ", indicator)


index_list = {
    # List
    "SZSE.159509": "纳指科技ETF",
    "SHSE.518880": "黄金ETF",
    "SHSE.512480": "半导体ETF",
    "SZSE.159531": "中证2000ETF",
    "SHSE.513100": "纳指ETF",
    "SHSE.513520": "日经ETF",
    "SZSE.159857": "光伏ETF",
    "SHSE.512100": "中证1000ETF",
    "SHSE.510180": "上证180ETF",
    "SHSE.588000": "科创50ETF",
    "SHSE.513330": "恒生互联网ETF",
    "SZSE.162719": "石油LOF",
    "SHSE.513500": "标普500ETF",
    "SZSE.159915": "创业板ETF",

}

if __name__ == "__main__":
    run(
        strategy_id='19236129-09e5-11f0-99ab-00155dd6c843',
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        token='6860051c58995ae01c30a27d5b72000bababa8e6',
        backtest_start_time="2023-09-19 09:30:00",
        backtest_end_time='2025-03-27 15:00:00',
        # backtest_end_time='2023-10-20 15:00:00',
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0000,
        backtest_commission_unit=1,
        backtest_slippage_ratio=0.0001,
        backtest_marginfloat_ratio1=0.2,
        backtest_marginfloat_ratio2=0.4,
        backtest_match_mode=0)

