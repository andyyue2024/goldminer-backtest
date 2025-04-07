# -*- coding: utf-8 -*-
import time

from gm.api import *
import os

from sq import *
from sq.test.test_gm.for159869.ForSZTrainer import ForSZTrainer

"""
sq test3
Main Features
-------------
- copy from sq_test3_3_2.py
- at 159869

"""


def init(context):
    context.start_time = time.time()
    print_log("Start...")
    # context.num = 8
    context.symbol = "SZSE.159869"
    context.start_date = "2021-02-01"
    context.end_date = "2025-02-15"
    context.model = init_model_13_6(context)
    schedule(schedule_func=algo, date_rule='1d', time_rule='10:31:00')


def init_model_13_2(context):
    """
    train model from train_data.xlsx
    :param context:
    :return:
    """
    model1 = GMModel()
    # model1.add(AllSymbolSelector(list(index_list.keys())))
    # model1.add(BlackFilter())
    model1.add(WhiteFilter([context.symbol]))
    # model1.add(TopNFilter(context.num))
    # model1.add(ManualOrder(context, model1))
    # model1.add_trainer(ForSZTrainer(context.symbol, model_index=0))
    model1.add_trainer(ForSZTrainer(context.symbol, start_date=context.start_date,
                                    end_date=context.end_date, model_index=0, fast_prediction=True))

    model1.train()


    return model1


def init_model_13_6(context):
    """
    predict by models
    :param context:
    :return:
    """
    model1 = GMModel()
    # model1.add(AllSymbolSelector(list(index_list.keys())))
    # model1.add(BlackFilter())
    model1.add(WhiteFilter([context.symbol]))
    # model1.add(TopNFilter(context.num))
    model1.add(ManualOrder(context, model1))
    # model1.add_trainer(ForSZTrainer(context.symbol, model_index=0))
    model1.add_trainer(ForSZTrainer(context.symbol, start_date=context.start_date,
                                    end_date=context.end_date, model_index=0, fast_prediction=True))
    model1.add_trainer(ForSZTrainer(context.symbol, start_date=context.start_date,
                                     end_date=context.end_date, model_index=1, fast_prediction=True))
    model1.add_trainer(ForSZTrainer(context.symbol, start_date=context.start_date,
                                     end_date=context.end_date, model_index=2, fast_prediction=True))
    model1.add_trainer(ForSZTrainer(context.symbol, start_date=context.start_date,
                                     end_date=context.end_date, model_index=3, fast_prediction=True))
    model1.add_trainer(ForSZTrainer(context.symbol, start_date=context.start_date,
                                     end_date=context.end_date, model_index=4, fast_prediction=True))
    model1.add_trainer(ForSZTrainer(context.symbol, start_date=context.start_date,
                                     end_date=context.end_date, model_index=5, fast_prediction=True))
    model1.add_trainer(ForSZTrainer(context.symbol, start_date=context.start_date,
                                     end_date=context.end_date, model_index=6, fast_prediction=True))
    # model1.add_trainer(ForSZTrainer(context.symbol, model_index=4))
    # model1.add_trainer(ForSZTrainer(context.symbol, model_index=6))
    model1.train()
    # 1.For SZSE.159869, by day, 20210201~20250214, pnl_ratio 2555.76%.
    # every executing spend about 0.001 seconds. And during period, it spends about (292) seconds.
    # model index 0: GBM_grid_1_AutoML_1_20250215_160840_model_5
    # model index 1: GBM_2_AutoML_1_20250215_162026
    # model index 2: GBM_grid_1_AutoML_1_20250215_162801_model_5
    # model index 3: GBM_grid_1_AutoML_1_20250215_162801_model_5
    # model index 4: StackedEnsemble_BestOfFamily_1_AutoML_1_20250215_164943
    # model index 5: GBM_3_AutoML_1_20250215_165858
    # model index 6: GBM_grid_1_AutoML_1_20250215_170647_model_5
    # init_model_13_6 backtest finished:
    #          {'account_id': 'a0612bb2-eb4f-11ef-83e4-00ffb355a5f1', 'pnl_ratio': 25.557646183634997,
    #           'pnl_ratio_annual': 6.324434479340186, 'sharp_ratio': 2.823978499004158,
    #           'max_drawdown': 0.07929804565190077, 'risk_ratio': 0.9999996367981517,
    #           'open_count': 24, 'close_count': 23, 'win_count': 22, 'lose_count': 1,
    #           'win_ratio': 0.9565217391304348, 'calmar_ratio': 79.75523769025688}

    return model1


def algo(context):
    print_log(context.now, " algo...")
    context.model.execute(context.now)


def on_backtest_finished(context, indicator):
    print_log(f"Done! From start, time elapsed: {time.time() - context.start_time} seconds")
    print_log(f"{context.symbol} backtest finished: ", indicator)


if __name__ == "__main__":
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
    run(
        strategy_id='a0612bb2-eb4f-11ef-83e4-00ffb355a5f1',
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        token='42c5e7778813e4a38aae2e65d70eb372ac8f2435',
        backtest_start_time="2021-02-01 09:30:00",
        backtest_end_time='2025-02-14 15:00:00',
        backtest_initial_cash=10000000,
        backtest_adjust=ADJUST_PREV,
        backtest_slippage_ratio=0.00000,
        backtest_commission_ratio=0.00000,
    )
