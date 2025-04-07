"""
-  ai-labx recommend
-  operate as follows: (bug sigal) + (hold list) - (sell sigal), and then sort
-  pnl_ratio_annual: 1.829
"""


name = "全球大类资产趋势轮动"
symbols = ["159509.SZ", "518880.SH", "159531.SZ", "513100.SH", "513520.SH", "159857.SZ", "512100.SH", "513500.SH",
           "159915.SZ", "588000.SH", "159813.SZ", "162719.SZ", "513330.SH"]
benchmark = "510300.SH"
# [date]
start_date = "20100101"
end_date = ""
# [factors]
exprs = ["roc(close,18)", "trend_score(close,26)*0.4+(roc(close,5)+roc(close,10))*0.2+ma(volume,5)/ma(volume,20)"]
names = ["roc_18", 'order_by']
# [period]
algo = "RunDaily"
# [selection]
algo = "SelectAll"
buy_rules = []
buy_at_least_count = 1
sell_rules = ['roc_18>0.15']
sell_at_least_count = 1
# [order]
factor = "order_by"
topK = 1
dropN = 0
is_desc = True
# [weight]
algo = "WeighEqually"
# [weight.fixed_weights]


# def run_task(task: Task,commission=0,slippage=0):
#     from common.dataloader_duckdb import DuckdbLoader
#     from config import DATA_ETF_QUOTES
#     loader = DuckdbLoader(path=DATA_ETF_QUOTES.resolve(),
#                           symbols=task.symbols,
#                           cols=['open', 'high', 'low', 'close', 'volume'],
#                           start_date=task.date.start_date, end_date=task.date.end_date)
#     loader.calc_all_expressions(fields=task.factors.exprs, names=task.factors.names)
#     print(loader.df)
#     e = BacktraderEngine(df_data=loader.df,commission=commission, slippage=slippage)
#     #print(e)
#     algos = task.get_algos()
#     print(algos)
#     e.run_algo_strategy(algo_list=algos, show_info=True)
#     e.show_result_empyrical()
#     e.plot(benchmark=task.benchmark)
