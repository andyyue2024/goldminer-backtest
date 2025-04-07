from time import sleep
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from gm.api import *
import time
import os
from collections import defaultdict, deque
import logging

# 可运行。但逻辑缺少行业数据

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 策略配置类（替代全局变量）
class StrategyConfig:
    def __init__(self):
        self.token = '6860051c58995ae01c30a27d5b72000bababa8e6'
        self.strategy_id = '95eadb57-f0cb-11ef-b05b-80304917db79'
        self.mode = MODE_BACKTEST
        self.benchmark = 'SHSE.000300'
        self.start_date = '2023-01-01 09:31:00'
        self.end_date = '2024-01-01 15:00:00'
        self.initial_capital = 1e7
        self.position_ratio = 0.95
        self.max_stock_weight = 0.05
        self.max_sector_deviation = 0.10
        self.circuit_breaker_dd = 0.15
        self.trade_price_adjust = 0.003
        self.risk_check_window = 30


class FactorEngine:
    """因子计算引擎（支持缓存优化）"""

    def __init__(self, context):
        self.context = context
        self.cache = {}

    def _get_fundamentals(self) -> pd.DataFrame:
        """获取财务数据（带缓存）"""
        cache_key = ('fundamentals', self.context.now)
        if cache_key not in self.cache:
            data = get_fundamentals(
                table='tq_sk_finindic',
                symbols=self.context.universe,
                fields='STATEMENTDATE,ROE,OPER_REV_GROWTH,EVEBITDA',
                filter="STATEMENTDATE>='2022-12-31'",
                df=True
            )
            self.cache[cache_key] = data.pivot(index='symbol', columns='statementdate')
        return self.cache[cache_key]

    def calc_fundamental(self) -> pd.DataFrame:
        """增强版基本面因子"""
        try:
            data = self._get_fundamentals()
            factors = {}
            for symbol in self.context.universe:
                # 示例因子：质量因子（需扩展）
                roe = data.loc[symbol]['ROE'].iloc[-1]
                rev_growth = data.loc[symbol]['OPER_REV_GROWTH'].iloc[-1]
                factors[symbol] = {
                    'quality': 0.4 * roe + 0.6 * (rev_growth - rev_growth.mean())
                }
            return pd.DataFrame(factors).T
        except Exception as e:
            logger.error(f"基本面因子计算异常: {str(e)}")
            return pd.DataFrame()


class PortfolioOptimizer:
    """组合优化器（支持多约束条件）"""

    def __init__(self, context):
        self.context = context

    def _sector_constraint(self, weights: np.ndarray) -> float:
        """行业偏离度约束"""
        sector_exposure = defaultdict(float)
        for i, symbol in enumerate(self.context.universe):
            sector = self.context.sectors.get(symbol, '其他')
            sector_exposure[sector] += weights[i]
        max_dev = max(
            abs(sector_exposure[s] - self.context.benchmark_sectors.get(s, 0))
            for s in sector_exposure
        )
        return self.context.config.max_sector_deviation - max_dev

    def optimize(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Black-Litterman优化"""
        n = len(expected_returns)
        initial_weights = np.ones(n) / n

        # 构建约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: self.context.config.max_stock_weight - x},
            {'type': 'ineq', 'fun': self._sector_constraint}
        ]

        # 优化目标函数
        result = minimize(
            lambda w: -np.dot(w, expected_returns) / np.sqrt(w.T @ cov_matrix @ w),
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, self.context.config.max_stock_weight)] * n
        )

        if not result.success:
            logger.warning("组合优化未收敛: %s", result.message)
            return {}

        return dict(zip(self.context.universe, result.x))


class RiskManager:
    """增强型风控管理"""

    def __init__(self, context):
        self.context = context
        self.nav_history = deque(maxlen=context.config.risk_check_window)

    def check_max_drawdown(self) -> bool:
        """动态回撤监控"""
        current_dd = (self.context.portfolio.peak_nav - self.context.portfolio.nav) / self.context.portfolio.peak_nav
        if current_dd > self.context.config.circuit_breaker_dd:
            logger.critical("触发熔断机制：当前回撤 %.2f%%", current_dd * 100)
            self._circuit_breaker()
            return False
        return True

    def check_sector_exposure(self) -> bool:
        """行业偏离检查"""
        sector_exposure = defaultdict(float)
        for pos in self.context.account().positions():
            sector = self.context.sectors.get(pos['symbol'], '其他')
            sector_exposure[sector] += pos['weight']

        max_dev = max(
            abs(sector_exposure[s] - self.context.benchmark_sectors.get(s, 0))
            for s in set(sector_exposure) | set(self.context.benchmark_sectors)
        )
        return max_dev <= self.context.config.max_sector_deviation

    def _circuit_breaker(self):
        """熔断执行：平仓所有头寸"""
        logger.warning("执行熔断平仓")
        for pos in self.context.account().positions():
            order_target_percent(pos['symbol'], 0)


class TradeExecutor:
    """智能交易执行器"""

    def __init__(self, context):
        self.context = context
        self.price_adjust = context.config.trade_price_adjust

    def _get_limit_price(self, symbol: str, side: str) -> float:
        """动态限价单定价"""
        try:
            tick = current(symbols=symbol)[0]
            spread = tick['ask_price1'] - tick['bid_price1']
            mid_price = (tick['ask_price1'] + tick['bid_price1']) / 2

            if side == 'BUY':
                return mid_price * (1 - self.price_adjust + spread / mid_price * 0.5)
            else:
                return mid_price * (1 + self.price_adjust - spread / mid_price * 0.5)
        except Exception as e:
            logger.warning("获取限价失败: %s", str(e))
            return history(symbol=symbol, frequency='1d', fields='close', count=1, df=True)['close'].iloc[0]

    def rebalance(self, target_weights: Dict[str, float]):
        """智能再平衡"""
        current_pos = {p['symbol']: p['weight'] for p in self.context.account().positions()}

        for symbol, target in target_weights.items():
            current = current_pos.get(symbol, 0)
            delta = target - current

            if abs(delta) < 0.005:  # 0.5%阈值
                continue

            if delta > 0:
                price = self._get_limit_price(symbol, 'BUY')
                order_target_percent(symbol, target, OrderType_Limit, price)
            else:
                price = self._get_limit_price(symbol, 'SELL')
                order_target_percent(symbol, target, OrderType_Limit, price)


class PortfolioState:
    """组合状态跟踪"""

    def __init__(self, initial_capital: float):
        self.nav = initial_capital
        self.peak_nav = initial_capital
        self.market_value = 0.0

    def update(self, context):
        total_asset = context.account().cash['available'] + sum(
            p['market_value'] for p in context.account().positions()
        )
        self.nav = total_asset
        self.peak_nav = max(self.peak_nav, self.nav)
        self.market_value = sum(p['market_value'] for p in context.account().positions())


def init(context):
    context.start_time = time.time()
    """策略初始化"""
    context.config = StrategyConfig()
    context.symbol = context.config.benchmark
    context.portfolio = PortfolioState(context.config.initial_capital)

    # 初始化模块
    context.factor_engine = FactorEngine(context)
    context.optimizer = PortfolioOptimizer(context)
    context.risk_manager = RiskManager(context)
    context.trade_executor = TradeExecutor(context)

    # 加载基础数据
    load_initial_data(context)

    # 定时任务
    schedule(algo_task, '1d', '09:30:00')
    schedule(lambda _: context.portfolio.update(context), '1d', '15:15:00')

    if context.config.mode == MODE_LIVE:
        subscribe(context.universe, frequency='tick', count=1)


def algo_task(context):
    """策略主逻辑"""
    # 更新组合状态
    context.portfolio.update(context)

    # 风控检查
    if not context.risk_manager.check_max_drawdown():
        return
    if not context.risk_manager.check_sector_exposure():
        return

    # 因子计算
    fund_factors = context.factor_engine.calc_fundamental()
    tech_factors = context.factor_engine.calc_technical()

    # 组合优化
    expected_returns = 0.6 * fund_factors['quality'] + 0.3 * tech_factors['momentum']
    cov_matrix = get_historical_covariance(context.universe)
    target_weights = context.optimizer.optimize(expected_returns, cov_matrix)

    # 执行交易
    context.trade_executor.rebalance(target_weights)


def calc_benchmark_sectors(ctx):
    return {}


def load_initial_data(context):
    """初始化数据加载"""
    # 加载成分股
    constituents = get_constituents(context.config.benchmark, df=True)
    context.universe = constituents['symbol'].tolist()
    # context.universe = []

    # 加载行业数据
    # context.sectors = load_sector_data(context.universe)
    # context.benchmark_sectors = calc_benchmark_sectors(context)
    context.sectors = {}
    context.benchmark_sectors = {}


def get_historical_covariance(symbols: List[str], lookback: int = 60) -> pd.DataFrame:
    """历史协方差矩阵计算"""
    prices = history(symbols, '1d', 'close', lookback, df=True)
    returns = prices.pct_change().dropna()
    return returns.cov()


def load_sector_data(symbols: list):
    """行业数据加载优化版"""
    BATCH_SIZE = 100  # 掘金API单次最大查询量
    sector_map = {}

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        try:
            # 获取基础信息
            # data = get_instrumentinfos(
            #     symbols=batch,
            #     # fields='symbol,industry,listed_date',
            #     df=True
            # )
            data = stk_get_symbol_industry(symbols=batch)
            # 处理返回数据
            if not data.empty:
                for _, row in data.iterrows():
                    sector = row['industry']
                    # 处理特殊字符
                    if pd.isna(sector) or sector.strip() == '':
                        sector = '其他'
                    sector_map[row['symbol']] = sector.split('-')[0]  # 取一级行业
        except Exception as e:
            print(f"行业数据获取异常：{str(e)}")
        finally:
            sleep(1)  # API限流保护
    # 填充缺失值
    missing = set(symbols) - set(sector_map.keys())
    for sym in missing:
        sector_map[sym] = '其他'

    return sector_map


def calc_benchmark_sector_weights():
    """动态计算基准行业权重"""
    constituents = get_constituents(config.benchmark, df=True)
    sectors = load_sector_data(constituents['symbol'].tolist())

    total_mktcap = constituents['mkt_cap'].sum()
    sector_weights = constituents.groupby(
        constituents['symbol'].map(sectors)
    )['mkt_cap'].sum() / total_mktcap

    return sector_weights.to_dict()


def on_backtest_finished(context, indicator):
    print(f"Done! From start, time elapsed: {time.time() - context.start_time} seconds")
    print(f"{context.symbol} backtest finished: ", indicator)


if __name__ == '__main__':
    config = StrategyConfig()
    run(strategy_id=config.strategy_id,
        filename=(os.path.basename(__file__)),
        mode=config.mode,
        token=config.token,
        backtest_start_time=config.start_date,
        backtest_end_time=config.end_date)
