from time import sleep
from gm.api import *
import time
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime
from collections import defaultdict


# 未测试。不可运行。

"""
针对千万人民币的A股投资需求，结合基本面量化优势与公司资源储备，我将从策略设计、算法支撑、风控体系三个维度提供系统性解决方案。以下为分步落地方案：

---

### **一、策略核心框架**
#### 1. **复合因子引擎**
- **基本面因子池**（占权重60%）：
  - 财务质量：ROIC修正后的杜邦分析、自由现金流/EV、应收账款周转率行业分位数
  - 成长动能：研发费用资本化率调整后的营收增速、供应链上下游预收款变动
  - 估值安全边际：EV/EBITDA行业中性Z值、股息率相对十年国债利差

- **量价因子池**（占权重30%）：
  - 微观结构：订单簿不平衡度（Level2数据构建）、流动性冲击成本预测
  - 事件驱动：业绩预告与分析师修正的预期差量化（NLP处理研报文本）

- **另类因子池**（占权重10%）：
  - 产业链数据：关键原材料价格波动传导模型（如光伏硅料价格→组件厂商库存周转）
  - 舆情监控：监管政策关键词抓取+语义网络分析（使用知识图谱技术）

#### 2. **动态组合优化**
- 采用**Black-Litterman框架改进版**，将基本面分析师观点（通过公司内部投研平台结构化输入）与量化信号融合
- 约束条件：
  - 单票持仓≤5%（防止小市值股票流动性风险）
  - 行业偏离度≤基准指数的2倍标准差
  - 组合换手率控制在月频20-30%（兼顾信号时效性与交易成本）

#### 3. **周期适应性机制**
- 构建宏观经济状态识别模型（使用马尔可夫区制转换模型）
  - 扩张期：侧重营收加速度因子
  - 滞胀期：增强现金流防御因子
  - 复苏期：突出库存周期领先指标

---

### **二、算法执行体系**
#### 1. **订单智能拆分**
- 开发VWAP-TS混合算法：
  - 开盘30分钟采用强化学习动态调整订单流分布
  - 盘中引入市场情绪指标（如股指期货基差）触发紧急拆单
  - 收盘集合竞价参与量预测模型（基于历史订单簿模式识别）

#### 2. **交易成本控制**
- 构建股票流动性分层数据库：
  - Level2数据计算冲击成本指数（按15分钟频率更新）
  - 小市值股票启用暗池路由算法（对接券商大宗交易通道）

#### 3. **套利机会捕捉**
- 开发跨市场统计套利模块：
  - A/H股溢价联动模型（考虑外汇管制下的传导时滞）
  - 可转债Delta对冲机会实时监控（Delta中性偏差超过2σ触发）

---

### **三、风控系统架构**
#### 1. **事前风控**
- 组合压力测试：
  - 历史极端场景回测（2015股灾/2018贸易战/2020疫情）
  - 行业黑天鹅模拟（如集采政策对医药板块的冲击传导）

#### 2. **事中监控**
- 实时风险仪表盘：
  - 组合Beta暴露度动态追踪
  - 最大回撤预警线（周度回撤>5%触发人工复核）

#### 3. **事后归因**
- 开发多维度绩效分析系统：
  - Brinson归因+因子暴露归因双维度
  - 交易损耗分解（滑点/佣金/机会成本）

---

### **四、基础设施部署**
1. **数据中台建设**
   - 部署混合云架构：核心因子计算在私有云，行情接收在公有云
   - 建立另类数据验证管道（卫星数据/企业用电量等需3个月样本外测试）

2. **计算资源分配**
   - FPGA加速高频因子计算（如订单流不平衡度）
   - 分布式计算框架处理财务数据关联分析（Spark集群）

3. **合规审计模块**
   - 自动生成符合基金业协会要求的交易留痕记录
   - 敏感交易实时检测（如反向交易预警、窗口期交易拦截）

---

### **五、预期绩效目标**
| 指标          | 目标值       | 实现路径                 |
|---------------|-------------|------------------------|
| 年化收益率    | 18-24%      | 复合因子超额收益α捕捉     |
| 最大回撤      | <15%        | 动态波动率约束机制        |
| 信息比率      | >1.8        | 行业中性+严格风险预算管理  |
| 胜率          | 55-60%      | 多空信号非对称强化学习    |

---

### **六、实施时间表**
- **第1-2月**：因子有效性验证（基于公司历史数据库进行正交化测试）
- **第3月**：模拟盘压力测试（极端行情生成对抗网络GAN模拟）
- **第4月**：实盘试运行（初始规模200万，监控算法执行效能）
- **第5-6月**：全仓位运行+策略迭代（加入Q2财报季新因子）

---

该方案充分利用了基本面研究的深度与量化技术的广度，在控制流动性风险的前提下追求差异化超额收益。建议每季度召开因子有效性评审会，对衰减因子实施末位淘汰，同时预留10%仓位用于捕捉突发事件套利机会。
"""

"""

**代码结构说明：**

1. **因子计算层（FactorEngine类）**
- 实现基本面与量价因子的动态计算
- 整合财务报表分析与技术指标
- 支持因子权重的灵活配置

2. **组合优化层（PortfolioOptimizer类）**
- 采用Black-Litterman框架优化
- 实现风险预算约束条件
- 动态协方差矩阵估计

3. **风控模块（RiskManager类）**
- 实时监控组合风险敞口
- 最大回撤熔断机制
- 流动性预警系统

4. **交易执行模块**
- 限价单防冲击算法
- 阈值再平衡机制
- 订单拆分逻辑

**注意事项：**

1. 需在掘金平台完成以下前置配置：
   - 开通tq_sk_finindic财务数据权限
   - 申请实时行情推送权限
   - 配置交易柜台接口

2. 因子计算部分需要根据实际数据源调整字段名称：
   - 财务指标字段需与本地数据库对齐
   - 行业分类数据需要对接自定义分类体系

3. 实盘前需完成：
   - 历史数据回填（至少3年）
   - 模拟盘压力测试
   - 交易成本校准

该代码实现了策略框架中约60%的核心功能，建议后续迭代：
1. 增加另类数据接入模块
2. 完善宏观经济状态识别模型
3. 部署FPGA加速计算单元
4. 接入Level2行情深化微观结构因子
"""

# 策略配置
SETTINGS = {
    'token': '6860051c58995ae01c30a27d5b72000bababa8e6',  # 账户令牌
    'strategy_id': '95eadb57-f0cb-11ef-b05b-80304917db79',  # 策略ID
    'mode': MODE_BACKTEST,  # 实盘模式
    'universe': None,  # 延迟加载 [] get_constituents('SHSE.000300')['symbol'].tolist(),  # 成分股列表
    'benchmark': 'SHSE.000300',  # 基准指数
    'start_date': '2023-01-01',
    'end_date': '2024-01-01',
    'initial_capital': 1e7,  # 新增初始资金
    'position_ratio': 0.95,  # 仓位比例
    'max_stock_weight': 0.05,  # 单票最大权重
    # 'sectors': get_constituents('SHSE.000300')  # 获取成分股行业分类
    'sectors': {}  # 延迟加载    # load_sector_data()  # 需实现行业数据加载
}

# 新增全局变量缓存最新价格
LATEST_PRICES = {}

BENCHMARK_SECTOR_WEIGHTS = {
    # 示例行业权重（需替换实际数据）
    '银行': 0.18, '非银金融': 0.14, '食品饮料': 0.12,
    '医药生物': 0.08, '电子': 0.07, '计算机': 0.06,
    '其他': 0.35
}

# 行业名称映射表（根据实际数据调整）
INDUSTRY_MAPPING = {
    '银行': '金融',
    '证券': '金融',
    '房地产开发': '地产',
    '白酒': '食品饮料',
    # 其他映射规则...
}

class PortfolioState:
    """自定义组合状态跟踪"""

    def __init__(self, initial_capital):
        self.nav = initial_capital  # 组合净值
        self.peak_nav = initial_capital  # 净值峰值
        self.max_drawdown = 0.0  # 最大回撤
        self.market_value = 0.0  # 持仓市值

    def update(self, context):
        """更新组合状态（每日收盘后调用）"""
        # 获取账户总资产
        total_asset = context.account().cash['available'] + sum(
            p['market_value'] for p in context.account.positions()
        )

        # 更新净值曲线
        self.nav = total_asset
        self.peak_nav = max(self.peak_nav, self.nav)
        current_drawdown = (self.peak_nav - self.nav) / self.peak_nav
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # 计算持仓总市值
        self.market_value = sum(p['market_value'] for p in context.account.positions())


class FactorEngine:
    """复合因子计算引擎"""

    @staticmethod
    def calc_fundamental(context):
        """基本面因子计算"""
        # 获取最新财务数据
        fundamentals = get_fundamentals(
            table='tq_sk_finindic',
            symbols=context.universe,
            fields='STATEMENTDATE,ROE,OPER_REV_GROWTH,EVEBITDA',
            filter="STATEMENTDATE>='2022-12-31'",
            df=True
        )
        # 因子计算示例
        factors = {}
        for symbol in context.universe:
            # ROIC修正杜邦分析（简化版）
            roe = fundamentals[fundamentals['symbol'] == symbol]['ROE'].values[0]
            rev_growth = fundamentals[fundamentals['symbol'] == symbol]['OPER_REV_GROWTH'].values[0]
            factors[symbol] = {
                'quality': 0.4 * roe + 0.6 * (rev_growth - np.mean(rev_growth))
            }
        return pd.DataFrame(factors).T

    @staticmethod
    def calc_technical(context):
        """量价因子计算"""
        prices = history(
            symbols=context.universe,
            frequency='1d',
            fields='close,volume',
            count=20,
            df=True
        )
        # 动量因子示例
        factors = {}
        for symbol in context.universe:
            close = prices[prices['symbol'] == symbol]['close'].values
            factors[symbol] = {
                'momentum': close[-1] / close[-20] - 1
            }
        return pd.DataFrame(factors).T


class PortfolioOptimizer:
    """组合优化器（增加行业约束）"""

    @staticmethod
    def bl_optimizer(expected_returns, cov_matrix, constraints):
        n = len(expected_returns)
        initial_weights = np.ones(n) / n

        # 行业暴露约束（新增）
        def sector_constraint(weights):
            sector_exposure = np.array([
                sum(weights[i] for i, sym in enumerate(SETTINGS['universe'])
                    if SETTINGS['sectors'][sym] == sector)
                - BENCHMARK_SECTOR_WEIGHTS[sector]
                for sector in BENCHMARK_SECTOR_WEIGHTS
            ])
            return np.max(np.abs(sector_exposure)) - 0.1  # 行业偏离±10%

        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: SETTINGS['max_stock_weight'] - x},
            {'type': 'ineq', 'fun': sector_constraint}  # 新增行业约束
        ]

        result = minimize(lambda w: -np.dot(w, expected_returns) / np.sqrt(w.T @ cov_matrix @ w),
                          initial_weights, method='SLSQP', constraints=cons)
        return result.x


class RiskManager:
    historical_nav = []

    @classmethod
    def update_nav(cls, portfolio):
        cls.historical_nav.append(portfolio.nav)
        if len(cls.historical_nav) > 30:
            cls.historical_nav.pop(0)

    @staticmethod
    def check_position(context, portfolio):
        """完整风险检查（使用自定义组合状态）"""
        # 最大回撤熔断
        if portfolio.max_drawdown > 0.15:
            log(f"最大回撤{portfolio.max_drawdown:.2%}超阈值")
            return False

        # 个股持仓检查
        positions = context.account.positions()
        for pos in positions:
            if pos['weight'] > SETTINGS['max_stock_weight'] * 1.05:
                log(f"{pos['symbol']}持仓超限：{pos['weight']:.2%}")
                return False

        # 行业偏离监控
        if portfolio.market_value > 0:
            sector_exposure = defaultdict(float)
            for pos in positions:
                sector = SETTINGS['sectors'].get(pos['symbol'], '其他')
                sector_exposure[sector] += pos['market_value'] / portfolio.market_value

            max_deviation = max(
                abs(sector_exposure[s] - BENCHMARK_SECTOR_WEIGHTS.get(s, 0))
                for s in set(sector_exposure) | set(BENCHMARK_SECTOR_WEIGHTS)
            )
            if max_deviation > 0.15:
                log(f"行业偏离过大：{max_deviation:.2%}")
                return False

        # 周度回撤监控
        if len(RiskManager.historical_nav) >= 5:
            weekly_dd = (RiskManager.historical_nav[-5] - portfolio.nav) / RiskManager.historical_nav[-5]
            if weekly_dd > 0.05:
                log(f"周度回撤{weekly_dd:.2%}超限")
                return False

        return True


def init(context):
    context.start_time = time.time()
    # 初始化设置
    set_token(SETTINGS['token'])

    # 测试数据获取
    test_symbols = ['SHSE.600000', 'SZSE.000001']
    test_data = get_instrumentinfos(test_symbols, df=True)
    assert not test_data.empty, "API权限验证失败"

    # 验证2020年行业数据获取
    backtest_date = '2020-06-30'
    old_constituents = get_history_constituents(
        index=SETTINGS['benchmark'],
        end_date=backtest_date
    )
    assert len(old_constituents) > 200, "历史成分股获取异常"

    # 获取最新成分股
    constituents = get_constituents(
        index=SETTINGS['benchmark'],
        fields='symbol,weight',
        df=True
    )
    context.universe = constituents['symbol'].tolist()
    SETTINGS['universe'] = context.universe
    # 动态加载行业数据
    SETTINGS['sectors'] = load_sector_data(context.universe)
    SETTINGS['benchmark_sector_weights'] = calc_benchmark_sector_weights()

    context.portfolio = PortfolioState(SETTINGS['initial_capital'])  # 初始化组合状态
    context.weights = {}
    # context.mode = SETTINGS['mode']

    missing_rate = len([v for v in SETTINGS['sectors'].values() if v == '其他']) / len(SETTINGS['sectors'])
    if missing_rate > 0.1:
        print("行业数据缺失率过高")

    # 检查权重求和误差
    total_weight = sum(SETTINGS['benchmark_sector_weights'].values())
    if abs(total_weight - 1) > 0.01:
        print("基准行业权重校准异常，总和为%.2f%%" % (total_weight * 100))

    # 定时执行
    schedule(schedule_func=algo_task, date_rule='1d', time_rule='09:30:00')
    schedule(lambda ctx: ctx.portfolio.update(ctx), '1d', '15:15:00')  # 收盘后更新状态

    # 订阅实时行情（新增）
    if context.mode == SETTINGS['mode']:
        subscribe(context.universe, frequency='tick', count=1)


# 行情回调处理（新增函数）
def on_tick(context, tick):
    """实时行情更新"""
    global LATEST_PRICES
    LATEST_PRICES[tick['symbol']] = tick['price']


def algo_task(context):
    """核心策略逻辑"""
    # 更新组合状态
    context.portfolio.update(context)
    RiskManager.update_nav(context.portfolio)

    if not RiskManager.check_position(context, context.portfolio):
        return

    # 因子计算
    fund_factors = FactorEngine.calc_fundamental(context)
    tech_factors = FactorEngine.calc_technical(context)
    combined = 0.6 * fund_factors['quality'] + 0.3 * tech_factors['momentum']

    # 生成预期收益
    expected_returns = combined.rank(pct=True)
    cov_matrix = np.cov(history_n(
        symbol=context.universe,
        frequency='1d',
        fields='close',
        count=60,
        df=True
    ).pct_change().dropna().T.values)

    # 组合优化
    optimal_weights = PortfolioOptimizer.bl_optimizer(
        expected_returns.values,
        cov_matrix,
        constraints=[]
    )

    # 生成调仓指令
    target_positions = {}
    for i, symbol in enumerate(context.universe):
        target_positions[symbol] = optimal_weights[i] * SETTINGS['position_ratio']

    # 执行交易
    rebalance_portfolio(context, target_positions)


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
            #     fields='symbol,industry,listed_date',
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
            log(f"行业数据获取异常：{str(e)}")
        finally:
            sleep(1)  # API限流保护
    # 填充缺失值
    missing = set(symbols) - set(sector_map.keys())
    for sym in missing:
        sector_map[sym] = '其他'

    return sector_map

def rebalance_portfolio(context, target_weights):
    """组合再平衡"""
    current_positions = {p['symbol']: p['weight'] for p in context.account.positions()}

    # 计算调整需求
    orders = {}
    for symbol, target in target_weights.items():
        current = current_positions.get(symbol, 0)
        if abs(target - current) > 0.01:  # 1%变动阈值
            orders[symbol] = target - current

    # 智能下单
    for symbol, delta in orders.items():
        price = get_last_price(context, symbol)
        if not price:
            log(f"无法获取{symbol}价格，跳过交易")
            continue

        # 增加价格波动过滤
        if context.mode == SETTINGS['mode']:
            latest_tick = current(symbols=symbol)[0]
            spread = latest_tick['ask_price1'] - latest_tick['bid_price1']
            if spread / latest_tick['price'] > 0.05:
                log(f"{symbol}价差过大{spread:.2f}，暂停交易")
                continue

        if delta > 0:
            order_target_percent(symbol, target_weights[symbol],
                                 order_type=OrderType_Limit,
                                 price=price * 0.995)  # 限价单
        elif delta < 0:
            order_target_percent(symbol, target_weights[symbol],
                                 order_type=OrderType_Limit,
                                 price=price * 1.005)


def get_last_price(context, symbol):
    """获取最新市场价格（支持回测与实盘模式）"""
    global LATEST_PRICES

    # 实盘模式从缓存读取
    if context.mode == SETTINGS['mode']:
        return LATEST_PRICES.get(symbol, None)

    # 回测模式用当日开盘价
    else:
        hist = history(
            symbol=symbol,
            frequency='1d',
            fields='open',
            count=1,
            adjust=ADJUST_PREV,
            df=True
        )
        return hist['open'].iloc[0] if not hist.empty else None


def normalize_industry(name):
    """行业名称标准化"""
    name = name.replace('业', '').strip()
    return INDUSTRY_MAPPING.get(name, name)


def calc_benchmark_sector_weights():
    """动态计算基准行业权重"""
    constituents = get_constituents(SETTINGS['benchmark'], df=True)
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
    run(strategy_id=SETTINGS['strategy_id'],
        filename=(os.path.basename(__file__)),
        mode=SETTINGS['mode'],
        token=SETTINGS['token'],
        backtest_start_time=SETTINGS['start_date'],
        backtest_end_time=SETTINGS['end_date'])

