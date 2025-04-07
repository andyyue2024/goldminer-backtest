# -*- coding: utf-8 -*-
from gm.api import *
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import minimize
from sklearn.preprocessing import quantile_transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os
import time
from datetime import datetime, timedelta

# 部分测试。部分可运行。

"""
针对幻方量化千万人民币A股投资需求，结合公司资源与市场特性，设计以下交易策略及执行框架：

---

### **一、策略核心架构**
1. **策略定位：中频统计套利（3-5天持仓周期）**
   - **优势**：平衡高频交易冲击成本与低频策略容量限制，规避T+1约束
   - **收益来源**：多因子选股+事件驱动套利+微观结构定价误差修正

2. **数据引擎**
   - **基础数据**：全市场Tick级订单簿、逐笔成交、Level2行情（覆盖沪深港通标的）
   - **增强数据**：
     - 另类数据：产业链供应链数据（海关/货运）、限售股解禁日历、大宗交易折溢价
     - 行为数据：龙虎榜游资模式识别、融资融券情绪指标
     - 舆情数据：基于NLP的监管文件语义分析（证监会/交易所公告）

---

### **二、模型开发体系**
1. **因子工场**
   - **传统因子**：改进型CAMPHA-HOLD（引入流动性调整的动量因子）
   - **创新因子**：
     - 盘口压力指数：委买/委卖订单簿厚度动态比
     - 资金流隐形拆解：基于VPIN理论的异常大单识别
     - 政策敏感性评分：监管关键词共现网络分析

2. **集成学习框架**
   - **初级组合**：XGBoost特征筛选（处理高维稀疏数据）
   - **深度强化学习**：DDPG算法优化仓位动态调整
   - **异构模型融合**：通过Attention机制整合基本面对冲模型（财务质量打分）与技术面模型

---

### **三、交易执行系统**
1. **智能拆单算法**
   - **动态冰山订单**：依据实时买卖队列深度调整订单暴露比例
   - **波动率自适应TWAP**：结合GARCH波动率预测调整时段内交易量分布

2. **冲击成本控制**
   - 构建个股流动性冲击曲面（基于历史大单成交弹性分析）
   - 设置单票最大冲击成本阈值（≤15bps）

---

### **四、风险控制矩阵**
1. **事前防控**
   - 行业敞口约束：单一申万一级行业≤12%
   - 风格中性：Barra风险模型多因子暴露度归零

2. **实时监控**
   - 熔断应急协议：监测市场广度指标（如腾落线突变）
   - 反脆弱模块：波动率突变时自动切换至防御性组合（高股息+低Beta）

3. **事后归因**
   - 引入AP（主动管理）风险预算体系
   - 基于SHAP值的因子风险贡献度分解

---

### **五、基础设施部署**
1. **硬件层**：部署FPGA加速订单簿解析（重点处理集合竞价阶段订单流）
2. **软件栈**：构建Docker化策略容器，实现策略间资源隔离与快速迭代
3. **合规审计**：交易日志区块链存证（满足《证券期货业网络信息安全管理办法》）

---

### **六、预期绩效指标**
| 指标         | 目标值       | 监控频率   |
|--------------|--------------|------------|
| 年化收益率   | 28%-35%      | 日度       |
| 最大回撤     | ≤12%         | 实时       |
| 胜率         | ≥58%         | 周度       |
| 信息比率     | ≥2.3         | 月度       |
| 换手率       | 45-55倍/月   | 日度       |

---

### **七、实施路线图**
1. **第1-2周**：因子有效性检验（Fama-MacBeth回归+正交化处理）
2. **第3-4周**：蒙特卡洛压力测试（极端市场情景建模）
3. **第5周**：10亿级历史交易回测（2016-2023全周期覆盖）
4. **第6周**：实盘模拟运行（中金所仿真环境）
5. **第7周**：启动500万试运行，动态调整参数
6. **第8周**：全资金投入，开启持续优化循环

---

该方案充分融合幻方量化在深度学习与高频交易领域的积累，通过多层次风险控制确保策略稳健性。建议每周召开三次跨团队策略会议（量化研究、IT、合规），利用公司已有GPU集群加速模型训练迭代。
"""

# 策略参数
SETTINGS = {
    'symbol': 'SHSE.000300',  # 沪深300成分股
    'trade_window': '14:50:00',  # 交易时间窗口
    'max_holding': 5,  # 最大持仓天数
    'industry_limit': 0.12,  # 行业限制
    'max_drawdown': 0.12,  # 风控
    'single_position_limit': 0.05  # 单票仓位限制
}

# 新增模型配置参数
MODEL_CONFIG = {
    'train_window': 500,  # 训练数据窗口长度
    'label_horizon': 3,  # 预测未来3日收益
    'test_size': 0.2,  # 验证集比例
    'retrain_interval': 20  # 每20个交易日重新训练
}

# 新增账户管理参数
ACCOUNT_CONFIG = {
    # 'account_id': '您的账户ID',  # 掘金实盘账户ID
    'cash_limit': 0.05,  # 最低现金保留比例
    'slippage': 0.001  # 默认滑点设置
}


def init(context):
    context.start_time = time.time()

    # 初始化账户对象
    context.account = context.account()

    # 持仓记录器（用于计算回撤）
    context.equity_curve = []

    # 加载行业分类数据
    # context.industry_map = get_instruments(exchanges='SHSE,SZSE', fields='symbol,industry', df=True)
    context.industry_map = get_instruments(exchanges='SHSE,SZSE', df=True)
    # context.industry_map = get_instruments(exchanges='SHSE,SZSE', fields='symbol,sw_industry1',  # 使用申万一级行业
    #                                        df=True).set_index('symbol')['sw_industry1']

    # 初始化机器学习模型
    context.model = None
    context.last_retrain = context.now - timedelta(days=30)
    # context.model = GradientBoostingRegressor(n_estimators=100)

    # 定时任务
    schedule(schedule_func=algo_task, date_rule='1d', time_rule=SETTINGS['trade_window'])
    # subscribe(symbols='SHSE.600519', frequency='1s', count=2)


def on_bar(context, bars):
    # data = context.data(symbol='SHSE.600519', frequency='60s', count=1)
    pass


def algo_task(context):
    print('algo_task:', context.now)
    # 获取候选池
    symbols = get_components(SETTINGS['symbol'])

    # 检查是否需要重新训练模型
    if (context.now - context.last_retrain).days >= MODEL_CONFIG['retrain_interval']:
        train_model(context, symbols)
        context.last_retrain = context.now

    # 特征工程（包含特征+标签）
    features, labels = build_features_and_labels(context, symbols)

    if context.model and not features.empty:
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # 模型预测
        predict = context.model.predict(X_scaled)
        # 生成信号
        # signals = generate_signals(pd.Series(predict, index=features.index))
        signals = generate_signals(predict, index=features.index)

        # 组合优化
        target_weights = portfolio_optimization(context, signals)

        # 执行交易
        execute_trades(context, target_weights)


def train_model(context, symbols):
    """模型训练模块"""
    # 获取历史数据
    features, labels = build_features_and_labels(context, symbols, mode='train')

    if len(features) > 100:  # 最小样本量要求
        # 创建预处理管道
        pipeline = make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            )
        )

        # 划分训练验证集
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=MODEL_CONFIG['test_size'], shuffle=False
        )

        # 训练模型
        pipeline.fit(X_train, y_train)

        # 验证集评估（示例打印R平方）
        val_score = pipeline.score(X_val, y_val)
        print(f'Model retrained | Validation R^2: {val_score:.3f}')

        # 更新模型
        context.model = pipeline


def build_features_and_labels(context, symbols, mode='infer'):
    """构建特征和标签数据集"""
    # 获取历史数据
    # data = get_multi_data(
    #     symbols=symbols,
    #     fields='open,high,low,close,volume,amount',
    #     frequency='1d',
    #     count=MODEL_CONFIG['train_window'] + MODEL_CONFIG['label_horizon'],
    #     df=True
    # )
    data = history(symbol=symbols, frequency='1d', fields='symbol,open,high,low,close,volume,amount',
                   start_time=str(context.now - timedelta(days=2*int(MODEL_CONFIG['train_window'] + MODEL_CONFIG['label_horizon'])))[:10],
                   end_time=str(context.now)[:10], df=True)

    features = pd.DataFrame()
    labels = pd.Series()

    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol]
        if len(symbol_data) < MODEL_CONFIG['train_window']:
            continue

        # 特征计算
        df = symbol_data.copy()
        df['returns'] = df['close'].pct_change()

        # 动量因子（5日收益率）
        df['momentum_5'] = df['close'].pct_change(5)

        # 流动性因子（20日平均成交额）
        df['liquidity_20'] = df['amount'].rolling(20).mean()

        # 波动率因子（20日波动率）
        df['volatility_20'] = df['returns'].rolling(20).std()

        # 特征选择
        feature_cols = ['momentum_5', 'liquidity_20', 'volatility_20']
        features = pd.concat(
            [features, df[feature_cols].iloc[-MODEL_CONFIG['train_window']:-MODEL_CONFIG['label_horizon']]])

        # 标签构建：未来N日收益率
        future_ret = df['close'].shift(-MODEL_CONFIG['label_horizon']).pct_change(MODEL_CONFIG['label_horizon'])
        labels = pd.concat([labels, future_ret.iloc[-MODEL_CONFIG['train_window']:-MODEL_CONFIG['label_horizon']]])

    # 清洗数据
    valid_idx = labels.dropna().index.intersection(features.dropna().index)
    return features.loc[valid_idx], labels.loc[valid_idx]


def build_features(context, symbols):
    # 获取多维度数据
    data = history(symbol=symbols, frequency='1d', fields='symbol,open,high,low,close,volume,amount',
                   start_time=str(context.now - timedelta(days=20))[:10], end_time=str(context.now)[:10], df=True)

    # 计算技术因子
    features = pd.DataFrame()
    for symbol in symbols:
        # 动量因子
        features.loc[symbol, 'momentum'] = data[data['symbol'] == symbol]['close'].pct_change(5).iloc[-1]

        # 流动性因子
        features.loc[symbol, 'liquidity'] = data[data['symbol'] == symbol]['amount'].rolling(5).mean().iloc[-1]

        # 波动率因子
        features.loc[symbol, 'volatility'] = data[data['symbol'] == symbol]['close'].pct_change().std()

    return features.dropna()


def portfolio_optimization(context, signals):
    # 行业中性约束
    industry_exposure = check_industry_exposure(context, signals)

    # 风险平价优化
    weights = risk_parity_optimization(context, signals, industry_exposure)

    return weights


def execute_trades(context, target_weights):
    """修正后的交易执行模块"""
    # 获取当前持仓
    current_positions = {p.symbol: p for p in context.account.positions()}

    # 计算可用资金（保留5%现金）
    available_cash = context.account.cash * (1 - ACCOUNT_CONFIG['cash_limit'])

    # 计算当前总权益
    total_equity = context.account.cash + sum(
        p.market_value for p in context.account.positions()
    )

    # 更新权益曲线
    context.equity_curve.append(total_equity)

    # 执行调仓
    for symbol, weight in target_weights.items():
        target_value = total_equity * weight
        current_pos = current_positions.get(symbol, None)

        # 计算需要调整的金额
        if current_pos:
            delta = target_value - current_pos.market_value
        else:
            delta = target_value

        # 控制单笔委托金额
        if abs(delta) < 10000:  # 忽略小于1万的调整
            continue

        # 方向判断
        side = OrderSide_Buy if delta > 0 else OrderSide_Sell

        # 冲击成本检查
        if check_impact_cost(symbol, abs(delta)):
            # 使用VWAP算法单
            order_volume(
                symbol=symbol,
                volume=abs(delta) / context.last_price(symbol),  # 近似股数
                side=side,
                order_type=OrderType_Market,
                position_effect=PositionEffect_Open,
                # price='vwap'  # 使用智能路由
            )

    # 止损检查
    check_drawdown_control(context)


def get_components(index_symbol):
    """
    获取指数成分股（需预先维护成分股列表）
    此处使用SHSE.000300的示例成分股，实际应接入公司成分股数据库
    """
    try:
        constituents = get_constituents(index=index_symbol, fields='symbol,weight')
        return [item['symbol'] for item in constituents]
    except Exception as e:
        print(f"成分股获取失败: {str(e)}")

    # 示例成分股（实际需维护完整列表）
    sample_constituents = [
        'SHSE.600519',
        'SZSE.000858', 'SHSE.601318',
        # 'SHSE.600036', 'SZSE.000333', 'SHSE.601012'
    ]
    return sample_constituents


def generate_signals(pred, index, quantile=0.2):
    """
    生成多空信号引擎
    :param index:
    :param pred: 模型预测的收益率序列
    :param quantile: 信号分位阈值
    :return: 标准化后的多空信号(-1到1)
    """
    # 收益率分位数转换
    signals = quantile_transform(pred.reshape(-1, 1), output_distribution='normal').flatten()

    # 生成多空信号
    threshold_upper = np.quantile(signals, 1 - quantile)
    threshold_lower = np.quantile(signals, quantile)

    # 信号标准化
    signals = np.where(signals > threshold_upper, 1,
                       np.where(signals < threshold_lower, -1, 0))
    return pd.Series(signals, index=index)


# ---------- 风控模块 ----------
def check_industry_exposure(context, signals):
    # # 合并权重与行业信息
    # industry_weights = pd.concat([
    #     weights.rename('weight'),
    #     context.industry_map.rename('industry')
    # ], axis=1)
    # # 按行业聚合
    # return industry_weights.groupby('industry')['weight'].sum()

    industry_exposure = signals.groupby(context.industry_map['industry']).sum()
    return industry_exposure.clip(upper=SETTINGS['industry_limit'])


def calculate_industry_exposure(context, weights):
    """
    计算当前组合的行业暴露
    :param weights: 个股权重Series，索引为symbol
    :return: 行业暴露Series
    """
    # 合并权重与行业信息
    industry_weights = pd.concat([
        weights.rename('weight'),
        context.industry_map.rename('industry')
    ], axis=1)

    # 按行业聚合
    return industry_weights.groupby('industry')['weight'].sum()


def risk_parity_optimization(context, signals, industry_exposure):
    """
    简化的风险平价优化器（含行业约束）
    :param context:
    :param signals: 原始信号序列
    :param industry_exposure: 行业暴露约束
    :return: 优化后的权重Series
    """
    # 初筛：保留绝对值前30%的信号
    valid_signals = signals[np.abs(signals) > np.quantile(np.abs(signals), 0.7)]

    # 基础权重：波动率倒数加权
    hist_vol = valid_signals.index.to_series().apply(
        lambda x: history_n(x, frequency='1d', fields='close', count=20,
                            end_time=context.now)['close'].pct_change().std())
    base_weights = (1 / hist_vol).replace(np.inf, 0)

    # 行业约束优化

    def optimization_target(weights):
        # 风险平价目标：各资产风险贡献方差最小化
        risk_contrib = weights * hist_vol.values
        risk_contrib /= risk_contrib.sum()
        return np.var(risk_contrib)

    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
        {'type': 'ineq', 'fun': lambda w: SETTINGS['single_position_limit'] - w.max()}  # 单票上限
    ]

    # 行业暴露约束（示例处理一个行业）
    industry_weights = base_weights.groupby(industry_exposure.loc[base_weights.index]).sum()
    for industry in industry_weights.index:
        constraints.append(
            {'type': 'ineq', 'fun': lambda w, ind=industry: SETTINGS['industry_limit'] - industry_weights[ind]}
        )

    # 求解优化问题
    result = minimize(
        optimization_target,
        x0=base_weights.values / base_weights.sum(),
        bounds=[(0, SETTINGS['single_position_limit'])] * len(base_weights),
        constraints=constraints,
        method='SLSQP'
    )

    return pd.Series(result.x, index=base_weights.index).sort_values(ascending=False)


def check_impact_cost(symbol, target_value):
    # 获取流动性指标
    liquidity = history_n(symbol, frequency='1d', fields='liquidate', count=5)
    impact_cost = target_value / liquidity.mean()
    return impact_cost < 0.0015  # 15bps阈值


def check_drawdown_control(context):
    """修正后的回撤监控模块"""
    if len(context.equity_curve) < 2:
        return

    # 计算当前回撤
    peak = np.maximum.accumulate(context.equity_curve)
    current_dd = (peak[-1] - context.equity_curve[-1]) / peak[-1]

    # 执行风控
    if current_dd > SETTINGS['max_drawdown']:
        print(f"触发最大回撤风控：{current_dd * 100:.1f}%")
        close_all_positions(context)


def close_all_positions(context):
    """清仓所有头寸"""
    for position in context.account.positions():
        order_target_percent(
            symbol=position.symbol,
            percent=0,
            order_type=OrderType_Market,
            position_side=PositionSide_Long
        )


def on_backtest_finished(context, indicator):
    print(f"Done! From start, time elapsed: {time.time() - context.start_time} seconds")
    print(f"{context.symbol} backtest finished: ", indicator)


# ---------- 回测配置 ----------
if __name__ == '__main__':
    run(
        # strategy_id='a0612bb2-eb4f-11ef-83e4-00ffb355a5f1',
        # filename=(os.path.basename(__file__)),
        # mode=MODE_BACKTEST,
        # token='42c5e7778813e4a38aae2e65d70eb372ac8f2435',
        strategy_id='94d38672-f0e0-11ef-8d7f-80304917db79',
        filename=(os.path.basename(__file__)),
        mode=MODE_BACKTEST,
        token='6860051c58995ae01c30a27d5b72000bababa8e6',
        # backtest_start_time=backtest_start_time,
        # backtest_end_time=backtest_end_time,
        # backtest_start_time="2021-03-05 09:30:00",
        # backtest_end_time='2025-02-14 15:00:00',
        backtest_start_time="2024-06-18 09:30:00",
        backtest_end_time='2024-09-24 15:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
