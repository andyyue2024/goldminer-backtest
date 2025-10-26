import optuna
import json
from dataclasses import fields, is_dataclass, dataclass, field, asdict
from typing import Dict, Any, List, Dict
import toml
from backtrader_engine import  Engine


@dataclass
class Task:
    name: str = '策略'
    symbols: List[str] = field(default_factory=list)

    start_date: str = '20100101'
    end_date: str = None

    benchmark: str = '510300.SH'
    select: str = 'SelectAll'

    select_buy: List[str] = field(default_factory=list)
    buy_at_least_count: int = 0
    select_sell: List[str] = field(default_factory=list)
    sell_at_least_count: int = 1

    order_by_signal: str = ''
    order_by_topK: int = 1
    order_by_dropN: int = 0
    order_by_DESC: bool = True  # 默认从大至小排序

    weight: str = 'WeighEqually'
    weight_fixed: Dict[str, int] = field(default_factory=dict)
    period: str = 'RunDaily'
    period_days: int = None

def json_to_task(json_str: str) -> Task:
    """将 JSON 字符串转换为 Task 实例"""
    data = json.loads(json_str)
    return Task(**data)

def task_to_toml(task: Task) -> str:
    """将 Task 实例转换为 TOML 字符串"""
    return toml.dumps(asdict(task))

def dict_to_task(data: Dict[str, Any]) -> Task:
    """将字典安全转换为 Task 实例"""
    # 获取 Task 类的字段集合
    valid_fields = {f.name for f in fields(Task)}

    # 过滤非法字段并进行类型检查
    filtered_data = {}
    for key, value in data.items():
        if key not in valid_fields:
            continue

        # 获取字段类型信息
        field_type = Task.__annotations__.get(key)

        # # 简单类型校验（可选）
        # if field_type and not isinstance(value, field_type):
        #     try:
        #         # 尝试类型转换（如 str -> List）
        #         value = field_type(value)
        #     except (TypeError, ValueError):
        #         raise ValueError(
        #             f"字段 '{key}' 类型不匹配，预期 {field_type}，实际 {type(value)}"
        #         )

        filtered_data[key] = value

    return Task(**filtered_data)

def objective(trial):
    # 建议参数范围
    P1 = trial.suggest_int('P1',3, 32, step=1)
    P2 = trial.suggest_int('P2', 3, 32, step=1)
    P3 = trial.suggest_int('P3', 3, 32, step=1)
    P4 = trial.suggest_int('P4', 10, 32, step=1)

    t = Task()
    t.name = 'etf轮动'
    # 排序
    t.period = 'RunDaily'
    t.weight = 'WeighEqually'
    t.order_by_signal = f'trend_score(close,{P1})*0.25+(roc(close,{P2})+roc(close,{P3}))*0.17+ma(volume,5)/ma(volume,18)'
    t.select_sell = ['roc(close,20)>0.158']
    t.start_date = '20231204'
    t.end_date = '20250601'

    t.symbols = [
        '513290.SH',
        '513520.SH',
        '159509.SZ',
        '513030.SH',
        '159915.SZ',
        '563300.SH',
        '588100.SH',
        '513040.SH',
        '563000.SH',
        '159939.SZ',
        '515230.SH',
        '515980.SH',
        '159819.SZ',
        '162719.SZ',
        '518880.SH',
        '513330.SH',
        '513180.SH',
        '513130.SH',
        '159505.SZ',
        '513090.SH',
        '159792.SZ',
        '159857.SZ',
        '159887.SZ',
        '561600.SH',
        '588000.SH',
        '513500.SH',
        '512480.SH',
        '513100.SH',
        '513380.SH',
        '588110.SH',
        '515880.SH',

    ]

    t.benchmark = '510300.SH'
    e = Engine(path='quotes')

    # t.order_by_signal = 'trend_score(close,$P)'.replace('$P', str(p))
    strategy_results = e.run(t)[0]
    # 获取分析结果

    sharpe_ratio = strategy_results.analyzers.sharpe.get_analysis()
    drawdown = strategy_results.analyzers.drawdown.get_analysis()
    returns = strategy_results.analyzers.returns.get_analysis()

    # 计算目标值（综合考量夏普比率和回撤）
    sharpe_value = sharpe_ratio.get('sharperatio', 0)
    max_drawdown = drawdown.get('max', {'drawdown': 100}).get('drawdown', 100)

    # 避免除零错误
    if max_drawdown == 0:
        max_drawdown = 0.1

    annual_return = returns.get('rnorm100', 0)  # 年化收益百分比

    # 综合评分：夏普比率越高越好，回撤越小越好
    composite_score = annual_return #sharpe_value * (1 - max_drawdown / 100)

    # 设置试验属性以便后续分析
    trial.set_user_attr('final_value', e.cerebro.broker.getvalue())
    trial.set_user_attr('max_drawdown', max_drawdown)
    trial.set_user_attr('sharpe_ratio', sharpe_value)

    return composite_score


if __name__ == '__main__':
    # 创建研究
    study = optuna.create_study(
        direction='maximize',  # 最大化综合评分
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # 开始优化
    study.optimize(objective, n_trials=100)

    # 输出最佳结果
    print("\n=== 优化结果 ===")
    print(f'最佳试验: {study.best_trial.number}')
    print(f'最佳参数:')
    for key, value in study.best_trial.params.items():
        print(f'  {key}: {value}')
    print(f'最佳目标值: {study.best_trial.value:.4f}')
    print(f'最终资产: {study.best_trial.user_attrs["final_value"]:.2f}')
    print(f'夏普比率: {study.best_trial.user_attrs["sharpe_ratio"]:.4f}')
    print(f'最大回撤: {study.best_trial.user_attrs["max_drawdown"]:.2f}%')