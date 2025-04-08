"""
name = "全球大类资产趋势轮动"
symbols = ["159509.SZ", "518880.SH", "159531.SZ", "513100.SH", "513520.SH", "159857.SZ", "512100.SH", "513500.SH",
           "159915.SZ", "588000.SH", "159813.SZ", "162719.SZ", "513330.SH"]
benchmark = "510300.SH"
# [date]
start_date = "20230919"
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
"""

import wx
import wx.html2 as webview
import threading
import time
import wx.lib.newevent as ne
import wx.adv
# from backtrader_extends.engine import run_task
# from core.backtrader_extends.task import local_tasks
from dataclasses import dataclass
from backtrader import Strategy
from backtrader import Cerebro
from datetime import datetime
import pandas as pd
import os

from backtrader.feeds import PandasData


# 定义策略任务的元数据容器
@dataclass
class TaskDate:
    start_date: str  # 格式："YYYYMMDD"
    end_date: str


@dataclass
class BacktestTask:
    # strategy_class: Strategy  # 策略类（需继承自Backtrader的Strategy）
    params: dict  # 策略参数
    date: TaskDate  # 回测时间范围
    data_path: str  # 数据文件路径（CSV等）
    benchmark: str = ""  # 基准指标（可选）


# 预定义的策略任务集合（示例）
class MovingAverageCrossStrategy(Strategy):
    pass


class MomentumStrategy(Strategy):
    pass


local_tasks = {
    "双均线策略": BacktestTask(
        # strategy_class=MovingAverageCrossStrategy,  # 需自定义策略类
        params=dict(fast_period=10, slow_period=20),
        date=TaskDate(start_date="20100101", end_date="20221231"),
        data_path="data/沪深300.csv"
    ),
    "动量策略": BacktestTask(
        # strategy_class=MomentumStrategy,
        params=dict(lookback_period=30, threshold=0.05),
        date=TaskDate(start_date="20150101", end_date="20231231"),
        data_path="data/中证500.csv"
    )
}


def run_task(task, output_dir="results"):
    """执行回测任务并生成结果"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化回测引擎
    cerebro = Cerebro()

    # 添加数据
    data = load_data(task.data_path, task.date)
    cerebro.adddata(data)

    # 添加策略
    cerebro.addstrategy(task.strategy_class, **task.params)

    # 设置初始资金和手续费
    cerebro.broker.setcash(1000000)
    cerebro.broker.setcommission(commission=0.001)

    # 运行回测
    results = cerebro.run()

    # 生成结果文件
    generate_html_report(
        strategy_name=task.strategy_class.__name__,
        start_date=task.date.start_date,
        end_date=task.date.end_date,
        portfolio_value=cerebro.broker.getvalue(),
        output_dir=output_dir
    )


def load_data(data_path, task_date):
    """加载数据（示例实现，需适配实际数据格式）"""
    df = pd.read_csv(data_path,
                     parse_dates=['date'],
                     index_col='date')
    df = df[(df.index >= pd.to_datetime(task_date.start_date)) &
            (df.index <= pd.to_datetime(task_date.end_date))]
    # return PandasData(dataname=df)
    return df


def generate_html_report(strategy_name, start_date, end_date, portfolio_value, output_dir):
    """生成HTML报告（需完善为实际分析逻辑）"""
    html_content = f"""
    <html>
    <body>
        <h1>{strategy_name} 回测结果</h1>
        <p>日期范围: {start_date} 至 {end_date}</p>
        <p>最终资产: {portfolio_value:.2f}</p>
        <!-- 此处应插入Pyecharts图表 -->
    </body>
    </html>
    """
    with open(f"{output_dir}/{strategy_name}.html", "w") as f:
        f.write(html_content)


# 自定义事件，用于线程与主线程通信
BacktestProgressEvent, EVT_BACKTEST_PROGRESS = ne.NewEvent()
BacktestCompleteEvent, EVT_BACKTEST_COMPLETE = ne.NewEvent()


class BacktestThread(threading.Thread):
    """回测线程"""

    def __init__(self, panel, strategy, start_date, end_date, benchmark):
        super().__init__()
        self.panel = panel
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self._stop_event = threading.Event()

    def run(self):
        """执行回测逻辑"""
        try:
            run_task(local_tasks[self.strategy])  # 模拟回测过程 - 实际使用时替换为真实回测逻辑
            for i in range(1, 11):
                if self._stop_event.is_set():
                    return  # 发送进度更新事件
                wx.PostEvent(self.panel, BacktestProgressEvent(progress=i * 10))
                time.sleep(0.5)
            # 模拟生成结果文件
            html_content = f"""                  
            <html>                  
            <head><title>{self.strategy}回测结果</title></head>                  
            <body>                        
            <h1>{self.strategy}回测结果</h1>                        
            <p>日期范围:  {self.start_date}  至  {self.end_date}</p>                        
            <p>基准指标:  {self.benchmark}</p>                        
            <div style="width:600px;height:400px;background:#f0f0f0;display:flex;justify-content:center;align-items:center;">                              
                <p>这里是pyecharts生成的图表区域</p>                        
            </div>                        
            <p>生成时间:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>                  
            </body>                  
            </html>                  
            """
            # 保存HTML文件
            html_file = f"{self.strategy}.html"
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            # 发送完成事件
            wx.PostEvent(self.panel, BacktestCompleteEvent(strategy=self.strategy))
        except Exception as e:
            wx.CallAfter(wx.MessageBox, f"回测出错:  {str(e)}", "错误", wx.OK | wx.ICON_ERROR)

    def stop(self):
        """停止回测"""
        self._stop_event.set()


class BacktestPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        # 当前回测线程
        self.backtest_thread = None

        # 主布局
        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # 左侧面板 - 回测条件
        self.left_panel = wx.Panel(self)
        left_sizer = wx.BoxSizer(wx.VERTICAL)

        # 策略选择器
        strategy_box = wx.StaticBox(self.left_panel, label="策略选择")
        strategy_sizer = wx.StaticBoxSizer(strategy_box, wx.VERTICAL)
        self.strategy_choice = wx.Choice(strategy_box, choices=list(local_tasks.keys()))
        # 默认选择第一项（索引为0）
        # 绑定策略选择事件
        self.strategy_choice.Bind(wx.EVT_CHOICE, self.on_strategy_changed)
        strategy_sizer.Add(self.strategy_choice, 0, wx.EXPAND | wx.ALL, 5)

        # 策略操作按钮
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.open_strategy_btn = wx.Button(strategy_box, label="打开策略")
        self.open_strategy_btn.Bind(wx.EVT_BUTTON, self.on_open_strategy)
        btn_sizer.Add(self.open_strategy_btn, 1, wx.EXPAND | wx.RIGHT, 5)

        self.edit_strategy_btn = wx.Button(strategy_box, label="编辑策略")
        btn_sizer.Add(self.edit_strategy_btn, 1, wx.EXPAND)
        strategy_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 日期选择
        date_box = wx.StaticBox(self.left_panel, label="回测日期范围")
        date_sizer = wx.StaticBoxSizer(date_box, wx.VERTICAL)

        start_date_sizer = wx.BoxSizer(wx.HORIZONTAL)
        start_date_sizer.Add(wx.StaticText(date_box, label="开始日期:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.start_date = wx.adv.DatePickerCtrl(date_box, style=wx.adv.DP_DROPDOWN | wx.adv.DP_SHOWCENTURY)
        start_date_sizer.Add(self.start_date, 1, wx.EXPAND)
        date_sizer.Add(start_date_sizer, 0, wx.EXPAND | wx.ALL, 5)

        end_date_sizer = wx.BoxSizer(wx.HORIZONTAL)
        end_date_sizer.Add(wx.StaticText(date_box, label="结束日期:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.end_date = wx.adv.DatePickerCtrl(date_box, style=wx.adv.DP_DROPDOWN | wx.adv.DP_SHOWCENTURY)
        end_date_sizer.Add(self.end_date, 1, wx.EXPAND)
        date_sizer.Add(end_date_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Benchmark选择
        benchmark_box = wx.StaticBox(self.left_panel, label="基准指标")
        benchmark_sizer = wx.StaticBoxSizer(benchmark_box, wx.VERTICAL)
        self.benchmark_choice = wx.Choice(benchmark_box, choices=["沪深300", "中证500", "上证指数", "创业板指"])
        benchmark_sizer.Add(self.benchmark_choice, 0, wx.EXPAND | wx.ALL, 5)

        # 回测进度条
        self.progress = wx.Gauge(self.left_panel, range=100)

        # 启动回测按钮
        self.start_btn = wx.Button(self.left_panel, label="启动回测")
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start_backtest)

        # 停止回测按钮
        self.stop_btn = wx.Button(self.left_panel, label="停止回测")
        self.stop_btn.Bind(wx.EVT_BUTTON, self.on_stop_backtest)
        self.stop_btn.Disable()

        # 按钮布局
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(self.start_btn, 1, wx.EXPAND | wx.RIGHT, 5)
        btn_sizer.Add(self.stop_btn, 1, wx.EXPAND)

        # 添加左侧组件
        left_sizer.Add(strategy_sizer, 0, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(date_sizer, 0, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(benchmark_sizer, 0, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(self.progress, 0, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.left_panel.SetSizer(left_sizer)

        # 右侧面板 - WebView显示结果
        self.right_panel = wx.Panel(self)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        # 创建WebView
        self.webview = webview.WebView.New(self.right_panel)
        right_sizer.Add(self.webview, 1, wx.EXPAND)
        self.right_panel.SetSizer(right_sizer)

        # 设置主布局比例
        self.main_sizer.Add(self.left_panel, 1, wx.EXPAND)
        self.main_sizer.Add(self.right_panel, 3, wx.EXPAND)
        self.SetSizer(self.main_sizer)
        # 获取当前日期
        today = datetime.today()  # 设置开始日期（1年前）
        start_date = wx.DateTime()
        start_date.Set(day=today.day,
                       month=today.month - 1,  # wx.DateTime 月份是 0-11
                       year=today.year - 1)  # 设置结束日期（当前日期）
        end_date = wx.DateTime()
        end_date.Set(day=today.day,
                     month=today.month - 1,  # wx.DateTime 月份是 0-11
                     year=today.year)

        # 应用到控件
        self.start_date.SetValue(start_date)
        self.end_date.SetValue(end_date)
        # 绑定自定义事件
        self.Bind(EVT_BACKTEST_PROGRESS, self.on_backtest_progress)
        self.Bind(EVT_BACKTEST_COMPLETE, self.on_backtest_complete)

        # 默认加载策略的HTML
        self.load_default_html()
        if self.strategy_choice.GetCount() > 0:  # 确保有选项
            self.strategy_choice.SetSelection(0)
            self.update_dates_from_task(self.strategy_choice.GetStringSelection())

    def on_strategy_changed(self, event):
        """当策略选择变化时更新日期"""
        selected_strategy = self.strategy_choice.GetStringSelection()
        self.update_dates_from_task(selected_strategy)
        event.Skip()

    def update_dates_from_task(self, strategy_name):
        """根据策略名称更新日期控件"""

        if strategy_name in local_tasks.keys():
            task = local_tasks[strategy_name]  # 解析开始日期
            start_date_str = task.date.start_date  # 格式: "20100101"
            start_date = datetime.strptime(start_date_str, "%Y%m%d")
            wx_start = wx.DateTime()
            wx_start.Set(day=start_date.day,
                         month=start_date.month - 1,  # wx月份是0-11
                         year=start_date.year)
            self.start_date.SetValue(wx_start)

            # 解析结束日期
            end_date_str = task.date.end_date  # 格式: "20201231"
            if end_date_str == '':
                end_date_str = datetime.now().strftime('%Y%m%d')
            end_date = datetime.strptime(end_date_str, "%Y%m%d")
            wx_end = wx.DateTime()
            wx_end.Set(day=end_date.day,
                       month=end_date.month - 1,  # wx月份是0-11
                       year=end_date.year)
            self.end_date.SetValue(wx_end)

    def load_default_html(self):
        """加载默认的策略HTML文件"""
        strategy_name = self.strategy_choice.GetStringSelection()
        if not strategy_name:
            strategy_name = self.strategy_choice.GetStrings()[0]
        html_file = f"{strategy_name}.html"
        if os.path.exists(html_file):
            self.webview.LoadURL(f"file://{os.path.abspath(html_file)}")
        else:  # 如果文件不存在，显示空白页面或默认页面
            self.webview.SetPage("<html><body><h1>回测结果将显示在这里</h1></body></html>", "")

    def on_open_strategy(self, event):
        """打开策略按钮事件处理"""
        strategy = self.strategy_choice.GetStringSelection()
        if not strategy:
            wx.MessageBox("请先选择一个策略", "提示", wx.OK | wx.ICON_INFORMATION)
            return

        # 这里应该是打开策略文件的逻辑
        # 例如：打开策略代码文件或策略配置对话框
        wx.MessageBox(f"打开策略:  {strategy}", "提示", wx.OK | wx.ICON_INFORMATION)

    def on_start_backtest(self, event):
        """启动回测按钮事件处理"""
        if self.backtest_thread and self.backtest_thread.is_alive():
            wx.MessageBox("已有回测正在运行", "提示", wx.OK | wx.ICON_INFORMATION)
            return

        # 获取选择的参数
        strategy = self.strategy_choice.GetStringSelection()
        if not strategy:
            wx.MessageBox("请选择一个策略", "错误", wx.OK | wx.ICON_ERROR)
            return

        start_date = self.start_date.GetValue().FormatISODate()
        end_date = self.end_date.GetValue().FormatISODate()
        benchmark = self.benchmark_choice.GetStringSelection()

        # 重置进度条
        self.progress.SetValue(0)
        self.start_btn.Disable()
        self.stop_btn.Enable()

        # 创建并启动回测线程
        self.backtest_thread = BacktestThread(self, strategy, start_date, end_date, benchmark)
        self.backtest_thread.start()

    def on_stop_backtest(self, event):
        """停止回测按钮事件处理"""
        if self.backtest_thread and self.backtest_thread.is_alive():
            self.backtest_thread.stop()
            self.stop_btn.Disable()
            wx.MessageBox("回测已停止", "提示", wx.OK | wx.ICON_INFORMATION)

    def on_backtest_progress(self, event):
        """回测进度更新事件处理"""
        self.progress.SetValue(event.progress)

    def on_backtest_complete(self, event):
        """回测完成事件处理"""
        self.progress.SetValue(100)
        self.start_btn.Enable()
        self.stop_btn.Disable()

        # 加载回测结果
        html_file = f"{event.strategy}.html"
        if os.path.exists(html_file):
            self.webview.LoadURL(f"file://{os.path.abspath(html_file)}")
        else:
            wx.MessageBox(f"回测完成，但未找到结果文件:  {html_file}", "提示", wx.OK | wx.ICON_INFORMATION)


# 使用示例
if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(None, title="量化回测工具", size=(1000, 600))
    panel = BacktestPanel(frame)
    frame.Show()
    app.MainLoop()
