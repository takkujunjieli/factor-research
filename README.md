# factor-research(独立研究环境,不属于 dashboard)

放"重"的东西:全市场横截面 / panel / 宏观数据的研究、因子验证、调参。用**公开数据**,
**不落 dashboard**。dashboard 保持 slim(只在 watchlist 上算选定因子)。

## macro_regime.py — FRED 宏观状态 / regime

拉一组利率/信用/波动/金融状况序列(FRED 无 key 公开 CSV),周频,**point-in-time**
滚动 z-score(无前视),合成 stress 复合 → risk-on / neutral / risk-off 三态 + 离散 flag。

**序列**:10Y/2Y 收益率、期限利差(T10Y2Y)、高收益/投资级 OAS(信用压力)、VIX、
NFCI(芝加哥联储金融状况)、STLFSI4(金融压力)、10Y 盈亏平衡通胀。

**跑**:
```bash
pip install -r requirements.txt
python3 macro_regime.py                 # → macro_regimes.csv / .parquet
python3 macro_regime.py --markov         # 额外出 2 态 Markov 平滑概率(需 statsmodels)
```

**输出** `macro_regimes.csv`:周频,列含原序列、`z_*`、`stress`、`regime`、
`curve_inverted`/`vix_elevated`/`credit_widening`/`nfci_tight`。

## 怎么用来做条件化 / regime 切换

把它按日期 join 到你(dashboard 那边)watchlist 的因子/信号 panel:
```python
import pandas as pd
mac = pd.read_csv("macro_regimes.csv", index_col=0, parse_dates=True)
sig = sig.merge(mac[["regime", "stress"]], left_index=True, right_index=True, how="left")
# 用 asof 对齐到最近一个已知周(避免用未来的宏观):
# sig = pd.merge_asof(sig.sort_index(), mac[["regime","stress"]], left_index=True, right_index=True)
```
然后:
- **条件化**:分 regime 统计每个因子的 IC / 收益(哪些因子只在 risk-off 有效)。
- **regime 切换**:按 `regime` 或 `stress` 调因子权重 / 敞口 / 多空 tilt。
- **回测**:接 dashboard 的 `strategy_*.py`,把 regime 作为状态变量。

## 要点
- **无前视**:z-score 用 trailing 窗口(默认 156 周);regime[t] 只用 ≤t 的数据。
- **发布时滞**:严格回测请按发布延迟 lag 月频序列(NFCI ~1 周、CPI ~2 周);利率/VIX 当日可得。
- 阈值(±0.5)、z 窗口都可调(`--zwin`),先看 regime 分布是否合理再定。
- 这里只放**公开数据 + 因子定义/参数**;私有的持仓/盈亏永远不进这个仓库。

## build_panel.py — 日频特征面板(GD/CNN 的直接输入)

拼 French 日频因子(FF5+Mom)+ 12 行业组合日收益 + `macro_regime`(FRED)→ 业务日对齐 →
加 forward-N-day 目标。产出 `panel_daily.parquet`。

```bash
python3 build_panel.py     # → panel_daily.parquet
```
- **特征**:FF 因子(7)+ 行业日收益(12,`ind_*`)+ 宏观(`z_*`/`stress`/`regime`/`regime_code`/flags)
- **目标**:`y_mkt_fwd{5,20}` + `y_mkt_dir{5,20}`;每行业 `y_ind_*_fwd20` + `_dir20`
- **⚠️ French ~1 个月发布时滞**:面板止于上月末,做研究/调参无碍;要实时信号需另补最近段。
- 频率取舍:日频是最细基准,可按需 `.resample()` 降到周/月;目标别用"下一日方向"(≈噪声),用 5/20 日。

## bear_panel.py — 熊市预警研究面板(月频,~1970 起)

专为「找排名靠前的熊市预警信号」。目标=**未来最差**(forward 6/12 月最差累计收益),自动检测
严重回撤段(总收益 ≥15% ≈ 传统 20% 价格熊市 + 准熊级),给 **leave-one-bear-out** 的 episode 标注。

```bash
python3 bear_panel.py     # → bear_panel.parquet(~666 月 × 22 特征)
```
**特征**:市场趋势/回撤/动量/已实现波动、防御-周期轮动、49行业广度、曲线(10Y-3M 长短/10Y-2Y)、
信用(Moody's BAA-AAA 长 + HY OAS 短)、VIX、NFCI、失业/Sahm/CFNAI/初请/CPI同比、**EBP + Fed 衰退概率 est_prob**。
**目标/标注**:`y_fwd_minret_6m/12m`(连续,主目标)、`y_bear12`(未来12月是否有熊顶)、
`next_bear_id`(warning 窗,供 LOBO 留出)、`in_bear_id`。
**检测到 11 段**严重回撤(1966/1968/1973/1980/1987/1990/1998/2000/2007/2020/2022)。

**要点**:
- **评估用 leave-one-bear-out**(按 `next_bear_id`/`in_bear_id` 留出整段),不要随机 CV——稀有事件+自相关下随机 CV 严重高估。
- PIT:真宏观(失业/CFNAI/Sahm/CPI/EBP)已 shift 1 月;市场可观测(曲线/VIX/利差/NFCI)不 shift。
- 阈值 0.15(总收益口径)、检测/窗口可调;想更多事件降到 0.12(纳入 2011/2018)。
- 主目标用**连续** `y_fwd_minret`(不依赖二元阈值),`y_bear12` 仅作分类对照。

## 稳健轻量预警模型(factorlab/model.py + run_bear_model.py)

不用 NN(样本少必过拟合):**L2 正则 logistic**(numpy IRLS 自实现),标准化用训练折统计(无前视),
**leave-one-bear-out** 评估,口径与单信号排名一致。

```bash
python3 run_bear_model.py            # λ=10;输出 LOBO AUC / 系数 / 当前预警概率
python3 run_bear_model.py --lam 20   # 更强正则
```
- 组合 vs 最佳单信号:组合 mean 略低但 **min/hit 更高**(9/9 全押对)——用一点峰值换**跨熊市稳健**。
- 系数标准化可读(+推高预警);⚠️ `mkt_dd` 等价格同步项权重大 → 模型是领先+确认混合,要纯领先就剔掉价格类。
- 产物:`out/bear_model_prob.csv`(预警概率序列)、`out/bear_model_coef.csv`。
- 只有 ~9 次样本外熊市:先看 min/hit 稳不稳,再谈 mean;深网禁用。
