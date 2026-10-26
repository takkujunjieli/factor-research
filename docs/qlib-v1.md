# Qlib 第一版：日频选股与预测账本

## 独立历史行情观察

`run_market_observations.py` 根据当前保存的股票池与门槛，输出过去5 / 20 / 60个交易日的实际动量和年化波动率。不导入模型、不调用 `run_qlib.py`、不改写研究缓存或预测账本。

```bash
.venv-qlib/bin/python run_market_observations.py --fetch yahoo
```

有凭据时可用 `--fetch alpaca`；不指定来源则读取缓存。输出到相邻 Dashboard 的 `data/market_observations.json`，独立行情缓存位于 `data/market-observations/`。动量是区间价格收益；波动率是区间日简单收益的样本标准差乘 `sqrt(252)`。仅使用完整交易日，不含分红，不插值缺价。

## 当前股票池配置（2026-10-25）

本地 Qlib 页新增「股票池配置」：可选择细分行业 / 主题、开启或关闭原始103只基础名单、手动 Include / Exclude，并保存多个命名股票池。默认名单加入 PURR。29个分类是人工可扩展的研究目录，不是全市场完整行业成分库；目录文件为 `config/qlib-industry-catalog.json`。

设置保存在 `config/qlib-universes.json`。筛选顺序：基础名单与已选分类、Include 取并集 → Exclude → 当前股价 / 平均成交额 / 市值 / 历史长度过滤。Include 仍须满足门槛。页面只展示筛选后的股票，不提供逐票排除原因。

门槛只应用一次当前快照；历史训练和 IC / ICIR 都在当前筛出的股票池上进行，不按历史日期重新应用市值、股价、流动性门槛。特征计算所需的数据完整性仍检查。选择变化、当前入选名单或门槛变化会生成新的股票池 ID，避免混合不同范围的前向 IC。

市值支持上下限（页面以亿美元输入，后端以美元保存）；0下限及空白上限表示关闭市值门槛。启用时通过 yfinance 获取最新市值，并将当天缓存保存在研究目录；它不用于历史市值回放。成交额可设置平均窗口及最低值。最短历史长度至少61个交易日，以满足现有量价特征。

保存配置后点击「刷新研究」，后台下载新范围的数据并重算。缩小到少量股票仍可生成单票分数与时间序列 IC；横截面不足3只时 IC 留空。原有自定义 `--config` 研究 / 测试文件保持自己的股票池，仅默认应用配置读取页面设置。

本轮预览服务使用 `http://127.0.0.1:8643/strategy.html`，保留原有8642服务。任何已运行的旧服务需重启后才能加载新增配置 API。

这是可以在本机运行的信号研究闭环：公开历史量价 → 特征 → Qlib Ridge → 滚动样本外分数 → 选股 / 择时 IC 与 ICIR → 前向账本。组合方式、权重与风险分配由现有 Kelly / Risk Allocation 处理。

只使用公开行情，不读取私人持仓，不连接券商执行，不将预测写入 Kelly。当前模型尚未验证为可交易策略。

## 已实现

- 可配置的当前美股研究股票池，另取 SPY 作数据完整性参照；5 / 20 个交易日两个期限。
- Alpaca SIP 日线采集、分页、限流重试；显式的 Yahoo 免费探索替代源；统一拆股调整、价格收益口径，不含分红。
- 每次完整刷新同一复权口径，不混接不同时间版本。重复、坏 OHLC、周末、未收盘日均检查；每票缺失记录可见。
- 9 个因果量价特征，实际调用 Qlib `StaticDataLoader`、`DataHandlerLP`、`DatasetH`、`LinearModel(estimator="ridge")`。
- 最少 252 个可训练交易日、最多 504 日训练窗；每 20 个交易日重新训练，期间每日重新推理。训练标准化只用训练集。
- 每个训练标签的最晚结果日期严格早于测试首日，剔除跨界标签。没有随机切分、全样本选参或收益导向调参。
- 选股：每天在股票横截面计算 Pearson IC 和 Spearman Rank IC，再对每日 IC 序列计算 ICIR。
- 择时：每只股票使用互不重叠的63个信号交易日窗口计算 IC，每窗口至少20个有效观测；对窗口 IC 序列计算 ICIR。用于诊断同票跨时间预测力，尚未定义机械交易规则。
- 标签为收盘信号后的 t+1 开盘至 t+h 收盘价格收益，只有定义预测目标的含义；没有模拟入场、退出、持仓、成本或净值。
- 全候选前向预测进入 SQLite。预测、结算均禁止更新或删除，重复运行不会覆盖原记录。
- 实际生成时间必须早于下一参考日开盘，才能登记前向预测；过期缓存、历史 OOS 和 demo 不会回填成前向实绩。
- 原始 OOS 分数、训练/测试边界、标准化参数、系数、数据/代码指纹及运行依赖版本保存在本地实验目录。
- Dashboard `strategy.html`（Qlib）：选股 / 择时 IC 与 ICIR、候选排名、fold 边界、数据覆盖和前向 IC 账本。默认预览前10行只控制显示数量，不是 Top-K 组合。

## 首次安装

在 `factor-research` 目录，使用 Python 3.12 的独立环境：

```bash
python3.12 -m venv .venv-qlib
.venv-qlib/bin/python -m pip install -r requirements-qlib.txt
```

`requirements-qlib.lock.txt` 记录首版本机验证过的完整依赖。常规安装可用简短 requirements；需要相同版本重现时使用 lock 文件。该 lock 采集自 macOS arm64，不承诺其他平台所有包均有二进制发行。

## 刷新真实研究

首选现有 Alpaca 凭据。脚本只从进程环境变量读取，不把凭据写到文件或网页：

```bash
# 在 shell / 密钥管理器中预先配置 ALPACA_API_KEY、ALPACA_API_SECRET。
# 也支持 APCA_API_KEY_ID、APCA_API_SECRET_KEY。
.venv-qlib/bin/python run_qlib.py --fetch alpaca
```

未配置凭据时，可先用免费 Yahoo 跑探索版本：

```bash
.venv-qlib/bin/python run_qlib.py --fetch yahoo
```

免费数据是探索材料；Yahoo 与 SIP 的日线聚合、缺失及成交量可能不同，切换数据源后必须重新评估。脚本不会在 Alpaca 请求失败时静默混入 Yahoo。

只复用已下载的快照重新计算：

```bash
.venv-qlib/bin/python run_qlib.py
```

支持 `--prices path.csv`。CSV 列为 `date,symbol,open,high,low,close,volume`，日期是美东交易日，必须采用与 Alpaca `adjustment=split` 相同的拆股调整口径。`--asof YYYY-MM-DD` 用于历史回放；`--demo` 明确标记合成演示，并使用独立账本。

默认输出到相邻的 `noetic-dashboard/data/qlib_research.json`；其他目录通过 `--output` 指定。

## 页面使用

在 `noetic-dashboard` 用既有服务运行：

```bash
.venv/bin/python scripts/serve_dashboard.py --port 8642
```

访问 `http://127.0.0.1:8642/strategy.html`，各页导航的 Qlib 直接进入该页。旧 `qlib.html` 自动跳转。

本地页面有「刷新研究」按钮。后台仅执行固定命令，不接受自定义路径或 shell。服务环境有 Alpaca 凭据时选 Alpaca，否则选 Yahoo。任务在后台运行，页面显示状态，成功后重新读取 JSON；失败时保留上次有效结果。日志在 `factor-research/data/qlib-v1/refresh.log`。现有服务需重启一次才能加载新增 API。

本版本提供本地按需刷新，尚未开启定时调度或发布到远端。后续每天盘后运行同一命令即可持续累计。未来自动化必须复用持久账本，不能使用会随 CI runner 消失的临时账本。

## 数据与备份

`data/qlib-v1/` 默认 gitignore，包含：

| 路径 | 用途 |
|---|---|
| `prices.csv` / `prices.meta.json` | 最近一次统一复权快照、来源及校验和 |
| `forecast-ledger.sqlite` | 全部真实前向预测和结算，核心持久资产 |
| `experiments/<fingerprint>/manifest.json` | 完整配置、版本与数据/代码指纹 |
| `experiments/<fingerprint>/model.json` | 训练范围、标准化参数、系数和模型 ID |
| `experiments/<fingerprint>/oos_predictions.csv` | 历史样本外分数，独立于前向账本 |
| `experiments/<fingerprint>/evaluation.json` | 逐日横截面 IC、单票窗口 IC、ICIR 及训练边界 |

**备份整个 `data/qlib-v1/`，尤其 SQLite 账本。** 页面只导出最近部分账本和汇总，并不是完整备份。修改代码或配置会创建新实验/模型版本，同日新版本也会保留；同一日期多个版本不是新的独立市场样本。禁止手工删除旧账本以重置成绩。

结算使用同一份最新复权快照中的历史入场价与退出价，避免拆股前后价格混算。第一次结算即冻结，并保留该快照的哈希。遇到缺价会显示缺失，不将其记为零收益或从成绩中悄悄删除。数据哈希可以校验来源版本，但如要独立重建旧数据，需额外归档相应完整原始快照。

## 如何解释结果

- 分数是回归的价格收益预测，不是上涨概率，也不是因果解释。
- IC 是 Pearson 相关；Rank IC 是 Spearman 相关（并列值使用平均秩）。ICIR / Rank ICIR = 相应 IC 序列均值除以样本标准差（ddof=1），均不年化。
- 横截面按日等权汇总；择时按单票窗口汇总，不能将所有股票/日期混在一起算相关性，也不能把股票间离散度当作时间 ICIR。
- 未到期、缺价、股票数不足或常量分数/收益使该期 IC 不可定义，显示为空值；零标准差或不足两期的 ICIR 也为空值。均不伪造为零。
- 未来收益标签相互重叠，同日股票相关，ICIR 仅为描述性稳定性，不是 t 值或独立样本显著性。63日窗口不重叠，也不保证窗口边界附近的未来收益标签独立。
- 前向评分只选择每个股票池、每个信号日参考区间开始前登记的最新模型版本；旧版保留审计，不叠加为新样本。不同股票池分别评估。
- 当前筛选股票池用于历史评估，包含幸存者及事后选池偏差，尚未覆盖历史退市收益。历史“上市后缺失”也可能来自停牌、代码/证券变更，不能一律当抓取故障。
- 单票时间序列 IC 只表示择时预测力，不能据此声称已有可执行择时策略。

## 第一轮真实数据验证

2026-10-24 ET 使用 Yahoo 探索数据，行情截至 2026-10-23，104 个证券（103 股票 + SPY）均覆盖最新交易日；SPY 仅保留为行情完整性参照，不参与组合评估。首轮 5日 / 20日平均横截面 Rank IC 均接近零。未据此调参。

下一阶段优先做 SIP 同口径复核、历史股票池与退市覆盖，再研究新的选股 / 择时信号。旧组合实验档案保留在本地，新流程不再调用或展示组合回测。

## 验证

```bash
.venv-qlib/bin/python -m unittest discover -s tests -v
```

覆盖未来数据扰动、标签期限、缺价、假日/未收盘过滤、训练标签跨界、IC / Rank IC 区分、ICIR 样本标准差、不可定义指标、横截面与时间序列区分、不可变账本、同日幂等、真实 Qlib 调用及端到端导出。合成数据仅用于这些验证，不进入真实 Dashboard。
