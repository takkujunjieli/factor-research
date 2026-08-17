# factor-research

全市场横截面/panel 因子研究 —— **与 stock-dashboard 完全隔离**。重活(全市场数据、调参、宏观)放这里;
dashboard 只在 watchlist 上实现选定因子,保持 slim。数据文件不进 git(见 `.gitignore`),仓库只留代码。

## 第一条:OpenAP 信号 → IC 衰减

```bash
python3 run_ic.py                      # 合成数据 demo(离线可跑,验证引擎)
```
合成数据的信号被设计成 IC 按几何衰减(IC_h ∝ 0.6^(h-1)),用来确认引擎算得对。

### 接真实 OpenAP 数据
1. **信号(免费)**:openassetpricing.com → 下 "Firm Level Characteristics" 宽表
   `signed_predictors_dl_wide.csv`(permno, yyyymm, ~210 信号列),放 `data/`。
2. **月度个股收益(要你补的一块)**:`[permno, yyyymm, ret]` 放 `data/`。
   - 有 WRDS/院校权限 → **CRSP 月度收益**(最佳,含退市收益、无幸存者偏差)。
   - 无 → 免费近似:按 ticker 映射 Stooq/yfinance 月度收益(有幸存者偏差、无退市收益,仅探索)。
3. 跑:
   ```bash
   python3 run_ic.py --source openap \
       --signals data/signed_predictors_dl_wide.csv \
       --returns data/crsp_monthly_ret.csv \
       --signal Mom12m --max-horizon 18
   ```

输出:`out/ic_decay_<signal>.csv`(每个 horizon 的 mean_IC / IC_IR / t_stat / pos_rate)+ `out/ic_decay_<signal>.svg`(衰减曲线)+ 控制台的 IC 半衰期。

## 指标口径
- **rank IC_h**:每月 signal_t 与 h 期后单期收益 ret_{t+h} 的 Spearman 秩相关,再对月份求均值。
- **IC_IR** = mean/std(单月);**t_stat** = mean/std·√n_months;**pos_rate** = IC>0 的月份占比。
- IC_h 随 h 衰减越慢 = 信号越持久;半衰期短 = 需高频再平衡。

## 结构
```
factorlab/loaders.py   OpenAP/收益加载 + 合成数据
factorlab/ic.py        IC 与 IC 衰减(键 join 前视收益,pandas 秩相关,无 scipy)
factorlab/plot.py      零依赖 SVG 出图
run_ic.py              CLI
data/  out/            gitignored(数据/产物不进仓库)
```

## 依赖
numpy · pandas · pyarrow(读 parquet)。出图是手写 SVG,不需要 matplotlib。
