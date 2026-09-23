# RAVEN Tushare 复现脚本模块讲解

对应脚本：raven_tushare_reproduction.py

这份说明按照“数据来源 → 数据清洗 → 因子构造 → 样本构造 → RAVEN 模型 → 训练 → 评估 → 与论文的差异 → 如何扩展到 Level2”的顺序解释。

---

## 1. 这份脚本复现的是什么

论文 RAVEN 的核心不是某一组固定因子，而是一个动态时间上下文模型：

~~~text
最大历史窗口
    ↓
Patch 切分
    ↓
学习每个 Patch 的重要性
    ↓
CIT 累计重要性阈值路由
    ↓
短期 / 中期 / 长期专家
    ↓
CAW 相关性感知融合
    ↓
GCR 全局历史分支
    ↓
预测未来累计收益
~~~

论文主要给出的是 OHLCV、价差和工程化因子这类输入描述，没有公开一份唯一、完整的私有因子清单。因此，脚本使用 Tushare 的日线 OHLCV 构造一套透明、可审计的因子。这样可以完整复现“数据到模型”的方法流程，但不能声称恢复作者没有公开的内部特征工程。

默认实验设置尽量贴近论文：

| 项目 | 默认值 |
|---|---:|
| 股票池 | HS300 |
| 数据频率 | 日频 |
| 数据区间 | 2023-01-01 至 2026-12-31（2026 年为当前可用数据的部分年度） |
| 训练集 | 2023-01-01 至 2024-12-31 |
| 验证集 | 2025-01-01 至 2025-12-31 |
| 测试集 | 2026-01-01 至 2026-12-31 |
| 最大历史窗口 | 120 个交易日 |
| Patch 长度 | 16 个交易日 |
| 专家数 | 3 |
| CIT 阈值 | 0.3、0.6、0.9 |
| 标签 | 未来 10 个交易日累计对数收益 |
| 训练损失 | MSE + 熵正则 + 专家多样性正则 |

---

## 2. 文件最开始需要改什么

脚本不再把真实 Token 写进 Python 文件，而是读取本地环境变量：

~~~bash
export TUSHARE_TOKEN="你的真实Tushare Token"
~~~

如果你使用 Windows PowerShell：

~~~powershell
$env:TUSHARE_TOKEN="你的真实Tushare Token"
~~~

脚本还允许直接在 Config 中修改：

~~~python
index_code = "000300.SH"
start_date = "20230101"
end_date = "20261231"
train_start = "20230101"
train_end = "20241231"
valid_start = "20250101"
valid_end = "20251231"
test_start = "20260101"
test_end = "20261231"
forecast_horizon = 10
max_lookback = 120
patch_len = 16
~~~

Token 只保存在你的本地终端环境中，不要提交到 GitHub。

---

## 3. 安装依赖

~~~bash
pip install tushare pandas numpy torch scikit-learn tqdm
~~~

脚本本身没有依赖 TA-Lib，RSI、波动率、Amihud 流动性等指标全部使用 pandas 和 numpy 计算。

如果使用 GPU，应根据自己的 CUDA 版本安装对应的 PyTorch。

---

## 4. 运行方式

### 4.1 先用 20 只股票做小样本测试

~~~bash
python raven_tushare_reproduction.py --mode all --max-stocks 20 --epochs 3
~~~

这个命令会：

1. 下载数据；
2. 清洗数据；
3. 构造因子；
4. 构造序列样本；
5. 训练 3 轮；
6. 输出测试集指标和简单 Top-K 回测。

### 4.2 只下载数据

~~~bash
python raven_tushare_reproduction.py --mode download
~~~

### 4.3 只清洗并构造因子

~~~bash
python raven_tushare_reproduction.py --mode prepare --max-stocks 20
~~~

### 4.4 正式训练

~~~bash
python raven_tushare_reproduction.py --mode all
~~~

或者在已有缓存数据的情况下：

~~~bash
python raven_tushare_reproduction.py --mode train
~~~

### 4.5 强制重新下载或重建因子

~~~bash
python raven_tushare_reproduction.py --mode all --force-download --force-rebuild-features
~~~

完整 HS300 数据下载和训练会比较耗时，建议先用 20 只股票验证整个流程，再扩大到完整股票池。

---

## 5. Tushare 数据下载模块

对应脚本第 2 部分：TUSHARE DOWNLOAD。

### 5.1 股票基础信息

函数：

~~~python
fetch_stock_basic(cfg, pro)
~~~

下载：

- 股票代码；
- 股票名称；
- 上市日期；
- 退市日期；
- 所属行业；
- 市场类型。

股票基础信息用于：

1. 排除 ST、*ST、退市整理股票；
2. 排除上市时间不足 180 天的股票；
3. 保留上市日期信息，检查样本是否具有足够历史。

### 5.2 HS300 成分股

函数：

~~~python
fetch_hs300_members(cfg, pro)
~~~

优先调用：

~~~python
pro.index_member(index_code="000300.SH")
~~~

如果账户有权限，脚本会使用 in_date 和 out_date 保留历史成分股信息，避免只使用当前 HS300 成分股造成幸存者偏差。

如果 index_member 没有权限，脚本会退回到 index_weight。这种回退只能得到一个成分股列表，不能完全恢复历史成分变化。脚本会给出警告，这种情况下不要把结果当作严格无幸存者偏差的论文复现。

### 5.3 日线数据

函数：

~~~python
fetch_one_daily(cfg, pro, ts_code)
~~~

每只股票下载：

- open；
- high；
- low；
- close；
- pre_close；
- change；
- pct_chg；
- vol；
- amount。

每只股票保存一个 CSV，同时全部数据保存到：

~~~text
raven_tushare_data/raw/daily/
raven_tushare_data/raw/daily_all.pkl
~~~

这样再次运行时会优先使用缓存，不会每次重新消耗 Tushare 调用次数。

### 5.4 可选 daily_basic

如果把：

~~~python
download_daily_basic = True
~~~

脚本还会下载：

- turnover_rate；
- turnover_rate_f；
- volume_ratio；
- pe；
- pb；
- ps；
- total_mv；
- circ_mv。

这些变量会合并到清洗后的数据中，作为可选特征。

默认关闭是因为 daily_basic 的调用权限和频率限制通常比 daily 更严格。如果只想先验证模型，不需要打开它。

---

## 6. 数据清洗模块

函数：

~~~python
clean_daily_data(daily, stock_basic, cfg)
~~~

主要清洗步骤如下。

### 6.1 日期和数值类型

把 Tushare 的 YYYYMMDD 字符串转换成 pandas 日期，并将价格、成交量、成交额转换为数值型。

### 6.2 去重

按照：

~~~text
ts_code + trade_date
~~~

去除重复记录。

### 6.3 价格合法性检查

保留：

~~~text
close > 0
high > 0
low > 0
high >= low
high >= close
low <= close
~~~

成交量和成交额缺失时按 0 处理，但价格缺失的记录直接排除。

### 6.4 ST 过滤

如果 exclude_st=True，股票名称中含有 ST 或 退 的样本会被剔除。

### 6.5 上市时间过滤

默认要求股票上市至少 180 天：

~~~python
min_listing_days = 180
~~~

这样可以减少新股上市初期极端波动、特征窗口不完整和样本不稳定带来的影响。

---

## 7. 因子构造模块

函数：

~~~python
construct_factors(cleaned, cfg)
~~~

论文没有给出唯一的原始因子清单，所以脚本采用透明的 OHLCV 因子。

所有因子都按股票单独计算，并且只使用当前时点及之前的信息。

### 7.1 收益率因子

首先计算：

~~~python
log_close = log(close)
ret_1 = log_close.diff()
~~~

然后构造：

~~~text
ret_1
ret_2
ret_5
ret_10
ret_20
~~~

这些变量表示不同时间跨度的对数收益。

### 7.2 动量因子

对窗口 w，脚本使用：

~~~python
mom_w = log_close.shift(1) - log_close.shift(w + 1)
~~~

这里故意使用 shift(1) 排除当前交易日，避免把当前收益直接混入动量因子。

构造：

~~~text
mom_5
mom_10
mom_20
mom_60
mom_120
~~~

### 7.3 反转因子

~~~text
reversal_1 = -ret_1
reversal_5 = -过去5日收益
~~~

它们用于表达短期均值回归。

### 7.4 波动率因子

对收益率计算滚动标准差：

~~~text
volatility_5
volatility_10
volatility_20
volatility_60
~~~

同时构造：

~~~text
downside_vol_20
~~~

只统计负收益的波动。

### 7.5 成交量和成交额因子

包括：

~~~text
volume_z_5
volume_z_10
volume_z_20
volume_z_60
amount_log
amount_change_5
~~~

其中 volume_z_w 是滚动 z-score。

### 7.6 日内价格结构因子

包括：

~~~text
intraday_range
gap_return
close_location
hl_range_20
price_ma20
price_ma60
~~~

它们分别表达：

- 当日高低价范围；
- 开盘相对前收的跳空；
- 收盘位于当日高低区间的位置；
- 20 日高低范围；
- 当前价格偏离 20 日、60 日均线的程度。

### 7.7 流动性和技术指标

包括：

~~~text
amihud_20
rsi14
~~~

Amihud 近似为：

~~~text
abs(return) / amount
~~~

它可以作为单位成交额价格冲击的粗略代理。

### 7.8 未来标签

标签在所有当前时点特征构造完成之后再生成：

~~~python
future_log_return = log_close.shift(-forecast_horizon) - log_close
~~~

当 forecast_horizon=10 时，表示从当前收盘到未来第 10 个交易日收盘的累计对数收益。

标签不能参与任何滚动均值、标准差或归一化参数的计算。

---

## 8. 样本构造和时间切分

函数：

~~~python
build_blocks(factors, factor_cols)
split_indices(blocks, cfg)
~~~

脚本不把所有序列一次性展开成巨大的三维数组，而是保存为每只股票一个 StockBlock：

~~~text
StockBlock
├── ts_code
├── dates
├── X: [时间长度, 因子数]
├── y: 未来收益标签
└── close
~~~

每个样本在 Dataset 中动态取出：

~~~text
X[t-lookback+1 : t+1]
~~~

因此单个输入形状是：

~~~text
[120, 因子数]
~~~

由于论文默认 120 // 16 = 7，脚本实际使用最近的 112 个交易日形成 7 个完整 Patch，并舍弃最老的 8 个不完整时间点。这是为了严格遵守论文中的：

~~~python
N = floor(Lmax / patch_len)
~~~

默认时间切分：

~~~text
训练：2023-01-01 至 2024-12-31
验证：2025-01-01 至 2025-12-31
测试：2026-01-01 至 2026-12-31
~~~

2026 年测试集只使用 Tushare 当前已经返回的交易日，因此截至当前日期属于部分年度样本外测试。模型根据验证集损失保存最佳状态，测试集只用于最终评估，不能用于调参。

由于标签是未来 10 个交易日累计收益，脚本会额外 purge 每个切分区间末尾的样本：如果样本的未来 10 日标签跨入下一个区间，就不把该样本放入当前区间，避免标签泄漏。

正式研究时，建议进一步使用 rolling / walk-forward 方式，而不是只做一次固定切分。

---

## 9. RAVEN 模型模块

模型类：

~~~python
class RAVEN(nn.Module)
~~~

### 9.1 输入形状

~~~text
x: [batch, time, channels]
~~~

例如：

~~~text
[512, 120, 35]
~~~

表示一个 batch 中有 512 个样本，每个样本有 120 个历史日、35 个因子。

### 9.2 实例归一化

模型内部对每个样本沿时间维度做：

~~~python
x_norm = (x - mean_time) / (std_time + 1e-5)
~~~

这对应论文中的 Instance Normalization，用于缓解金融时间序列的分布漂移。

它不是全市场截面标准化。若以后用于你的 Level2 因子，还要在模型外部按照严格历史信息做滚动标准化或截面标准化。

### 9.3 Patch Embedding

时间序列切成 7 个长度为 16 的 Patch：

~~~text
[120, D]
    ↓
[7, 16, D]
    ↓
[7, embed_dim]
~~~

脚本先对每个因子通道使用共享线性投影，再对通道表示进行平均池化，得到 Patch token。

Patch 顺序会被反转：

~~~text
Patch 1 = 最近的 Patch
Patch 2 = 更早的 Patch
...
~~~

这样后面的累计重要性天然从最近时间开始向过去累积。

### 9.4 Patch importance

每个 Patch token 进入一个两层 MLP：

~~~text
embed_dim → embed_dim/2 → 1
~~~

得到分数 s_i，然后做 Softmax：

~~~python
patch_prob = softmax(scores, dim=1)
~~~

因此同一个样本内所有 Patch 的重要性加总为 1。

### 9.5 CIT 路由

从最近 Patch 开始累加：

~~~python
cumulative = cumsum(patch_prob, dim=1)
~~~

对于阈值 0.3、0.6、0.9，分别找到累计重要性达到阈值所需的最短连续前缀。

例如某个样本的路由长度可能是：

~~~text
短期专家：1 个 Patch
中期专家：3 个 Patch
长期专家：6 个 Patch
~~~

另一个样本可能是：

~~~text
短期专家：2 个 Patch
中期专家：5 个 Patch
长期专家：7 个 Patch
~~~

这就是“同一个模型对不同样本动态选择历史长度”。

脚本使用按长度分组的方式运行专家：同一个 batch 内长度相同的样本放在一起计算，避免为了补齐长度而人为填充大量零值。

### 9.6 Temporal-scale experts

每个 CIT 窗口进入独立的 Transformer Encoder：

~~~text
Expert 1：短期窗口
Expert 2：中期窗口
Expert 3：长期窗口
~~~

每个专家默认 3 层 Encoder、8 个 Attention Head、128 维表示。

专家输出序列长度不同，所以脚本对每个专家做时间维度平均池化：

~~~python
z_k = h_k.mean(dim=1)
~~~

最终每个专家都得到一个固定的 128 维向量。

### 9.7 CAW 融合

由于长期窗口包含短期窗口，专家输出有结构化重叠。

脚本先计算专家表示之间的余弦相似度：

~~~python
cosine = normalize(z) @ normalize(z).T
~~~

再计算每个专家的冗余度：

~~~python
redundancy_k = sum(max(cosine_kj, 0))
~~~

最后通过：

~~~python
weight_k ∝ alpha_k * exp(-lambda * redundancy_k)
~~~

降低与其他专家高度相似的专家权重。

其中：

- alpha_k：一个可学习的原始专家置信度；
- lambda：通过 Softplus 保证为非负的可学习参数；
- redundancy_k：专家输出冗余程度。

### 9.8 GCR 全局分支

局部专家只看不同长度的窗口，因此脚本并行保留一个完整历史分支：

~~~python
global_hidden = global_encoder(E)
z_global = global_hidden.mean(dim=1)
~~~

最后将 z_local 和 z_global 拼接后送入 MLP 输出预测结果。

### 9.9 输出

输出是一个标量：

~~~text
未来 10 个交易日累计对数收益率预测值
~~~

如果以后改成多周期预测，可以把输出层改成多个 horizon 的向量。

---

## 10. 损失函数

代码函数：

~~~python
raven_loss(prediction, target, aux, cfg)
~~~

总损失：

~~~text
L = L_MSE + λ_ent * L_ent + λ_div * L_div
~~~

### 10.1 MSE

~~~python
F.mse_loss(prediction, target)
~~~

训练标签先使用训练集均值和标准差标准化，这样 MSE 更容易优化。Pearson、RankIC 和回测仍使用恢复尺度后的原始收益率。

### 10.2 路由熵正则

~~~python
L_ent = sum(p * log(p))
~~~

它的作用是防止重要性分布过于尖锐。

如果没有它，最近 Patch 可能快速获得几乎全部权重，导致：

~~~text
短期、中期、长期窗口都变成同一个短窗口
~~~

### 10.3 专家多样性正则

~~~python
L_div = ||R - I||²
~~~

其中 R 是专家表示的余弦相似度矩阵。

它鼓励不同专家学习互补表示，而不是三个专家都输出几乎相同的向量。

---

## 11. 训练流程

训练函数：

~~~python
train_and_evaluate(cfg, factors, factor_cols)
~~~

训练流程是：

1. 构造 StockBlock；
2. 按日期切分训练集和测试集；
3. 用训练集标签计算 target mean/std；
4. 创建 Dataset 和 DataLoader；
5. 初始化 RAVEN；
6. 使用 AdamW；
7. 使用 CosineAnnealingLR；
8. 每个 epoch 计算 MSE、熵正则和多样性正则；
9. 梯度裁剪；
10. 保存模型和训练历史；
11. 在测试集上生成预测。

论文使用 AdamW、60 epochs、余弦退火。脚本默认沿用这个设置。

如果显存不够：

~~~python
batch_size = 128
~~~

正式训练可以进一步改造成带 padding mask 的完全向量化版本。

---

## 12. 评估指标

函数：

~~~python
evaluate_predictions(predictions)
~~~

脚本输出：

### 12.1 Pearson Corr

所有样本预测收益和真实收益之间的 Pearson 相关系数。

### 12.2 MSE / MAE

使用标准化后的预测和标签计算，便于与论文中的 MSE 数值比较。

### 12.3 每日 IC

对每个交易日的股票截面计算：

~~~text
预测收益 vs 真实未来收益
~~~

得到每日 IC。

### 12.4 RankIC

对预测值和真实值分别排名后计算 Spearman 相关系数。

这更接近你的选股目标。

### 12.5 ICIR / RankICIR

脚本采用论文中的形式：

~~~text
ICIR = mean(IC) / std(IC)
~~~

没有额外乘以平方根频率。

同时输出：

- 正 IC 比例；
- 正 RankIC 比例。

---

## 13. Top-K 简单回测

函数：

~~~python
simple_topk_backtest(predictions, cfg)
~~~

默认设置：

~~~text
每 10 个交易日调仓
选预测值最高的 30 只股票
等权持有
按照股票集合变化计算换手
按照 transaction_cost_bps 扣交易成本
~~~

它输出：

- 总收益；
- 年化收益；
- 近似 Sharpe；
- 最大回撤；
- 平均换手率；
- 调仓次数。

这只是透明的研究级回测，不等价于 Qlib 的完整撮合模拟，也没有模拟：

- 涨跌停；
- 停牌无法成交；
- 开盘/收盘撮合细节；
- 冲击成本曲线；
- 容量约束；
- 行业和风格暴露约束。

正式研究时应把预测输出接入你自己的组合优化和真实成本模块。

---

## 14. 输出文件

运行后主要得到：

~~~text
raven_tushare_data/
├── raw/
│   ├── stock_basic.pkl
│   ├── hs300_members.pkl
│   ├── daily_all.pkl
│   └── daily/
├── processed/
│   ├── cleaned_daily.pkl
│   ├── features.pkl
│   └── feature_names.json
└── outputs/
    ├── raven_model.pt
    ├── training_history.json
    ├── test_predictions.csv
    ├── topk_backtest.csv
    └── metrics.json
~~~

---

## 15. 与论文完全一致和不完全一致的地方

### 已实现的核心结构

- Instance Normalization；
- Channel-independent Patch Embedding；
- 反向时间 Patch 顺序；
- Patch importance scoring；
- CIT 动态连续前缀窗口；
- 短期、中期、长期独立 Transformer 专家；
- Shape-aligned average pooling；
- CAW 相关性感知融合；
- GCR 全局压缩表示；
- MSE + entropy + diversity loss；
- HS300 日频收益预测；
- 2026 年部分年度样本外测试；
- ICIR 和 Top-K 回测。

### 无法完全恢复的地方

1. 论文没有公开完整私有因子清单；
2. 论文正文没有给出全部数据下载和清洗细节；
3. 论文没有完全公开训练随机种子、学习率等所有细节；
4. 表格中的部分基线实现和预处理细节无法仅凭论文复原；
5. 论文使用 Qlib 的完整模拟器，脚本中的 Top-K 回测是透明简化版；
6. 论文的 alpha_k 原始专家置信度具体实现没有完全展开，脚本用可学习的 alpha_head 实现；
7. 论文的 CIT 公式存在阈值边界解释空间，脚本使用“累计重要性首次达到阈值”的实际实现。

因此，脚本适合作为：

~~~text
论文方法复现
+ 可执行的 Tushare 数据管线
+ 可审计的因子工程
+ 可迁移到 Level2 的模型骨架
~~~

而不是宣称可以逐位恢复作者的内部生产代码。

---

## 16. 如何改成你的 Level2 / 分钟级版本

你现在的研究目标是 Level2/逐笔成交聚合到分钟级、预测未来约 3 分钟收益。改造重点如下。

### 16.1 替换数据下载层

保留 RAVEN 模型和 Dataset，只替换：

~~~python
download_market_data()
clean_daily_data()
construct_factors()
~~~

输入改成：

~~~text
[分钟, 订单流/盘口/成交结构因子]
~~~

例如：

- OBI；
- OFI；
- 主动买卖成交；
- 撤单率；
- 盘口深度；
- Spread；
- 价量冲击；
- 集合竞价因子；
- 订单流共振；
- 短期波动和流动性因子。

### 16.2 重新设置时间尺度

论文中的 120 日和 16 日 Patch 不能直接搬到分钟数据。

你需要根据未来 3 分钟的预测目标，用 rolling OOS 比较不同设置，例如：

~~~text
max_lookback = 60 / 120 / 240 根分钟
patch_len = 4 / 8 / 16 根分钟
~~~

最终不能只看训练 MSE，要看：

- RankIC；
- ICIR；
- 月度 IC 同号率；
- IC Decay；
- Top-K / Bottom-K；
- 换手率；
- breakeven cost；
- 扣成本收益；
- 跨股票和跨月份稳定性。

### 16.3 改损失函数

论文是收益回归，因此默认 MSE。

你的主要目标是排序，可以增加：

~~~text
L = MSE + λ_rank * RankLoss + λ_ent * Lent + λ_div * Ldiv
~~~

并用验证集 RankIC 或成本后 Top-K 表现选择模型，而不是只看 MSE。

### 16.4 严格避免泄漏

Level2 场景必须特别检查：

1. 因子滚动窗口只能使用当前时点以前的数据；
2. 标准化参数不能使用未来区间；
3. 标签窗口和训练窗口之间要做 purge；
4. 训练、验证、测试必须按时间切分；
5. 不要用测试集来确定 Patch 长度、专家数量或阈值；
6. 训练后的最终测试只能做一次。

---

## 17. 推荐的实际使用顺序

~~~text
第一步：20只股票、3个epoch，确认脚本能跑通
第二步：100只股票、5~10个epoch，检查损失和路由长度
第三步：完整HS300、60个epoch
第四步：查看 patch_prob 和 expert_weights 是否塌缩
第五步：做去掉 CAW、去掉 GCR、去掉动态路由的消融
第六步：比较 RankIC、ICIR、分组收益、成本后收益
第七步：再迁移到你的 Level2 分钟数据
~~~

尤其要检查：

~~~text
patch_prob 是否总是集中在最近一个 Patch
三个专家的长度是否几乎完全相同
三个专家的余弦相似度是否接近 1
专家权重是否长期只由某一个专家占据
~~~

如果出现这些情况，说明发生了路由塌缩或专家表示塌缩，需要调整熵正则、学习率、Dropout、Patch 长度或专家数量。

