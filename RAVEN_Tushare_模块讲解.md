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
| 数据区间 | 2008-01-01 至 2024-12-31（窗口预热） |
| 拟合区间 | 2009-01-01 至 2018-12-31（2008 年用于窗口预热） |
| 验证集 | 2019-01-01 至 2019-12-31（从训练期留出） |
| 测试集 | 2020-01-01 至 2024-12-31 |
| 最大历史窗口 | 120 个交易日 |
| Patch 长度 | 16 个交易日 |
| 专家数 | 3 |
| CIT 阈值 | 0.3、0.6、0.9 |
| 标签 | 未来 10 个交易日累计对数收益 |
| 训练损失 | MSE + 熵正则 + 专家多样性正则 |

---

## 2. 文件最开始需要改什么

脚本不再把真实 Token 写进 Python 文件，而是读取本地环境变量：

PowerShell 设置环境变量：

~~~powershell
$env:TUSHARE_TOKEN="你的新 Tushare Token"
~~~

脚本还允许直接在 Config 中修改：

~~~python
index_code = "000300.SH"
start_date = "20080101"
end_date = "20241231"
train_start = "20090101"
train_end = "20191231"
valid_start = "20190101"
valid_end = "20191231"
test_start = "20200101"
test_end = "20241231"
forecast_horizon = 10
max_lookback = 120
patch_len = 16
~~~

曾有真实 Token 被提交到公开仓库。请将其视为已泄露，在 Tushare 管理后台立即撤销并重新生成；新 Token 只保存在本地环境变量中，不要提交到 GitHub。脚本现已从环境变量读取 Token。

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
python raven_tushare_reproduction.py --mode all --max-stocks 20 --epochs 3 --force-download --force-rebuild-features
~~~

这个命令会：

1. 下载数据；
2. 清洗数据；
3. 构造因子；
4. 构造序列样本；
5. 训练 3 轮，并输出验证集 Loss、RankIC；
6. 根据验证集 RankIC 保存最佳模型；
7. 输出验证集与测试集指标、简单 Top-K 回测。

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

缓存会记录指数、日期范围、股票数量和是否下载 `daily_basic`。其中任何一项改变，脚本会自动重新下载并重建因子；`--force-download` 也会同时让处理后的因子缓存失效，避免“配置已改、实际数据仍是旧年份”的错配。

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
拟合：2009-01-01 至 2018-12-31（2008 年只用于窗口预热）
验证：2019-01-01 至 2019-12-31（从拟合样本中留出）
测试：2020-01-01 至 2024-12-31
~~~

该切分遵循论文 2009–2019 训练期、2020–2024 样本外测试期；2008 年数据用于滚动特征和 120 日输入窗口预热。2019 年作为训练末段的独立验证集，不参与梯度拟合或标签标准化。测试集只用于最终评估，不能用于调参。

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
