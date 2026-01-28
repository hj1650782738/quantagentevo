# QA方法 vs AA方法 2023年因子表现对比分析报告

## 概述
我们通过观察不同框架在CSI300上挖掘出的因子在不同年份上的预测指标差异性，发现在2022年、2023年QuantaAlpha (QA)除外的框架方法性能表现出现巨大下滑。我们希望结合金融视角洞见来分析。2022-2023年A股市场风格出现巨大转变，逐步由大盘股风格转向小盘股，这一时期游资、私募大量涌入
沪深300 大市值股票涨的更好 整个市场与csi300比市值小 2023年小市值市场风格       多样性 可以挖掘小市值风格     


市场风格很关键，市场风格会发生变化，因子要有效捕捉不同风格
2023 2024 反转效应比较强
23年私募出现 做DMA 小市值风格强，与此前的风格转变
17-21年 反转效应不太强 小市值风格也不太强
故顺着单一路径的学习框架 无法有效捕捉多样化的市场风格转变
反转因子


补充指标 补充指标的优化方向 可能可以加入惩罚项
报告对比分析了**QuantaAlpha (QA)** 方法与**AlphaAgent (AA)** 方法生成的因子库在2023年CSI300股票池上的表现差异。通过策略回测发现，QA方法在2023年的年化收益和最大回撤控制上均显著优于AA方法。本分析旨在从因子层面深入剖析造成这一差异的根本原因，识别主导性因子，并揭示两种方法在因子设计理念上的核心差异。

---

## 一、整体表现对比

### 1.1 统计摘要对比

| 指标 | QA方法 | AA方法 | 差异 |
|------|--------|--------|------|
| 有效因子占比 | 65% | 80% | +15% |
| 无效因子占比 | 35% | 20% | -15% |
| 平均 Rank IC | **0.0057** | 0.0012 |-|
| 最高 Rank IC | **0.0793** | 0.0323 |-|
| 最低 Rank IC | -0.0720 | -0.0280 |-|
| Rank IC > 0.03 因子数 | **17** | 2 |-|

### 1.2 核心发现

> **QA方法的决定性优势来源于三个层面：**
> 1. **Top因子预测能力显著更强**：QA的Top 10因子平均Rank IC为0.052，AA仅为0.015
> 2. **高预测力因子数量更多**：QA有17个因子Rank IC超过0.03，AA仅有2个
> 3. **因子设计维度更丰富**：QA捕获了AA完全缺失的隔夜跳空(Gap)和流动性重估信息

---

## 二、QA方法优势因子深度分析

### 2.1 Top 10 因子列表

| 排名 | 因子名称 | Rank IC | IC IR | 因子类型 |
|------|----------|---------|-------|----------|
| 1 | GapZ10_Overnight_vs_TR | **0.0793** | 0.389 | 隔夜跳空 |
| 2 | Gap_IntradayAcceptanceScore_20D | **0.0744** | 0.392 | 隔夜跳空 |
| 3 | Gap_IntradayAcceptance_VolWeighted_20D | **0.0606** | 0.347 | 隔夜跳空 |
| 4 | CleanTrend_Continuation_Score_RS10_KLEN10_WVMA5 | **0.0590** | 0.334 | 趋势质量 |
| 5 | OrderlyTrend_x_Absorption_10D_5D_20D | **0.0465** | 0.270 | 趋势+流动性 |
| 6 | Liquidity_Rerating_DlvUp_IlliqDown_RangeVol_20_60 | **0.0458** | 0.249 | 流动性重估 |
| 7 | Mom120_RangeStdRatio_SignedCLV_5 | **0.0438** | 0.210 | 动量+波动 |
| 8 | DonchianWidthCompress20_HighBreak55_VolumeRatio20 | **0.0424** | 0.261 | 突破确认 |
| 9 | Stealth_Accum_VolSurprise60_IlliqDown20_VolStd20_Pen20 | **0.0415** | 0.249 | 隐蔽积累 |
| 10 | VolSqueeze_ReturnStd20_BreakoutClose55_VolZ20_RangePos1 | **0.0409** | 0.241 | 波动压缩 |

### 2.2 核心优势因子群分析

#### 特征一：隔夜跳空信息的独特利用（AA完全缺失）

**代表因子**: `GapZ10_Overnight_vs_TR`, `Gap_IntradayAcceptanceScore_20D`

**核心表达式**:
```
ABS(LOG($open/(DELAY($close,1)+1e-8))) / (TS_MEAN(TrueRange/Close, 10)+1e-8)
```

**经济学逻辑**:
- 隔夜跳空（Open vs 前日Close）反映了非交易时段的信息冲击
- 通过与近期真实波幅(True Range)的比值进行标准化，识别"异常大"的跳空
- 日内价格行为（收盘位置、成交量）判断市场对跳空的"接受"或"拒绝"

**2023年有效原因**:
1. **集合竞价信息密度提升**：2023年量化资金在集合竞价的参与度提高，隔夜跳空包含更多前瞻性信息
2. **Gap-Fill规律稳定**：震荡市中，极端跳空后的均值回归（Gap-Fill）规律更加显著
3. **T+1制度下的隔夜风险溢价**：A股无法日内平仓，隔夜持仓风险需要补偿，形成可预测的价格模式

**关键洞察**:
> AA方法生成因子中，**没有任何一个因子利用了隔夜跳空信息**。这是QA方法相对AA方法的**最大信息增量来源**，单凭Gap系列因子就贡献了Top 3的表现。

#### 特征二：趋势质量过滤机制

**代表因子**: `CleanTrend_Continuation_Score_RS10_KLEN10_WVMA5`

**核心表达式**:
```
ZSCORE(SIGN(趋势斜率) * INV(残差波动)) - ZSCORE(日内波幅) - ZSCORE(|成交量加权收益|)
```

**经济学逻辑**:
- 不是简单的动量因子，而是"高质量趋势"的筛选器
- 要求价格路径呈现清晰的线性趋势（高R²），同时日内波动和成交量冲击都较小
- 识别"有序上涨/下跌"而非"震荡中的偶然方向"

**2023年有效原因**:
1. **噪声市中的信号提纯**：2023年市场噪声大，简单动量失效，但"干净趋势"仍有延续性
2. **机构行为特征**：机构建仓/减仓通常体现为平滑、持续的价格轨迹，而非剧烈波动
3. **与成交量冲击的负相关**：避开了"放量冲高回落"的陷阱

#### 特征三：流动性重估（Liquidity Rerating）

**代表因子**: `Liquidity_Rerating_DlvUp_IlliqDown_RangeVol_20_60`

**核心表达式**:
```
RANK(成交额增长趋势) + RANK(-Amihud冲击下降) - RANK(波动率) - 0.5*RANK(|价格变动|)
```

**经济学逻辑**:
- 寻找"流动性正在改善"的股票：成交额上升、价格冲击下降
- 同时控制波动率和动量暴露，避免单纯追涨
- 本质是捕捉"隐蔽积累"(Stealth Accumulation)的早期信号

**2023年有效原因**:
1. **机构换仓的前瞻信号**：2023年结构性行情中，资金从传统板块流向AI等新方向，流动性重估先于价格变动
2. **反映真实需求**：成交额上升但冲击下降，说明是"有承接的买入"而非"恐慌性抛售"
3. **与AA的穷竭逻辑相反**：不是等待"量价背离后的反转"，而是追随"流动性改善的延续"

---

## 三、AA方法因子分析

### 3.1 Top 10 因子列表

| 排名 | 因子名称 | Rank IC | IC IR | 因子类型 |
|------|----------|---------|-------|----------|
| 1 | Exhaustion_Intensity_Index_10D | 0.0323 | 0.147 | 穷竭指数 |
| 2 | Climax_Exhaustion_Intensity | 0.0242 | 0.098 | 高潮穷竭 |
| 3 | Relative_Climax_Reversal_Index | 0.0127 | 0.051 | 高潮反转 |
| 4 | Exhaustion_Volume_Instability_Index | 0.0121 | 0.050 | 量价穷竭 |
| 5 | Relative_Range_Efficiency_Climax | 0.0121 | 0.080 | 效率高潮 |
| 6 | Relative_Volatility_Force_Index | 0.0117 | 0.054 | 波动力度 |
| 7 | Ranked_Trend_Exhaustion_Factor | 0.0116 | 0.055 | 趋势穷竭 |
| 8 | Asymmetric_Liquidity_Climax_V1 | 0.0104 | 0.047 | 流动性高潮 |
| 9 | Stability_Adjusted_Reversal_V2 | 0.0099 | 0.060 | 稳定性反转 |
| 10 | Volatility_Adjusted_Exhaustion_V1 | 0.0098 | 0.046 | 波动穷竭 |

### 3.2 AA方法的局限性分析

#### 局限一：因子同质性过高

**观察**: AA的Top 10因子中，有**8个因子名称包含"Exhaustion"（穷竭）或"Climax"（高潮）**，核心逻辑高度相似：

```
价格偏离(60日均线/Z-Score) × 成交量比率(短期/长期)
```

**问题**:
1. 因子之间相关性高，组合后的边际增益有限
2. 押注单一市场假设：价格极端+量价背离 → 反转
3. 缺乏对不同市场微观结构的利用

#### 局限二：反转逻辑在2023年效果受限

**AA因子的核心假设**: "价格穷竭+成交量异常 → 均值回归"

**2023年市场特点**:
- **结构性趋势强劲**：AI概念股持续上涨，传统行业持续下跌，"穷竭后反转"的假设失效
- **资金流向单边化**：北向资金持续流出、量化资金集中持仓，加剧了趋势延续而非反转
- **反转窗口缩短**：即使出现技术性反弹，持续时间也较短，难以捕获

**数据佐证**: AA因子中"Bottom_Fishing"（抄底）类因子表现最差
- `LVR_Bottom_Fishing_20D`: Rank IC = **-0.019**
- `Relative_Volume_Calm_Reversal`: Rank IC = **-0.028**

#### 局限三：缺失关键信息维度

| 信息维度 | QA方法 | AA方法 |
|----------|--------|--------|
| 隔夜跳空(Gap) | ✅ 多个Top因子 | ❌ 完全缺失 |
| 趋势质量(R²/残差) | ✅ CleanTrend系列 | ❌ 仅有简单动量 |
| 流动性重估 | ✅ Rerating系列 | ⚠️ 仅有静态比率 |
| 突破确认 | ✅ Donchian/Bollinger | ❌ 缺失 |
| 隐蔽积累 | ✅ Stealth_Accum | ❌ 缺失 |

---

## 四、共有因子的表现差异

部分因子在QA和AA因子库中都有出现（或有相似逻辑），其2023年表现差异反映了因子设计细节的重要性：

| 因子逻辑 | QA版本Rank IC | AA版本Rank IC | 差异原因 |
|----------|---------------|---------------|----------|
| 穷竭强度 | 0.0323 (同名) | 0.0323 | 相同 |
| 流动性反转 | 0.0217 (Boundary) | -0.0133 (LVR_V3) | QA使用边界检测，AA使用条件门控 |
| 波动压缩 | 0.0409 (VolSqueeze) | N/A | AA缺失该维度 |
| 价格Z-Score | 0.0319 (Trend组合) | 0.0009 (纯Z-Score) | QA与趋势质量组合，AA单独使用 |

**关键发现**: 即使是相似的经济学假设，QA的因子设计在以下方面更优：
1. **多维度交叉验证**：不单独使用任何单一信号
2. **动态调整机制**：根据市场状态调整权重
3. **噪声过滤**：通过排名(RANK)、Z-Score等方式降低极端值影响

---

## 五、2023年主导因子解析

### 5.1 QA方法的主导因子群

根据预测能力和独特性，QA方法2023年的主导因子可分为三个层级：

**第一梯队（决定性贡献，Rank IC > 0.05）**:
1. **Gap系列**：`GapZ10_Overnight_vs_TR`, `Gap_IntradayAcceptanceScore_20D`
   - 贡献：捕获隔夜定价效率，预测次日方向
   - 独特性：AA完全不具备

2. **CleanTrend系列**：`CleanTrend_Continuation_Score`
   - 贡献：过滤噪声，提取高质量趋势信号
   - 独特性：AA仅有简单动量

**第二梯队（重要贡献，Rank IC 0.03-0.05）**:
- `OrderlyTrend_x_Absorption`: 趋势+流动性吸收的交叉验证
- `Liquidity_Rerating`: 流动性改善的前瞻信号
- `Stealth_Accum`: 隐蔽积累的早期识别

**第三梯队（边际贡献，Rank IC 0.02-0.03）**:
- `DonchianWidthCompress`: 波动压缩后的突破
- `BollingerSqueeze`: 布林带收窄信号

### 5.2 AA方法的有限优势

AA方法在以下细分场景仍有价值：

1. **极端穷竭事件**：当价格确实出现极端偏离且成交量异常时，`Exhaustion_Intensity_Index_10D`能捕获反弹机会
2. **波动率均值回归**：部分基于波动率的因子在特定区间有效

但这些场景在2023年出现频率较低，整体贡献有限。

---

## 六、结论与策略建议

### 6.1 核心结论

1. **QA方法2023年优势的根本来源是"信息维度的丰富性"**
   - Gap（隔夜跳空）、CleanTrend（趋势质量）、Liquidity Rerating（流动性重估）三类因子是AA完全缺失或显著不足的
   - 这些因子贡献了QA策略回测中大部分的超额收益

2. **AA方法的"穷竭-反转"假设在2023年表现不佳**
   - 结构性行情中，趋势延续比均值回归更常见
   - 因子同质性高，难以通过组合分散风险

3. **因子设计的精细程度决定了有效性**
   - 同样的经济学假设，QA的多维度交叉验证设计显著优于AA的单一信号

### 6.2 因子组合配置建议

基于2023年表现分析，未来因子组合可考虑：

**超配（权重 > 均值）**:
- 隔夜跳空类：Gap系列
- 趋势质量类：CleanTrend、OrderlyTrend系列
- 流动性重估类：Liquidity_Rerating系列

**标配（权重 = 均值）**:
- 穷竭强度类（仅保留Top 2-3个）
- 波动压缩类：Squeeze系列

**低配或剔除（权重 < 均值）**:
- 纯反转类：Bottom_Fishing、Calm_Reversal等
- 高同质性因子：保留代表性因子，剔除冗余

### 6.3 方法论改进建议

**对AA方法的改进方向**:
1. **引入隔夜信息**：增加Open vs 前日Close的分析
2. **趋势质量过滤**：在反转因子中加入趋势强度的条件门控
3. **降低因子同质性**：限制同一假设下生成的因子数量
4. **动态权重机制**：根据市场状态调整因子权重
---

## 附录：关键因子表达式对照

### QA Top 3 因子

**1. GapZ10_Overnight_vs_TR**
```
ABS(LOG($open/(DELAY($close,1)+1e-8))) / 
(TS_MEAN(MAX($high-$low, MAX(ABS($high-DELAY($close,1)), ABS($low-DELAY($close,1)))) / 
(DELAY($close,1)+1e-8), 10) + 1e-8)
```
**2. Gap_IntradayAcceptanceScore_20D**
```
SIGN(LOG($close/($open+1e-8))) * ABS(LOG($open/(DELAY($close,1)+1e-8))) / (TS_STD($return,20)+1e-8)
```
**3. CleanTrend_Continuation_Score**
```
ZSCORE(SIGN(REGBETA(LOG($close),SEQUENCE(10),10)) * INV(TS_STD(REGRESI(LOG($close),SEQUENCE(10),10),10)+1e-8)) 
- ZSCORE(TS_MEAN(($high-$low)/($close+1e-8),10)) 
- ZSCORE(ABS(TS_SUM($volume*$return,5)/(TS_SUM($volume,5)+1e-8)))
```

### AA Top 3 因子

**1. Exhaustion_Intensity_Index_10D**
```
RANK(TS_PCTCHANGE($close, 60)) * (TS_MEAN($volume, 60) / (TS_MEAN($volume, 10) + 1e-8))
```
**2. Climax_Exhaustion_Intensity**
```
RANK(TS_PCTCHANGE($close, 60)) * (TS_MEAN($volume, 5) / (TS_MEDIAN($volume, 60) + 1e-8))
```
**3. Relative_Climax_Reversal_Index**
```
(($close - BB_MIDDLE($close, 60)) / $close) * (TS_MEAN($volume, 5) / (TS_MEDIAN($volume, 60) + 1e-8))
```


