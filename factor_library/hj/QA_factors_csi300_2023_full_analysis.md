**QA: CSI300 2023 因子年化 IC / RankIC 完整计算**

概述：使用 `AlphaAgent/factor_library/hj/RANKIC_desc_150_QA_round11_best_gpt_123_csi300.json` 中的因子，强制仅使用 CSI300 缓存（已为大部分缺失因子重算 CSI300 缓存），运行项目自带的 `yearly_ic_calculator.py` 计算 2023 年度的因子 annual IC 与 annual RankIC，并导出结果 CSV（已保存为 `AlphaAgent/factor_library/hj/csi300_2023_ic_metrics.csv`）。

关键事实：
- 有效因子数：147 / 150（2 个因子计算失败）
- 结果 CSV：AlphaAgent/factor_library/hj/csi300_2023_ic_metrics.csv

Top10（按 annual_ic 排序）:
1. GapZ10_Overnight_vs_TR — annual_ic=0.033494, annual_rank_ic=0.079341
2. Gap_IntradayAcceptanceScore_20D — annual_ic=0.033009, annual_rank_ic=0.074397
3. Gap_IntradayAcceptance_VolWeighted_20D — annual_ic=0.031406, annual_rank_ic=0.060574
4. Mom120_RangeStdRatio_SignedCLV_5 — annual_ic=0.030543, annual_rank_ic=0.043825
5. Liquidity_Rerating_DlvUp_IlliqDown_RangeVol_20_60 — annual_ic=0.029720, annual_rank_ic=0.045774
6. Stealth_Accum_VolSurprise60_IlliqDown20_VolStd20_Pen20 — annual_ic=0.028346, annual_rank_ic=0.041514
7. OrderlyTrend_x_Absorption_10D_5D_20D — annual_ic=0.027189, annual_rank_ic=0.046534
8. CleanTrend_Continuation_Score_RS10_KLEN10_WVMA5 — annual_ic=0.026687, annual_rank_ic=0.059028
9. BollingerSqueeze20_TSRankClose55_LogVolZ20 — annual_ic=0.024360, annual_rank_ic=0.035291
10. DonchianWidthCompress20_HighBreak55_VolumeRatio20 — annual_ic=0.023874, annual_rank_ic=0.042396

Top10（按 annual_rank_ic 排序）:
1. GapZ10_Overnight_vs_TR — annual_rank_ic=0.079341, annual_ic=0.033494
2. Gap_IntradayAcceptanceScore_20D — annual_rank_ic=0.074397, annual_ic=0.033009
3. Gap_IntradayAcceptance_VolWeighted_20D — annual_rank_ic=0.060574, annual_ic=0.031406
4. CleanTrend_Continuation_Score_RS10_KLEN10_WVMA5 — annual_rank_ic=0.059028, annual_ic=0.026687
5. OrderlyTrend_x_Absorption_10D_5D_20D — annual_rank_ic=0.046534, annual_ic=0.027189
6. Liquidity_Rerating_DlvUp_IlliqDown_RangeVol_20_60 — annual_rank_ic=0.045774, annual_ic=0.029720
7. Mom120_RangeStdRatio_SignedCLV_5 — annual_rank_ic=0.043825, annual_ic=0.030543
8. DonchianWidthCompress20_HighBreak55_VolumeRatio20 — annual_rank_ic=0.042396, annual_ic=0.023874
9. Stealth_Accum_VolSurprise60_IlliqDown20_VolStd20_Pen20 — annual_rank_ic=0.041514, annual_ic=0.028346
10. VolSqueeze_ReturnStd20_BreakoutClose55_VolZ20_RangePos1 — annual_rank_ic=0.040892, annual_ic=0.023834

简要分析（结论导向）：
- Gap 类（隔夜/开盘缺口相关）因子在 2023 年度表现突出，既在 Pearson IC 也在 Rank IC 上均列于前列，说明缺口驱动的短中期信号在 CSI300 市场的 2023 年有较稳定的信息量。 
- 多个流动性/吸收类因子（如 `Liquidity_Rerating...`、`Stealth_Accum...`）也上榜，表明成交量/交投相关特征对收益有解释能力。 
- annual_ic 与 annual_rank_ic 的排序总体一致，但有细微差别（例如 `CleanTrend_Continuation...` 在 rankIC 更靠前），提示一些因子在排序稳定性上优于线性相关强度。 
- 当前还有 2 个因子计算失败，建议：
  - 查看 `AlphaAgent/factor_library/hj/RANKIC_desc_150_QA_round11_best_gpt_123_csi300.json` 中对应因子的 `factor_expression`，定位不兼容表达式或数据问题；
  - 可尝试启用 LLM 辅助（若可接受并且有相应依赖）或逐因子调试表达式以完成缓存。 

下一步建议：
- 如需，我可以：
  1) 帮你定位并修复那 2 个失败的因子（逐因子调试表达式）；
  2) 将完整的 `csi300_2023_ic_metrics.csv` 上传为 MLflow artifact 或导出为 PDF 报告；
  3) 基于 top 因子构建等权/回归回测来验证信息可转化为因子收益。

文件位置：
- 结果 CSV：AlphaAgent/factor_library/hj/csi300_2023_ic_metrics.csv
- 本分析文件：AlphaAgent/factor_library/hj/QA_factors_csi300_2023_full_analysis.md
