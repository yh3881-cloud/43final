# 项目流程：纽约 Airbnb 高性价比房源自动挖掘

## 1. 项目目标

识别纽约市 Airbnb 中 **实际价格低于模型预测理论价格**，同时 **评分较高、交通便利、周边条件合理** 的高性价比房源。

核心思路：

```text
多源数据
→ 数据清洗与整合
→ 特征工程
→ 理论价格预测
→ 价格残差分析
→ 低估房源识别
→ 空间聚类分析
→ 结果解释
```

---

## 2. 研究问题

> 哪些纽约 Airbnb 房源在房屋条件、交通、安全和评分相近的情况下，实际价格显著低于理论价格？

进一步分析：

1. 哪些特征最影响 Airbnb 的理论价格？
2. 哪些房源存在明显低估？
3. 这些低估房源是否集中在特定社区？
4. 这些区域是否形成空间上的“价值洼地”？

---

## 3. 整体 Pipeline

```text
Raw Data
   ↓
Data Cleaning
   ↓
Data Integration
   ↓
Feature Engineering
   ↓
Price Prediction Model
   ↓
Residual Analysis
   ↓
Undervalued Candidate Detection
   ↓
Spatial Clustering
   ↓
Interpretation & Reporting
```

---

## 4. 数据来源

| 数据 | 作用 |
|---|---|
| Airbnb listings | 房源属性、房型、价格、评分、房东信息、经纬度 |
| Airbnb calendar | 每日价格、可订性、价格波动 |
| Airbnb reviews | 评论活跃度、房源可靠性 proxy |
| MTA subway stations | 地铁距离、交通便利性 |
| NYPD complaints | 周边 reported-crime intensity |

---

## 5. 数据粒度统一

| 原始数据 | 原始粒度 | 处理方式 |
|---|---|---|
| listings | 一行一个房源 | 作为主表 |
| calendar | 一行一个房源的一天 | 按 listing 聚合 |
| reviews | 一行一条评论 | 按 listing 聚合 |
| subway | 一行一个地铁站 | 计算距离特征 |
| crime | 一行一条报案记录 | 计算周边报案强度 |

最终数据结构：

```text
One row = One Airbnb listing
```

---

## 6. 数据清洗

| 数据 | 清洗内容 |
|---|---|
| listings | 清洗价格、百分比、t/f 字段、经纬度、设施数量 |
| calendar | 过滤时间窗、清洗每日价格、计算可订性 |
| reviews | 过滤时间窗、计算评论活跃度 |
| subway | 清洗地铁站经纬度 |
| crime | 清洗报案日期、经纬度、犯罪类别 |

---

## 7. 数据整合

整合逻辑：

```text
listings
+ aggregated calendar features
+ aggregated review features
+ subway accessibility features
+ reported-crime intensity features
= listing-level modeling table
```

关键要求：

```text
calendar 先按 listing_id 聚合
reviews 先按 listing_id 聚合
subway 和 crime 通过经纬度计算空间特征
最终表格保持一行一个房源
```

---

## 8. Feature Engineering

| 特征组 | 示例变量 | 含义 |
|---|---|---|
| 房源属性 | room_type, bedrooms, beds, bathrooms, accommodates | 房源硬件条件 |
| 房东特征 | host_is_superhost, host_response_rate, host_acceptance_rate | 房东可靠性 |
| 价格特征 | effective_price, log_effective_price | 实际价格与建模目标 |
| Calendar 特征 | calendar_median_price, price_volatility, available_rate | 动态价格与可订性 |
| 评论特征 | reviews_in_window, reviews_last_90d | 房源活跃度 |
| 交通特征 | subway_distance, stations_within_500m | 地铁便利性 |
| Crime 特征 | crime_count_1000m, crime_intensity_log | 周边报案强度 |
| 缺失特征 | missing_rating, missing_calendar_price | 缺失信息记录 |

---

## 9. 核心价格变量

实际价格：

```text
effective_price = calendar_median_price if available
                  else listing price
```

建模目标：

```text
log_effective_price = log1p(effective_price)
```

价格预测任务：

```text
Predict log_effective_price
```

---

## 10. 价格预测模型

| 模型 | 作用 |
|---|---|
| Ridge Regression | 线性 baseline |
| Random Forest Regressor | 捕捉非线性关系 |
| Gradient Boosting / XGBoost | 提升预测性能 |

评估指标：

| 指标 | 含义 |
|---|---|
| MAE | 平均绝对误差 |
| RMSE | 对大误差更敏感 |
| R² | 模型解释度 |
| MAPE | 百分比误差 |

---

## 11. 理论价格与残差

模型输出：

```text
predicted_log_price
predicted_price = expm1(predicted_log_price)
```

残差指标：

```text
price_gap = predicted_price - effective_price
```

```text
undervaluation_ratio = predicted_price / effective_price
```

```text
price_residual_log = predicted_log_price - log_effective_price
```

含义：

```text
price_residual_log 越大，房源越可能被低估
undervaluation_ratio 越大，实际价格相对理论价格越低
```

---

## 12. Undervalued Candidate 定义

一个房源被标记为 `undervalued_candidate = 1`，需要满足：

| 条件 | 含义 |
|---|---|
| price_residual_log 位于前 5% 或 10% | 模型认为价格明显低估 |
| review_scores_rating ≥ 4.8 | 评分较高 |
| number_of_reviews ≥ 5 或 reviews_in_window ≥ 2 | 评论证据足够 |
| effective_price > 0 | 价格有效 |
| distance_to_nearest_subway_km 合理 | 交通条件可接受 |
| crime_intensity_log_1000m 不高 | 周边 reported-crime intensity 可接受 |

核心输出字段：

```text
undervalued_candidate
predicted_price
effective_price
price_gap
undervaluation_ratio
price_residual_log
```

---

## 13. 空间聚类分析

目标：

> 分析低估房源是否集中在特定区域，识别纽约市 Airbnb 的价值洼地。

推荐方法：

| 方法 | 用途 |
|---|---|
| DBSCAN | 发现自然空间聚集 |
| K-Means | 对比性聚类方法 |

聚类特征：

```text
latitude
longitude
effective_price
predicted_price
undervaluation_ratio
review_scores_rating
distance_to_nearest_subway_km
crime_intensity_log_1000m
```

聚类输出：

| 字段 | 含义 |
|---|---|
| cluster_id | 价值洼地编号 |
| number_of_listings | 房源数量 |
| median_effective_price | 中位实际价格 |
| median_predicted_price | 中位理论价格 |
| median_undervaluation_ratio | 中位低估比例 |
| median_rating | 中位评分 |
| common_neighborhoods | 主要社区 |
| median_subway_distance | 中位地铁距离 |
| median_crime_intensity | 中位 reported-crime intensity |

---

## 14. EDA 重点

| 分析方向 | 目标 |
|---|---|
| 价格分布 | 理解价格偏态和异常值 |
| log price 分布 | 改善价格建模稳定性 |
| 实际价格 vs 预测价格 | 检查模型预测效果 |
| 残差分布 | 找出明显低估房源 |
| 低估房源评分 | 确认房源质量 |
| 地铁距离 | 检查交通条件 |
| crime intensity | 检查周边环境 |
| 社区分布 | 找出价值洼地 |
| 聚类地图 | 展示空间集中区域 |
| 特征重要性 | 解释理论价格驱动因素 |

---

## 15. 推荐图表

| 图表 | 作用 |
|---|---|
| effective_price histogram | 查看价格分布 |
| log_effective_price histogram | 查看 log 价格分布 |
| predicted vs actual price scatter plot | 查看预测效果 |
| residual distribution plot | 查看低估程度 |
| undervaluation_ratio by borough | 比较区域差异 |
| undervalued candidates map | 展示低估房源位置 |
| DBSCAN cluster map | 展示价值洼地聚类 |
| feature importance plot | 解释模型结果 |

---

## 16. 最终输出

| 输出 | 内容 |
|---|---|
| processed dataset | 清洗后的 listing-level 数据 |
| model comparison table | 不同价格预测模型表现 |
| predicted price table | 理论价格、实际价格、残差 |
| undervalued candidates | 高性价比低估房源 |
| cluster summary | 价值洼地聚类结果 |
| EDA figures | 价格、残差、地图、聚类图 |
| final report | 数据、模型、结果解释 |

建议输出文件：

```text
data/processed/nyc_airbnb_model_table.csv
data/processed/nyc_airbnb_undervalued_model_table.csv
outputs/model_comparison.csv
outputs/undervalued_candidates.csv
outputs/undervalued_cluster_summary.csv
outputs/figures/
```

---

## 17. 报告主线

报告可以按照以下结构展开：

```text
1. 数据来源与清洗
2. 多源数据整合
3. 特征工程
4. 理论价格预测模型
5. 价格残差与低估房源识别
6. 空间聚类与价值洼地分析
7. 模型解释与商业意义
8. 局限性与改进方向
```

---

## 18. 一句话总结

本项目使用多源时空数据和机器学习模型估计纽约 Airbnb 房源的理论市场价格，识别实际价格显著低于理论价格且评分较高的高性价比房源，并分析这些房源在空间上形成的价值洼地。
