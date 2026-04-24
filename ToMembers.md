# Project Instructions: NYC Airbnb Consumer Value Identification Pipeline

## 1. Project Positioning 项目定位

我们的目标是构建一个：

> **可复现、可扩展的机器学习分析流程，用于从消费者视角识别并解释纽约市 Airbnb 房源的消费者价值。**
项目核心思想是：

```text
Raw multi-source data
→ Data cleaning and integration
→ Listing-level feature table
→ Consumer-value feature engineering
→ Hidden Gem / Overpriced Trap label construction
→ EDA and unsupervised learning
→ Supervised classification
→ Model interpretation
```

也就是说，本项目的重点不是一次性找出几个“好房源”，而是设计一个可以重复运行的分析框架。  
当未来获得新的 Airbnb 数据或外部环境数据时，这个 pipeline 可以重新完成数据清洗、特征构造、标签生成和模型分析。

---

## 2. Main Research Question

### 主研究问题

我们希望回答：

> 能否通过 Airbnb listings、calendar、reviews、地铁站和 reported-crime 数据，构建一个可复现的机器学习流程，识别哪些纽约 Airbnb 房源对消费者来说是高性价比选择，哪些可能是高价低值选择？

---

## 3. Project Scope 当前项目范围

本项目当前重点包括：

1. Data Acquisition  
2. Data Cleaning  
3. Data Integration  
4. Data Preprocessing  
5. Feature Engineering  
6. Label Construction  
7. Exploratory Data Analysis  
8. Unsupervised Learning for structure discovery  
9. Preparation for supervised modeling  

---

## 4. Machine Learning Value 机器学习价值

最终数据集可以支持：

```text
EDA
Clustering
PCA / dimensionality reduction
Multiclass classification
Binary classification
Model comparison
Feature importance
Model interpretation
```

建模目标不是简单复现规则，而是进一步分析：

```text
哪些特征组合能够有效识别消费者价值？
哪些变量对 Hidden Gem / Overpriced Trap 最有解释力？
即使排除直接构造标签的变量，模型是否仍能学到有意义的模式？
```

---

## 5. EDA Instructions

EDA 的目标不是直接建模，而是理解数据结构、变量分布、标签合理性和潜在建模方向。

---

### 5.1 Basic Data Quality EDA

需要检查：

```python
df.shape
df.head()
df.info()
df.describe(include="all")
df.duplicated(subset=["id"]).sum()
df.isna().mean().sort_values(ascending=False).head(30)
```

必须确认：

```text
最终数据是一行一个 listing
id 没有重复
核心字段没有严重缺失
calendar / reviews / subway / crime features 是否成功合并
consumer_value_class 是否存在
```

---

### 5.2 Label Distribution EDA

检查：

```python
df["consumer_value_class"].value_counts()
df["consumer_value_class"].value_counts(normalize=True)
```

需要回答：

```text
Hidden Gem 是否过少？
Overpriced Trap 是否过多？
是否存在 class imbalance？
后续建模是否需要 class weight 或 resampling？
```

建议图表：

```text
Bar chart of consumer_value_class
Hidden Gem / Normal / Trap percentage chart
```

---

### 5.3 Price and Value EDA

重点变量：

```text
price
effective_price
calendar_median_price
calendar_avg_price
calendar_price_volatility
weekend_price_premium
```

建议分析：

```text
price distribution
log price distribution
effective_price by class
price volatility by class
weekend premium by class
```

建议图表：

```text
Histogram of effective_price
Histogram of log_effective_price
Boxplot of effective_price by consumer_value_class
Boxplot of calendar_price_volatility by class
Boxplot of weekend_price_premium by class
```

需要回答：

```text
价格是否右偏？
是否存在极端高价？
Hidden Gem 是否明显更便宜？
Overpriced Trap 是否价格明显偏高？
calendar price 和 listing price 差异是否明显？
```

---

### 5.4 Rating and Review EDA

重点变量：

```text
review_scores_rating
review_scores_cleanliness
review_scores_location
review_scores_value
number_of_reviews
reviews_in_window
reviews_last_90d
reviews_last_180d
days_since_last_review_in_window
```

建议图表：

```text
Boxplot of review_scores_rating by class
Boxplot of review_scores_value by class
Boxplot of reviews_in_window by class
Histogram of days_since_last_review_in_window
Scatter plot of effective_price vs review_scores_value
```

需要回答：

```text
Hidden Gem 是否评分和 value score 更高？
Overpriced Trap 是否 value score 更低？
是否存在高评分但长期没有新评论的房源？
评论活跃度是否区分不同类别？
```

---

### 5.5 Room Type and Property Type EDA

重点变量：

```text
room_type
property_type
consumer_value_class
effective_price
review_scores_rating
```

建议分析：

```python
pd.crosstab(df["room_type"], df["consumer_value_class"], normalize="index")
```

建议图表：

```text
Stacked bar chart of class by room_type
Boxplot of effective_price by room_type
Boxplot of review_scores_value by room_type
```

需要回答：

```text
哪种 room type 更容易出现 Hidden Gem？
哪种 room type 更容易出现 Overpriced Trap？
Private room 是否更容易有高性价比？
Entire home/apt 是否更容易高价？
```

---

### 5.6 Geographic EDA

重点变量：

```text
neighbourhood_group_cleansed
neighbourhood_cleansed
latitude
longitude
consumer_value_class
effective_price
```

建议计算：

```python
neigh_summary = (
    df.groupby("neighbourhood_cleansed")
      .agg(
          listings=("id", "count"),
          hidden_gems=("hidden_gem_label", "sum"),
          overpriced_traps=("overpriced_trap_label", "sum"),
          median_price=("effective_price", "median")
      )
      .reset_index()
)

neigh_summary["hidden_gem_rate"] = neigh_summary["hidden_gems"] / neigh_summary["listings"]
neigh_summary["overpriced_trap_rate"] = neigh_summary["overpriced_traps"] / neigh_summary["listings"]
```

建议只分析 listing 数足够的社区：

```python
neigh_summary = neigh_summary[neigh_summary["listings"] >= 30]
```

建议图表：

```text
Top neighborhoods by Hidden Gem count
Top neighborhoods by Hidden Gem rate
Top neighborhoods by Overpriced Trap count
Top neighborhoods by Overpriced Trap rate
Map scatter plot colored by consumer_value_class
Map scatter plot colored by effective_price
```

需要回答：

```text
Hidden Gem 集中在哪些 borough / neighborhood？
Overpriced Trap 集中在哪里？
是否存在非核心旅游区但高性价比的社区？
```

---

### 5.7 Subway Accessibility EDA

重点变量：

```text
distance_to_nearest_subway_km
subway_stations_within_500m
subway_stations_within_1000m
consumer_value_class
```

建议图表：

```text
Boxplot of subway distance by class
Boxplot of subway stations within 500m by class
Scatter plot of subway distance vs effective_price
Scatter plot of subway distance vs review_scores_location
```

需要回答：

```text
Hidden Gem 是否更靠近地铁？
Overpriced Trap 是否交通表现较弱？
低价房源是否可能因为远离地铁才便宜？
地铁距离和 location score 是否一致？
```

---

### 5.8 Reported-Crime Intensity EDA

如果 crime data 可用，分析：

```text
crime_count_500m
crime_count_1000m
violent_crime_count_1000m
property_crime_count_1000m
crime_intensity_log_1000m
consumer_value_class
```

建议图表：

```text
Boxplot of crime_intensity_log_1000m by class
Boxplot of violent_crime_count_1000m by class
Scatter plot of crime_intensity_log_1000m vs effective_price
Crime intensity by borough
Crime intensity by neighborhood
```

需要回答：

```text
Hidden Gem 是否 reported-crime intensity 较低？
Overpriced Trap 是否周边 reported incidents 较高？
crime intensity 是否与 borough、价格、地铁距离相关？
是否需要 log transform？
```

---

### 5.9 Host Feature EDA

重点变量：

```text
host_is_superhost
host_response_rate
host_acceptance_rate
host_identity_verified
calculated_host_listings_count
consumer_value_class
```

建议分析：

```python
pd.crosstab(df["host_is_superhost"], df["consumer_value_class"], normalize="index")
```

建议图表：

```text
Superhost share by class
Boxplot of host_response_rate by class
Boxplot of calculated_host_listings_count by class
```

需要回答：

```text
Superhost 是否更容易出现在 Hidden Gem 中？
Host response rate 是否和评分或类别有关？
多房源房东是否更容易出现高价房源？
```

---

### 5.10 Correlation Analysis

建议变量：

```text
effective_price
calendar_price_volatility
calendar_available_rate
review_scores_rating
review_scores_value
review_scores_location
reviews_in_window
distance_to_nearest_subway_km
crime_intensity_log_1000m
amenity_count
host_response_rate
host_acceptance_rate
```

建议图表：

```text
Correlation heatmap
Selected scatter plots
Pair plots for key numeric variables
```

需要回答：

```text
价格和评分是否相关？
价格和地铁距离是否相关？
value score 和 effective price 是否负相关？
crime intensity 和价格、地铁距离是否相关？
哪些变量高度相关？
```

---

### 5.11 Outlier Analysis

重点检查：

```text
effective_price extremely high
calendar_price_volatility extremely high
distance_to_nearest_subway_km extremely high
crime_count_1000m extremely high
missing rating but high price
```

建议代码：

```python
df.sort_values("effective_price", ascending=False).head(20)
df.sort_values("calendar_price_volatility", ascending=False).head(20)
df.sort_values("distance_to_nearest_subway_km", ascending=False).head(20)
```

需要回答：

```text
是否需要 log transform？
是否需要 winsorization？
是否有经纬度错误？
是否有异常高价房源？
```

---

## 6. Unsupervised Learning Suggestions

课程项目需要体现 unsupervised learning 的使用。  
本项目可以将 unsupervised learning 用于 EDA 和 feature insight，而不是最终预测。

---

### 6.1 PCA

目的：

```text
降低维度，观察 listing 是否在消费者价值特征空间中自然分离。
```

建议特征：

```text
log_effective_price
calendar_available_rate
calendar_price_volatility
review_scores_rating
review_scores_value
reviews_in_window
distance_to_nearest_subway_km
crime_intensity_log_1000m
amenity_count
```

分析问题：

```text
Hidden Gem 和 Overpriced Trap 是否在 PCA space 中有分离？
哪些特征驱动主成分？
PCA 是否支持现有 label construction？
```

---

### 6.2 Clustering

建议模型：

```text
K-Means
Gaussian Mixture Model
Hierarchical Clustering
```

目的：

```text
发现 Airbnb listings 的自然分群。
```

可能的 cluster profile：

```text
Low price + high rating + good access
High price + central location
High price + weak value
Low activity listings
Transit-convenient budget listings
```

分析方式：

```python
pd.crosstab(df["cluster"], df["consumer_value_class"], normalize="index")
```

需要回答：

```text
自然 cluster 是否和 Hidden Gem / Trap 标签一致？
哪些 cluster 更像高价值房源？
哪些 cluster 更像高价低值房源？
```

---

## 7. Feature Leakage Notes

后续 supervised modeling 必须注意 feature leakage。

因为标签构造直接使用了：

```text
effective_price
review_scores_rating
review_scores_value
review_scores_location
distance_to_nearest_subway_km
crime_intensity_log_1000m
```

如果直接用这些变量预测 label，模型可能只是复现规则。

建议建模时准备两组 features：

---

### 7.1 Feature Set A: Rule-Reconstruction Baseline

可以包含直接标签构造变量。

目的：

```text
验证 label construction 是否逻辑一致。
```

---

### 7.2 Feature Set B: Reduced-Leakage Model

减少或排除直接构造标签的变量。

可使用：

```text
room_type
property_type
accommodates
bedrooms
beds
bathrooms
minimum_nights
maximum_nights
amenity_count
host_is_superhost
host_response_rate
host_acceptance_rate
host_identity_verified
calculated_host_listings_count
calendar_available_rate
calendar_price_volatility
weekend_price_premium
reviews_last_90d
reviews_last_180d
subway_stations_within_500m
subway_stations_within_1000m
crime_count_500m
violent_crime_count_1000m
property_crime_count_1000m
neighbourhood_cleansed
neighbourhood_group_cleansed
```

目的：

```text
检验在不直接使用标签构造变量的情况下，模型是否仍能识别消费者价值模式。
```

---

## 8. Suggested Modeling Direction

后续建模可以分为三个层次。

### Level 1: Baseline Model

任务：

```text
Predict consumer_value_class
```

建议模型：

```text
Multinomial Logistic Regression
Decision Tree
```

用途：

```text
提供简单可解释 baseline。
```

---

### Level 2: Stronger Supervised Models

建议模型：

```text
Random Forest
Gradient Boosting
XGBoost / LightGBM / CatBoost
```

评估指标：

```text
Accuracy
Macro F1
Weighted F1
Precision for Hidden Gem
Recall for Hidden Gem
Precision for Overpriced Trap
Recall for Overpriced Trap
Confusion Matrix
```

重点：

```text
不能只看 accuracy。
因为类别可能不平衡，Macro F1 和 per-class precision / recall 更重要。
```

---

### Level 3: Model Interpretation

需要输出：

```text
Feature importance
Permutation importance
SHAP values if available
Partial dependence plots if useful
```

重点回答：

```text
哪些特征最能区分 Hidden Gem？
哪些特征最能识别 Overpriced Trap？
价格、评分、交通、评论活跃度、crime intensity 中谁最重要？
Reduced-leakage model 是否仍有较好表现？
```
