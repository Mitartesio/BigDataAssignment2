# Scalable Data Processing
Yelp Reviews and Authenticity

# Big Data Analysis and Authenticity Study

This project explores large-scale text analysis on over **6.9 million Yelp reviews**, using PySpark and distributed computing. The objective is to investigate how authenticity-related language varies across cuisines, regions, and contexts — inspired by a 2019 study on how “authenticity” in food reviews reinforces cultural stereotypes.

---

## 1. Project Overview

The study focuses on examining:

- How often users apply authenticity-related words such as *“authentic”* or *“legitimate”*
- Whether certain cuisines receive more authenticity-focused reviews
- Whether authenticity language has **positive** or **negative** connotations depending on cuisine
- How geography (Northeast, Northwest, Southeast, Southwest USA) influences review patterns
- Whether the findings support earlier work showing biased or stereotyped use of authenticity terminology

Distributed computing via **PySpark** enables efficient filtering, transformation, and aggregation of millions of reviews.

---

## 2. Data Familiarization

Dataset exploration included:

- Counting total reviews: **6,990,280**
- Identifying businesses with 5-star averages and ≥500 reviews  
- Filtering users with >1000 reviews (“super users” or informal influencers)
- Finding businesses reviewed by ≥5 of these influencers  
- Extracting cuisine labels from the business dataset to support later aggregation

These basic explorations established the dataset’s breadth, popularity distribution across cuisines, and the presence of high-activity users.

---

## 3. Authenticity Language Analysis

Initial keyword filtering yielded:

- **124,634** reviews containing *“authentic”*
- **23,187** reviews containing *“legitimate”*
- Combined filtered reviews (including “legit”): **146,428**
- **Overall: 1.7%** of Yelp reviews mention “authentic,” while **0.3%** mention “legitimate”

Cuisine categories (Italian, French, Vietnamese, Caribbean, etc.) were extracted, but the method captured only the first listed category per business — a limitation acknowledged for future improvement.

### Authentic Reviews by Cuisine

When comparing the number of authenticity-tagged reviews to total cuisine reviews:

- Italian cuisine: **3.34%** authentic-related reviews  
- Caribbean cuisine: **9.08%** (≈3× higher)  
- Suggests differing uses of authenticity language that may align with cultural stereotypes identified in prior research

---

## 4. Geographic Segmentation

Using business latitude and longitude relative to the geographic midpoint of the US, the country was segmented into:

- Northeast  
- Northwest  
- Southeast  
- Southwest  

Results:

| Region     | Total Reviews | Authentic Reviews | % Authentic |
|------------|---------------|-------------------|-------------|
| Southeast  | 3,460,933     | 72,058            | 2.08%       |
| Southwest  | 1,211,163     | 22,815            | 1.88%       |
| Northwest  | 267,039       | 5,495             | 2.06%       |
| Northeast  | 2,051,145     | 46,060            | 2.25%       |

**Insight:**  
Geographical differences are small but hint at slightly higher authenticity-review usage in the Northeast. However, the approach likely oversimplifies regional variation, as each quadrant encompasses highly diverse states and urban areas.

---

## 5. Hypothesis Testing

The original hypothesis (from the 2019 study) suggests:

- Authenticity language is used **positively** for European cuisines  
- It is used **negatively** for cuisines associated with marginalized groups  
- These patterns reinforce cultural stereotypes

### Negative Phrasing (e.g., *“dirty”*, *“tacky”*, *“greasy”*, *“cheap”*)

**Highest % of negative-authentic reviews:**

| Cuisine     | % Negative |
|-------------|------------|
| Chinese     | 13.47%     |
| Vietnamese  | 11.78%     |
| Mexican     | 10.93%     |
| Japanese    | 10.20%     |
| French      | 10.72%     |

Insights:
- The cuisines most stereotyped in the original paper (**Mexican, Chinese**) again show high negative percentages.
- **French and Italian**, expected to be positively described in authenticity contexts, unexpectedly appear around 8–10% negative — contrary to the hypothesis.

### Positive Phrasing (e.g., *“elegant”*, *“refined”*, *“beautiful”*)

**Highest % of positive-authentic reviews:**

| Cuisine   | % Positive |
|-----------|------------|
| French    | 8.57%      |
| Thai      | 5.10%      |
| Italian   | 4.88%      |
| Mediterranean | 4.74%  |

**Lowest %:**

| Cuisine   | % Positive |
|-----------|------------|
| Mexican   | 2.16%      |
| Korean    | 2.20%      |
| Vietnamese| 2.33%      |
| Chinese   | 2.43%      |

Insights:
- European cuisines (French, Italian) show strong positive associations.
- Mexican and Chinese cuisines show the *lowest* positive-association rates.
- This **strongly supports the original hypothesis**: authenticity language is more flattering when applied to European cuisines.

---

## 6. Results & Insights

**Supported by findings:**

- Authenticity language is unevenly distributed across cuisines.
- European cuisines receive more **positive-authentic** descriptors.
- Several Asian, Latin American, and African cuisines receive more **negative-authentic** descriptors.
- Geographic segmentation shows minimal effect; cuisine-based differences dominate.
- Popular cuisines receive more total authenticity mentions, but percentages reveal underlying stereotype patterns.

**Key takeaway:**  
While not perfectly aligned with the original study, this large-scale dataset shows clear linguistic biases in how Yelp reviewers apply authenticity terminology across different cuisines.

---

## 7. Conclusion

This Big Data Analysis demonstrates that:

- Authenticity-related language reflects cultural biases in large-scale online revi
