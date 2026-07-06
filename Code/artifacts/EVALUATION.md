# 评估报告 · Discovering London's Data

> 按 8 步评估方案，对已跑完的 enrichment pipeline 做诚实评估。
> 数据：`ckan_catalogue.json`（20,685 条）；嵌入：MiniLM（`all-MiniLM-L6-v2`）。
> 复现脚本：`scripts/evaluate_final.py`（量化指标）+ `evaluate_report.py`（明细）。
> 原始数据：`artifacts/evaluation_quant.json`、`artifacts/evaluation_full.json`。

---

## 第 1 步 · 数据质量检查（先看输入值不值得信）

| Check | Result | Meaning |
|---|---|---|
| 缺 tags | **92.5%**（19,128 / 20,685） | 打标签是刚需 |
| 缺 themes | **100%** | 主题生成是刚需 |
| 描述过短（notes < 40 字符） | 1.8% | 文本原料充足，NLP 可行 |
| 完全同名标题 | 178 组 / **378 条** | 重复检测有意义 |
| 来源倾斜 | **Croydon 占 72.9%**（共 144 个机构） | 结果可能被来源污染，必须在主题评估里查 |

**结论**：输入的**结构化元数据几乎为空**（痛点真实），但**正文文本充足**（方法可行）。同时来源高度倾斜——这是后面所有结果都要打的折扣。

---

## 第 2–3 步 · Baseline 对照 + 语义搜索 Precision@5

**对照设置（3 条线）**：
- `keyword` = 原始词频匹配（代表网站现状）
- `tfidf` = 更公平的词面 baseline（IDF 加权 + 长度归一）
- `semantic` = 我们的方法（MiniLM 向量余弦）

**相关性判定**：作者预定义的 concept 正则（对 title+notes），刻意包含**同义词/缩写**（如 NEET），词面搜索无法靠"对字面"取巧。8 个政策查询，Precision@5：

| 查询 | keyword | tfidf | **semantic** |
|---|---|---|---|
| young people not in work（NEET 测试） | 0.20 | **0.00** | **1.00** |
| energy efficiency retrofit of homes | 0.60 | 1.00 | 1.00 |
| air quality and pollution | 0.40 | 1.00 | 1.00 |
| affordable housing and homelessness | 0.00 | 1.00 | 1.00 |
| childhood obesity | 1.00 | 0.60 | 1.00 |
| domestic abuse against women | 0.80 | 1.00 | 1.00 |
| cycling and walking infrastructure | 0.00 | 0.80 | 1.00 |
| household recycling and waste | 0.80 | 1.00 | 1.00 |
| **平均 P@5** | **0.475** | **0.800** | **1.000** |

**关键发现（可放进 slide 的"惊艳时刻"）**：查询 `young people not in work`——
- keyword 只找到 1/5（被 COVID 等高词频数据集淹没）；
- **TF-IDF = 0/5**（"NEET / not in education" 与查询没有共同词，词面搜索必然失败）；
- **semantic = 5/5**，命中全部 "16–17 year olds not in education, employment or training (NEET)"。

这**同时证明了两件事**：语义搜索有真实增量；而 TF-IDF 兜底会当场丢掉这种增量——所以 **demo 必须用 MiniLM，不能降级到 TF-IDF**。

---

## 第 4 步 · 自动标签质量（抽样 + 自动代理指标）

| 指标 | 值 | 说明 |
|---|---|---|
| 含数字的标签 | **15.8%** | 明显噪声（如 `32`、`500`） |
| 纯数字标签 | 7.7% | 应过滤 |
| 每个标签平均词数 | 1.43 | 偏碎 |
| 无标签的数据集 | 0% | 覆盖率完整 |
| 30 条抽样中"具体且非噪声"标签占比 | **93.3%** | 大部分可用作**建议** |

**结论**：标签**可作建议**，但有 ~8–16% 的数字/噪声需过滤，**不能直接当官方标签**。

---

## 第 5 步 · 主题聚类（可解释性 / 稳定性 / 来源污染，不叫"准确率"）

- **规模**：HDBSCAN 自动得 **76 个主题**，35.4% 判为"无明确主题"（noise），silhouette **0.353**。
- **k 敏感性**：KMeans 扫 k=8–20，silhouette 仅 0.07–0.11、**无肘部** → 说明"球形硬分 k"不合适，改用 HDBSCAN 有据可依。
- **稳定性（ARI，跨随机种子）**：均值 **0.718**（1.0=完全一致，0=随机）→ 主题**中等偏稳**，不是随机产物，但也非铁板一块。
- **来源污染（最重要的自检）**：KMeans k=15 里 **12/15 个簇**由单一机构占比 ≥60%（多为 Croydon，个别簇 Croydon 占 97–99%）→ **部分"主题"其实是"机构"聚出来的**，不是纯语义主题。
- **可复现性 ≠ 正确性**：分类器还原簇标签 accuracy 0.976–0.998。**这不是主题准确率**——标签本身由同一批向量产生，属循环验证，只能证明"簇几何可分/可复现"。

**结论**：主题**可解释、中等稳定、几何可分**，但**受来源倾斜污染**，是**探索性分类**而非客观真理。

---

## 第 6 步 · 近重复检测（只能测 precision，测不了 recall）

| 阈值 | 检查的 top pairs | precision@50 |
|---|---|---|
| cosine ≥ 0.99 | 50 | **1.00** |
| cosine ≥ 0.97 | 50 | **1.00** |
| cosine ≥ 0.95 | 50 | **1.00** |

**诚实标注**：top-50 全部是**完全同名**的精确重复——precision 高，但这段主要复刻了"标题完全一致"。相似度阈值下探时对数暴增（≥0.90 有 84 万对），**近重复（非同名）的 precision 尚未在 top 段充分验证**。**recall 完全未知**（从未标注 2 万条里到底有多少真重复）。

**结论**：适合做**人工 review 队列**（高阈值 top pairs 基本可信），但**不能声称"找全了重复"**。

---

## 第 7 步 · skore 方法论检查

- **能用**：`skore.train_test_split` 跑通，并**主动触发一条真实方法论警告** `HighClassImbalanceTooFewExamplesWarning`——提示部分主题在测试集样本 <100，建议改用 `CrossValidationReport`。这正是 skore 该起的作用。
- **注意版本差异**：幻灯片里的 `report.checks.summarize()` 是**更新版 API**；本环境 skore **0.9.1** 没有 `report.checks`，方法论检查通过 `train_test_split` 的 warning + `EstimatorReport.metrics` 体现。
- **定位正确**：skore 检查的是**pipeline / 指标 / 方法论**，**不判断业务结果对不对**。特别地——**不要用 skore 的指标去给第 5 步那个循环的"主题分类器"背书**，那会弄巧成拙。

---

## 第 8 步 · 什么能信，什么不能完全信

### ✅ What we can trust
1. **数据痛点真实**：92.5% 无 tags、100% 无 themes（全量统计）。
2. **语义搜索有真实价值**：平均 P@5 **1.00 vs keyword 0.475**；NEET 案例 keyword/TF-IDF 均失败、语义 5/5。
3. **自动标签可作建议**：抽样 93% 具体可用（需过滤 ~8–16% 数字噪声）。
4. **高阈值重复对适合 review 队列**：top-50 precision = 1.00。
5. **方法选择有据**：k-sweep 无肘部 → 改 HDBSCAN；ARI 0.718 证明主题非随机。

### ⚠️ What we cannot fully trust
1. **自动 tags 不能直接当官方标签**：需人工审核 + 过滤数字。
2. **主题 cluster 不是客观真理**：依赖 embedding / k / metadata 文本，稳定性中等（ARI 0.718）。
3. **来源倾斜污染结果**：Croydon 72.9%，12/15 个簇由单一机构主导——部分"主题"实为"机构"。
4. **描述短/模糊的数据集效果差**：模型只能基于 metadata 文本。
5. **TF-IDF 语义能力有限**：NEET 案例 P@5=0，兜底会丢语义优势 → demo 用 MiniLM。
6. **"主题准确率 99.8%"是循环验证**：只证明可复现，**不证明正确**；应改称"主题稳定性/可分性"。
7. **重复检测测不了 recall**：不知道漏了多少真重复。
8. **评估未做盲标 / 多人标注**：相关性判定为作者单人 + 规则式 silver oracle，属定性/示例级证据，非统计显著。

---

## 一句话总结
**方向可信、且比多数队伍成熟。** 语义搜索的增量是**真实且可量化的**（P@5 1.00 vs 0.475）；标签与重复检测**适合做建议/review**而非终判；主题聚类**可解释但受来源污染**。最该守住的诚实边界：**别把"循环验证的 99.8%"当主题准确率，demo 别用 TF-IDF 兜底，主动说明 recall 与来源偏差未解。**
