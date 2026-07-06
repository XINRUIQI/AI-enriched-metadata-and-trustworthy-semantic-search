# 方案：让伦敦的 2 万个数据集"自己开口说话"

> London Data Week 2026 Hackathon · 题目三 *Discovering London's data*
> **How might we improve the Data for London Library search by enriching the metadata in its catalogue?**
> 技术方向：无监督学习 + NLP

---

## 1. 一句话主题（pitch 定位）

用 NLP 为 **100% 缺主题、92% 缺标签** 的 Data for London 目录**自动丰富元数据**，
并在此之上提供**语义搜索**，把"只能猜关键词"变成"用自然语言就能找到数据"。
—— 而且我们用 **skore** 证明结果**可评估、可解释、可背书**。

## 2. 我们证实的真实痛点（数据说话）

对 `ckan_catalogue.json`（20,685 条数据集）的量化：

| 痛点 | 现状 |
|---|---|
| 无标签 tags | 19,128 条 = **92%** |
| 无主题 groups/themes | 20,685 条 = **100%** |
| 完全重复标题 | 涉及 378 条记录（178 组） |
| 数据来源高度倾斜 | Croydon 独占 73%（→ 主题可能被污染，需在评估中检查） |
| 好消息 | 描述 `notes` 基本齐全（仅 1% 缺）→ NLP 有充足原料 |

## 3. 做哪些？主次与顺序

**三个都做**，共享同一套 embedding（一次向量化，多处复用）：

| 优先级 | 交付物 | 方法 | 对应题目 |
|---|---|---|---|
| ⭐ 主打 | **自动主题分类** | embedding → HDBSCAN 密度聚类（自动定主题数）→ 每簇 TF-IDF 命名 | 补 100% 缺失的 themes |
| ⭐ 主打 | **语义搜索 Demo** | 同一 embedding 做余弦相似度检索 | 直接"改进搜索"，现场最出彩 |
| ➕ 加分 | **自动打标签** | 每条文本 TF-IDF 抽 top 关键词 | 补 92% 缺失的 tags |
| ➕ 加分 | **近重复检测** | embedding 相似度找相似/重复数据集 | 解决"重复数据集"问题 |

**执行顺序（pipeline）：**

```
① 清洗      去 HTML/实体、拼 title+notes 为 search_text
② 向量化    本地 sentence-transformers 编码（TF-IDF 兜底）
③ 主题聚类  HDBSCAN 自动聚类（默认；KMeans 可选）→ 每簇命名 → 写回 theme
④ 自动打标签 TF-IDF 每条抽 top-5 关键词 → 写回 tags
⑤ 语义搜索  余弦相似度检索；与关键词搜索并排对比
⑥ skore 评估 silhouette + 主题连贯性报告 + 混淆矩阵可视化
⑦ 讲故事    "改进前 vs 改进后"，突出方法论
```

## 4. Embedding 用哪种？

- **主方案**：本地 `sentence-transformers` 的 **`all-MiniLM-L6-v2`**
  - ✅ 无需 API key、可离线、速度快、语义质量好
- **兜底**：`scikit-learn` 的 **TF-IDF**（零下载、极快，语义稍弱）
  - pipeline 自动降级，保证"永远能跑"，demo 不翻车

## 5. 用 skore 怎么评估（"can you vouch for it?"）

这是评分核心（评委最看重**方法论**）。两个角度：

1. **聚类质量（无监督、诚实）**：`silhouette_score` 看主题是否几何上可分。
2. **主题连贯性（监督式验证 + skore）**：用 embedding 训练分类器预测主题标签，
   在 **held-out 测试集**上看 F1——若 held-out 也能高分还原主题，说明主题**连贯、可信**。
   由 `skore.EstimatorReport` 输出指标、方法论检查、混淆矩阵可视化。
3. **语义 vs 关键词搜索**：真实查询并排对比，展示语义搜索召回了关键词漏掉的相关数据集。

**主动自检（vouch for it 清单）**：
- Croydon 占 73% 会不会污染主题？→ 按主题看来源分布
- 有没有用错字段 / 数据泄漏？→ 严格 train/test 划分
- 聚类数 k 选得对不对？→ 对比不同 k 的 silhouette
- 自动标签是否真相关？→ 抽样人工核对
- 语义搜索是否真优于关键词？→ 真实查询对比

## 6. 交付物（artifacts/）

- `enriched_catalogue.json` — 丰富后的目录（含 theme + auto_tags）
- `themes.md` — 发现的主题及各主题来源分布
- `theme_confusion_matrix.png` — skore 主题连贯性可视化
- `search_examples.json` — 关键词 vs 语义搜索对比
- `evaluation.json` — 指标汇总（silhouette、规模、改进前质量）

## 7. 技术栈

`Python 3.9` · `scikit-learn` · `sentence-transformers` · `skore` · `skrub` · `pandas` · `numpy` · `matplotlib`

## 8. 代码结构

```
src/data.py       加载 + 清洗 + 质量报告
src/enrich.py     embedding + 聚类 + 主题命名 + 自动打标签
src/search.py     关键词搜索 + 语义搜索
src/evaluate.py   skore 评估（silhouette + 主题连贯性 + 混淆矩阵）
run.py            端到端编排
```

## 9. 分工

- **人（你）**：讲故事 / pitch、拍板方向、用真实痛点打动评委
- **AI（我）**：写全部代码、跑 pipeline、生成 skore 报告与图、做 vouch 自检
