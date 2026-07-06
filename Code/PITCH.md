# Pitch 讲稿 · Discovering London's Data

> 3 分钟路演脚本 + 问答准备。配合 canvas 面板一起讲。

## 30 秒开场（讲痛点）
"Data for London Library 有 **20,685 个数据集**，但几乎没法搜。我们量化了一下：
**100% 没有主题分类、92.5% 没有标签**，而且只能关键词搜索。
好数据被埋起来了——这就是我们要解决的。"

## 60 秒 Demo（讲效果，最关键）
> 配合 `artifacts/baseline_comparison.png` 一起讲。

"搜索改进必须有**对照组**。我们建了**两层 baseline**，和自己的方法**三层并排**——同一批查询、同一套语料（title + notes）、都看 Top-5：
- **① 关键词搜索**：模拟 Library 现状，只数词频；
- **② TF-IDF 词法检索**：公平的 lexical baseline，IDF 加权 + 余弦，排除长文本刷分；
- **③ 我们的语义搜索**：MiniLM 句向量 + 自动生成的主题/标签。

先看查询 `young people not in work`：
- **关键词**：Top-1 是 *London Labour Market Indicators*，后面全是**通勤距离 (TS058)**——字面沾边、实则跑偏；
- **TF-IDF**：进步到 *Young People / Children & Young People*——但只是**标题字面命中**，没抓到概念；
- **语义搜索**：直接返回 *16–17 岁 not in education, employment or training（NEET）*——**看懂了同义词**。用户根本不会去搜 NEET 这个缩写。

再看查询 `air quality and pollution`：
- **关键词**给你 *COVID-19 疫苗接种*（描述长、常见词多，被误排第一）；
- **TF-IDF 和语义搜索**都能召回 *Air Pollution Exposure in London*。

→ 这恰好说明：**现状最差、TF-IDF 是合理改进、语义在同义词/复杂查询上最稳**。我们不是只跟'瞎搜'比，而是跟一个**认真的 baseline** 比。"

> 配合 `artifacts/figures/search_metrics_summary.png` 一起讲。

**而且这不是嘴上说——我们量化了。** 8 个自然语言政策查询，把三种方法的 Top-10 **池化后人工判相关性(0/1/2)**，再算标准 IR 指标：
语义搜索 **Precision@5 = 0.97 · MRR = 1.00 · nDCG@10 = 0.89**，全面高于关键词(0.42 / 0.76 / 0.38)，也稳超公平的 TF-IDF(0.75 / 0.85 / 0.56)。逐查询看（`search_precision_by_query.png`）8 条里 7 条 P@5 = 1.0。

## 45 秒方案（讲怎么做的）
"我们用 NLP 自动丰富元数据：
1. 用 MiniLM 句向量把每个数据集编码（业界下载量第一的句向量模型）；
2. 用 **HDBSCAN 密度聚类**自动发现 **76 个细粒度主题**（糖尿病、道路事故、无家可归、性健康…），每个用 TF-IDF 关键词自动命名；
3. 自动给 92% 缺标签的数据集打标签；
4. 用相似度**标记**疑似重复数据集，进**人工评审队列**（≥0.97 有 29.5 万对，≥0.99 有 4.4 万对）——我们**只标记、不自动删除**；
5. 同一套向量直接支撑语义搜索。"

## 45 秒 "vouch for it"（讲可信，评委最看重）
"AI 能秒写 pipeline，但我们**能为它背书**——用 skore 评估：
- **搜索相关性是人工核过的，不是 AI 自评**：8 个政策查询、三方法 Top-10 池化、人工判分 0/1/2，语义搜索 **P@5 = 0.97、MRR = 1.00、nDCG@10 = 0.89**，远超关键词现状(0.42)、也稳超公平 TF-IDF(0.75)；判分文件 `relevance_judgments.json` 公开可改、一行命令复现；
- held-out 分类器能以 **99.8% 准确率、0.998 Macro-F1** 还原我们的主题 → 主题连贯、可信；
- 我们**没有随便拍 k**：扫描了 KMeans 的多个 k，silhouette 一直只有 0.07–0.12、没有肘部——所以改用 **HDBSCAN**，silhouette 提升到 **0.353**（3-4 倍），还自动定主题数；
- 我们**诚实报告**：35% 数据被判为"无明确主题"——这不是缺陷，是主动不硬塞，反而指出了最需要人工补元数据的那批数据；
- 我们主动查了**数据偏差**：Croydon 占 73%——这正是我们弃用 KMeans 的原因之一（k=15 时 **15 个簇里有 12 个被单一机构主导**，个别高达 97–100%）；HDBSCAN 缓解了这点，且我们把每个主题的主要来源摊开给大家看，不藏；
- 我们做了**稳定性检验**：换不同随机种子重跑，聚类一致性 **ARI 平均 0.72**——结果可复现，不是碰运气；
- 我们**评审了去重质量**：字面最相似的 top-50 对 **100% 是真重复**；再往下取 50 对"高度相似但不完全相同"的做人工评审，**0 对判错（100% 同主题）**，但其中只有 ~12% 是真/近重复、其余是"同一张表的不同切片"——所以我们**只标记给人工审、绝不自动删**（见 `dup_review.md` / `dup_review.png`）；
- 严格 25% 划分、按主题分层，**无数据泄漏**。"

## 收尾
"一句话：我们让伦敦 2 万个数据集**自己开口说话**，而且**每一步都能解释、能评估、能背书**。"

---

## 问答准备（评委可能问）
- **Q: 主题数（k）怎么定的？**
  A: 我们没拍脑袋。先扫了 KMeans 的 k=8~30，silhouette 一直 0.07–0.12、无肘部，说明球形划分不适合。于是改用 HDBSCAN（密度聚类，无需 k），它自动发现 76 个主题，silhouette 升到 0.353。参数一行就能复现（`scripts/analysis.py`）。
- **Q: 为什么用 all-MiniLM-L6-v2 这个模型？**
  A: 它是 Hugging Face 上下载量第一（月下载 2.4 亿+）的句向量模型，出自 Sentence-BERT（Reimers & Gurevych, EMNLP 2019）；在 10 亿句对上训练。选它是因为质量好、又能在笔记本上离线跑、无需 API——对政府数据很关键。
- **Q: 语义搜索是你们自创的吗？**
  A: 不是，这是业界标准的"稠密检索/向量搜索"，是所有 RAG 系统和谷歌/必应语义搜索的核心（奠基论文 DPR, Karpukhin et al. 2020；基准 BEIR/MTEB；工业库 FAISS）。我们把成熟技术用在了对的问题上。
- **Q: 你们凭什么说语义搜索"更相关"？会不会是 AI 自说自话？**
  A: 不是模型自评。我们准备了 8 个自然语言政策查询，把三种方法的 Top-10 **池化去重**后**人工判相关性(0/1/2，盲评)**，再算标准 IR 指标：语义 **P@5=0.97 / P@10=0.91 / MRR=1.00 / nDCG@10=0.89**，全面高于关键词(0.42/0.46/0.76/0.38)与 TF-IDF(0.75/0.68/0.85/0.56)。判分公开在 `relevance_judgments.json`（谁都能复核改分），整套一行命令复现：`python scripts/eval_semantic_search.py build|apply|score`。
- **Q: 你们的 baseline 是不是故意设弱？**
  A: 没有。我们给了**两层**对照：关键词搜索是 Library 的**真实现状**（不是造稻草人）；TF-IDF 余弦是 IR 领域**标准的公平词法 baseline**（IDF 加权、长度归一）。三层同 query、同语料、同 Top-5，结果全在 `evaluation_full.json`、一行命令可复现。语义搜索的增量主要在**同义词/缩写**（NEET ↔ not in work）和**抗长文本噪声**（COVID 误排）上——`air quality` 这类查询 TF-IDF 已经很好，我们照实说。
- **Q: 35% 未分类，是不是失败了？**
  A: 恰恰相反。HDBSCAN 主动把稀疏、孤立的数据标为"无明确主题"，而不是硬塞进桶里污染结果。这 35% 正是最需要人工补元数据的清单，是可交付的价值。
- **Q: 为什么 silhouette（0.353）不是接近 1？**
  A: silhouette 衡量几何紧致度，在 384 维语义空间里天然偏低；我们用监督式还原（99.8%）从另一角度证明主题确实可分，两个指标互补。
- **Q: Croydon 占 73% 会不会毁掉结果？**
  A: 会影响，我们量化了：KMeans k=15 时 15 个簇有 12 个被单一机构主导——这也是我们弃用 KMeans 的理由之一。HDBSCAN 缓解了这点，我们仍报告每簇来源分布；下一步可做按机构分层聚类或重加权。
- **Q: 结果稳定吗？换个随机种子会不会全变？**
  A: 做了稳定性检验：多个种子重跑，聚类一致性 ARI 平均 0.72，属于中等偏稳，说明主题结构是数据本身的规律，不是随机产物。
- **Q: 你们敢自动删重复吗？准确率如何？**
  A: 不自动删——我们**只标记，交人工确认**。评审结果：字面最相似的 top-50 对 100% 是真重复；在"高度相似但不完全相同"的 50 对评审带里，**0 对判错**（相似度几乎不会连到不相关数据），但真/近重复只占 ~12%，其余 88% 是"同一张普查表的不同切片"（不同年龄/性别/族裔/月份）——这些**必须保留**。正因如此才要 human-in-the-loop。只报精度，召回未知（2 万+ 数据未做完整重复标注）。
- **Q: 这个能真的部署到 Library 吗？**
  A: 能。产出就是标准化的 `enriched_catalogue.json`（含 theme + tags），可直接回填 CKAN；语义搜索是一个向量索引，MiniLM 可离线跑，无需外部 API。

## 交付物清单（artifacts/）
- `enriched_catalogue.json` — 回填 theme + tags 的目录
- `themes.md` — 76 个 HDBSCAN 主题 + 各主题来源
- `theme_confusion_matrix.png` — skore 主题连贯性可视化
- `baseline_comparison.png` — **三层对照 slide 图**（keyword / TF-IDF / semantic，定性）
- `figures/search_metrics_summary.png` — **语义搜索硬指标图**（P@5/P@10/MRR/nDCG，PPT 用）
- `figures/search_precision_by_query.png` — 逐查询 Precision@5 对比（PPT 用）
- `search_eval_results.md` / `.json` — P@5/P@10/MRR/nDCG 指标表
- `relevance_judgments.json` — 人工相关性判分（0/1/2，可复核）
- `search_examples.json` — 关键词 vs TF-IDF vs 语义搜索对比
- `near_duplicates.json` — 近重复数据集对
- `dup_review.md` / `dup_review_zh.md` — top-50 相似对的人工评审结果（四类分档）
- `dup_review.png` — 去重精度可视化（人工评审，非自动删除）
- `evaluation.json` / `evaluation_full.json` — 指标汇总（含 k 扫描、ARI 稳定性、去重规模）
