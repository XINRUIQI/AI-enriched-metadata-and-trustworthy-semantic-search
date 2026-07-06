# 数据质量检查 · Data Quality Check

> 第一步：先确认输入值不值得信。若 metadata 本身很差，后续 NLP 结果也不能完全信。
> 数据来源：`artifacts/evaluation_full.json` 的 `p1_quality`（由 `evaluate_report.py` 对 `ckan_catalogue.json` 全量计算，20,685 条数据集）。

| Check | Result | Meaning |
| --- | --- | --- |
| Missing tags | 92.5%（19,128 / 20,685 条无 tags） | Tags enrichment is necessary |
| Missing themes | 100%（全部无 groups/theme） | Theme generation is necessary |
| Short / missing notes | 1.8%（notes 长度 < 40 字符） | Text field is usable for NLP |
| Duplicate titles | 378 条记录（178 组完全重复标题） | Duplicate detection is useful |
| Source imbalance | Croydon 独占 72.9%（共 144 个来源机构） | Results may be biased by source distribution |

## 口径说明

- **Missing tags / themes**：基于 CKAN 记录的 `tags` / `groups` 字段是否为空。
- **Short / missing notes**：并非严格的"缺失"，而是 `notes`（描述）清洗后长度 < 40 字符的占比，用来衡量可供 NLP 使用的正文原料是否充足——仅 1.8%，说明绝大多数记录有足够文本。
- **Duplicate titles**：把 `title` 去空格并转小写后统计完全一致的分组，得到 178 组、共 378 条记录（之前 PLAN 中的 372 为早期估算，已统一为 378）。
- **Source imbalance**：`organization` 出现频次最高的是 Croydon，占 72.9%，需在主题评估时检查是否有簇按机构而非主题聚集。
