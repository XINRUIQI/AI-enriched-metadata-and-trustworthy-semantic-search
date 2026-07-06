"""Turn the human-labelled duplicate review sample into reports + a figure.

Reads artifacts/dup_review_sample.json (stratified, non-identical pairs), applies
the manual review labels below, and produces:
  * artifacts/dup_review.md            (English results table + narrative)
  * artifacts/dup_review_zh.md         (Chinese version)
  * artifacts/figures/dup_review.png   (two-panel visual for the pitch)
  * updates the p6_human_review block in artifacts/evaluation_quant.json

It also computes the "literal top-50" duplicate rate (INCLUDING byte-identical
titles) so we can honestly contrast the two regimes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_catalogue  # noqa: E402

ART = ROOT / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CATS = ["duplicate", "near_duplicate", "related", "wrong"]
CAT_LABEL = {
    "duplicate": "Duplicate",
    "near_duplicate": "Near-duplicate",
    "related": "Related but not duplicate",
    "wrong": "Wrong",
}
CAT_ZH = {
    "duplicate": "重复 (Duplicate)",
    "near_duplicate": "近重复 (Near-duplicate)",
    "related": "相关但非重复 (Related)",
    "wrong": "错误 (Wrong)",
}
CAT_COLOR = {
    "duplicate": "#6aa66a",
    "near_duplicate": "#9ecae1",
    "related": "#f0a35e",
    "wrong": "#d95f5f",
}

# --- Manual human-review labels for the 50 stratified non-identical pairs. ---
# Rationale: a pair is only duplicate/near-duplicate if it is the SAME indicator
# differing by a cosmetic label or a single statistical column (numerator/
# denominator, upper/lower confidence limit). Pairs that differ by a genuine
# data slice (a different age, sex, ethnic group, month, industry, measure) are
# "related but not duplicate" -- you must keep both. This is the whole reason we
# flag for human review instead of auto-deleting.
LABELS = {
    1: "near_duplicate",   # Stroke QOF: Lower vs Upper 95% CL of same indicator
    2: "near_duplicate",   # OST005: Denominator vs Numerator of same indicator
    3: "related",          # LRTI: under-1 vs age-1 (different age band)
    4: "near_duplicate",   # Pedestrians: Male vs Persons, same CL/age/indicator
    5: "duplicate",        # TS028 "Other identity only" vs "...: Other" (same)
    6: "related",          # UK Business Counts: 500-999 vs 1000+ size band
    7: "near_duplicate",   # Diabetes 0-9: Upper 95% vs Upper 99.8% CL
    8: "related",          # Redbridge payments: June 2014 vs July 2015 (month)
    9: "related",          # Census age: Female 33 vs 37
    10: "related",         # Census age: Male 25 vs 28
    11: "related",         # Census age: 31 vs 70
    12: "related",         # Pop projection: Females 35-39 vs 25-29 %
    13: "related",         # Country of birth: EU14 vs Non-EU European
    14: "related",         # Pop projection: Females 0-4 vs 18-24
    15: "near_duplicate",  # Life exp 65: Lower 95% CL vs Denominator, same slice
    16: "related",         # Projected pop: persons 30-34 vs females 70-74
    17: "related",         # Employment RQF4+: females vs males 16-64
    18: "related",         # Census age: Female 50 vs Male 12
    19: "related",         # Census age: Female 65 vs Male 32
    20: "related",         # Census age: 72 vs 96
    21: "related",         # ASHE jobs: Annual pay Male FT vs Weekly pay PT
    22: "related",         # Ethnic detailed: Ethiopian vs Somali (Black African)
    23: "related",         # Census age: Female 80 vs Male 12
    24: "related",         # ASHE jobs: Weekly pay Male FT vs Annual pay Male
    25: "related",         # ASHE: Median annual Male FT vs jobs weekly Female
    26: "related",         # Pop: Males 25-29 vs Males 0-19
    27: "related",         # UK Business Counts: Education vs Medium-sized band
    28: "related",         # Ethnic detailed: Somalilander vs Somali
    29: "related",         # Pop: Persons 16-64 % vs Persons 18 %
    30: "related",         # Ethnic detailed: Spanish vs Other White
    31: "related",         # Ethnic detailed: Indian vs Mixed Black British
    32: "related",         # Religion detailed: Druze vs Hindu
    33: "related",         # Pop projection: Females 35-44 vs Persons All
    34: "related",         # Pop projection: Persons 75-79 vs Males 0-15 %
    35: "related",         # ASCOF social care: carers payments vs info access
    36: "related",         # Ethnic detailed: Brazilian vs Moroccan
    37: "related",         # Life exp: at birth vs at 65, different decile/sex
    38: "related",         # Passports held: Europe vs Japan (different region)
    39: "related",         # Occupation: 222 Therapy vs 243 Business
    40: "related",         # Occupation: 412 Admin Finance vs 612 Animal Care
    41: "related",         # Redbridge payments: 2021-22 vs 2013-14 (period)
    42: "related",         # Census: age 77 vs sex-by-age 24
    43: "related",         # Industry: 28 Manufacture vs 52 Warehousing
    44: "related",         # Census: age 45 vs sex-by-age Male 68
    45: "related",         # KS1: maths White vs science (different subject)
    46: "related",         # Census: age 42 vs sex-by-age 74
    47: "related",         # Pop counts: Males 70 vs Females 45
    48: "related",         # LRTI admissions: infants 1yr Female vs 0-4 Male
    49: "related",         # Pop: Males 10-14 % vs Persons 14 %
    50: "related",         # Census: age 32 vs sex-by-age Female 82
}


def norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def literal_top50_duplicate_rate() -> dict:
    """Top-50 most similar pairs INCLUDING identical titles -> % exact duplicate."""
    df = load_catalogue(ROOT / "ckan_catalogue.json")
    X = np.load(ART / "doc_emb.npy").astype("float32")
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    X = X / n
    ntitles = [norm_title(t) for t in df["title"].fillna("")]
    pairs = []
    block = 1024
    nrows = X.shape[0]
    for start in range(0, nrows, block):
        end = min(start + block, nrows)
        sims = X[start:end] @ X.T
        for li, gi in enumerate(range(start, end)):
            row = sims[li]
            cand = np.where(row >= 0.99)[0]
            for j in cand:
                if j > gi:
                    pairs.append((gi, int(j), float(row[j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    top = pairs[:50]
    identical = sum(1 for i, j, _ in top if ntitles[i] == ntitles[j])
    return {"n": len(top), "identical_titles": identical,
            "duplicate_rate_pct": round(100 * identical / len(top), 1)}


def main() -> None:
    sample = json.loads((ART / "dup_review_sample.json").read_text())
    for rec in sample:
        rec["label"] = LABELS[rec["rank"]]
    (ART / "dup_review_sample.json").write_text(
        json.dumps(sample, indent=2, ensure_ascii=False))

    counts = {c: sum(1 for r in sample if r["label"] == c) for c in CATS}
    total = len(sample)
    dup_near = counts["duplicate"] + counts["near_duplicate"]
    on_topic = total - counts["wrong"]

    literal = literal_top50_duplicate_rate()

    # ---------------------------------------------------------------- English MD
    def pct(x):
        return f"{100 * x / total:.0f}%"

    lines = [
        "# P6 — Duplicate detection: human review of the top similarity pairs",
        "",
        "**Question:** when the model flags two datasets as near-duplicates via "
        "cosine similarity, is it right?",
        "",
        "## Method",
        "- Embed every dataset (all-MiniLM-L6-v2), L2-normalise, cosine similarity.",
        "- **Regime A (literal top-50):** the 50 most-similar pairs overall. These "
        "are dominated by byte-for-byte identical titles.",
        "- **Regime B (review band, N=50):** we drop pairs whose titles are "
        "*identical* (a trivial string match) and stratify-sample 50 pairs across "
        "the 0.86–1.00 similarity range. This is the realistic *human review "
        "queue*: highly similar but not identical.",
        "- Each pair labelled: **Duplicate** (same dataset) / **Near-duplicate** "
        "(same indicator, one statistical column differs) / **Related but not "
        "duplicate** (same table/theme, genuinely different slice) / **Wrong** "
        "(unrelated).",
        "",
        "## Regime A — literal top-50 most similar pairs",
        f"- **{literal['identical_titles']}/{literal['n']} "
        f"({literal['duplicate_rate_pct']}%)** have identical titles → true "
        "duplicates. At the very top of the ranking, duplicate detection is "
        "essentially perfect.",
        "",
        "## Regime B — review band (non-identical, N=50)",
        "",
        "| Pair type | Count | Share |",
        "| --- | --- | --- |",
        f"| Duplicate | {counts['duplicate']} | {pct(counts['duplicate'])} |",
        f"| Near-duplicate | {counts['near_duplicate']} | {pct(counts['near_duplicate'])} |",
        f"| Related but not duplicate | {counts['related']} | {pct(counts['related'])} |",
        f"| Wrong | {counts['wrong']} | {pct(counts['wrong'])} |",
        f"| **Total** | **{total}** | 100% |",
        "",
        "## Key findings (honest)",
        f"- **{on_topic}/{total} ({pct(on_topic)}) pairs are genuinely on-topic — "
        "0 were 'wrong'.** Embedding similarity almost never links unrelated "
        "datasets: precision on *\"these two belong together\"* is very high.",
        f"- But only **{dup_near}/{total} ({pct(dup_near)})** are true or near "
        "duplicates. In the non-identical band, the model mostly surfaces "
        "**related dataset families** — the *same* census table sliced by a "
        "different age, sex, ethnic group, month or measure.",
        "- Those related slices are **not** things you would delete — you need all "
        "single-year ages, both sexes, every ethnic category. This is exactly why "
        "we **flag likely duplicates for human review, never auto-delete.**",
        "",
        "## What to say in the pitch",
        "> \"Exact duplicates are flagged with ~100% precision. Below that, "
        "similarity groups datasets into families for a **human review queue** — "
        "our model *flags* likely duplicates for a human to confirm, it does "
        "**not** auto-remove anything. Across the top-50 review band, **0%** were "
        "wrong.\"",
        "",
        "_Note: precision only. Recall is unknown — the 20,685 datasets were never "
        "exhaustively labelled for duplicates._",
        "",
        "## Sample (review band, sorted by similarity)",
        "",
        "| # | sim | label | title A | title B |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in sample:
        lines.append(
            f"| {r['rank']} | {r['similarity']:.3f} | {CAT_LABEL[r['label']]} | "
            f"{r['title_a'][:60]} | {r['title_b'][:60]} |")
    (ART / "dup_review.md").write_text("\n".join(lines))

    # ---------------------------------------------------------------- Chinese MD
    zh = [
        "# P6 — 重复数据集检测：对 top 相似对的人工评审",
        "",
        "**问题：** 模型用 cosine similarity 把两个数据集判为近重复时，准不准？",
        "",
        "## 做法",
        "- 对每个数据集做句向量（all-MiniLM-L6-v2），L2 归一化后算 cosine 相似度。",
        "- **口径 A（字面 top-50）：** 全库相似度最高的 50 对——几乎全是标题完全相同的。",
        "- **口径 B（评审带，N=50）：** 去掉标题**完全相同**的对（那只是字符串匹配），"
        "在 0.86–1.00 相似度区间**分层抽样** 50 对。这才是真实的**人工评审队列**：高度相似但不完全相同。",
        "- 每对人工判为：**重复**（同一数据集）/ **近重复**（同一指标，只差一个统计列）/ "
        "**相关但非重复**（同一张表/主题，但确实是不同切片）/ **错误**（不相关）。",
        "",
        "## 口径 A — 字面 top-50 最相似对",
        f"- **{literal['identical_titles']}/{literal['n']} "
        f"（{literal['duplicate_rate_pct']}%）** 标题完全相同 → 真重复。"
        "在排名最顶端，重复检测基本 100% 准。",
        "",
        "## 口径 B — 评审带（不完全相同，N=50）",
        "",
        "| 类型 | 数量 | 占比 |",
        "| --- | --- | --- |",
        f"| 重复 Duplicate | {counts['duplicate']} | {pct(counts['duplicate'])} |",
        f"| 近重复 Near-duplicate | {counts['near_duplicate']} | {pct(counts['near_duplicate'])} |",
        f"| 相关但非重复 Related | {counts['related']} | {pct(counts['related'])} |",
        f"| 错误 Wrong | {counts['wrong']} | {pct(counts['wrong'])} |",
        f"| **合计** | **{total}** | 100% |",
        "",
        "## 关键结论（诚实版）",
        f"- **{on_topic}/{total}（{pct(on_topic)}）的对确实同主题——0 对'错误'。** "
        "向量相似几乎不会把不相关的数据集连到一起：'这两个属于一类'的精度非常高。",
        f"- 但只有 **{dup_near}/{total}（{pct(dup_near)}）** 是真重复/近重复。"
        "在'不完全相同'的相似度带里，模型主要找出的是**相关的数据集族**——"
        "同一张普查表按不同年龄/性别/族裔/月份/指标切出来的不同片。",
        "- 这些相关切片**不能删**——你需要每个单岁年龄、两种性别、每个族裔类别。"
        "这正是我们**只把疑似重复推给人工评审、绝不自动删除**的原因。",
        "",
        "## Pitch 里怎么讲",
        "> \"完全相同的重复，我们能以约 100% 精度标出来；再往下，相似度把数据集聚成'族'，"
        "进入**人工评审队列**——我们的模型只**标记**疑似重复交由人确认，**不会**自动删除任何东西。"
        "在 top-50 评审带里，**0%** 是判错的。\"",
        "",
        "_注：这里只报精度（precision）。召回未知——2 万多个数据集从未被完整标注过重复。_",
    ]
    (ART / "dup_review_zh.md").write_text("\n".join(zh))

    # ------------------------------------------------------------------- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    plt.rcParams.update({"axes.titleweight": "bold"})

    # Panel 1: regime A vs B duplicate rate
    a_dup = literal["duplicate_rate_pct"]
    b_dup = 100 * dup_near / total
    bars = ax1.bar(["Literal\ntop-50", "Review band\n(non-identical)"],
                   [a_dup, b_dup], color=["#6aa66a", "#9ecae1"], width=0.6)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("% true or near duplicate")
    ax1.set_title("P6  Duplicate precision:\ntop of ranking vs review band")
    ax1.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, [a_dup, b_dup]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                 ha="center", fontsize=11, fontweight="bold")

    # Panel 2: 4-category breakdown of the review band
    order = ["duplicate", "near_duplicate", "related", "wrong"]
    vals = [counts[c] for c in order]
    colors = [CAT_COLOR[c] for c in order]
    bars = ax2.barh([CAT_LABEL[c] for c in order][::-1], vals[::-1],
                    color=colors[::-1])
    ax2.set_xlabel("pairs (out of 50)")
    ax2.set_title("P6  Human review of top-50 similarity pairs\n"
                  f"(review band; 0 wrong → 100% on-topic)")
    ax2.grid(axis="x", alpha=0.25)
    ax2.set_xlim(0, max(vals) + 6)
    for b, v in zip(bars, vals[::-1]):
        ax2.text(v + 0.6, b.get_y() + b.get_height() / 2,
                 f"{v}  ({100*v/total:.0f}%)", va="center", fontsize=10)

    fig.suptitle("Near-duplicate detection — flagged for human review, not auto-deleted",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "dup_review.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------- update evaluation_quant
    eq_path = ART / "evaluation_quant.json"
    eq = json.loads(eq_path.read_text())
    eq["p6_human_review"] = {
        "regime_a_literal_top50": literal,
        "regime_b_review_band": {
            "n": total,
            "counts": counts,
            "dup_or_neardup": dup_near,
            "dup_or_neardup_pct": round(100 * dup_near / total, 1),
            "on_topic_pct": round(100 * on_topic / total, 1),
            "wrong": counts["wrong"],
            "sampling": "stratified across cosine bands 0.86-1.00, identical "
                        "titles excluded, seed=0",
        },
        "framing": "Model FLAGS likely duplicates for human review; it does NOT "
                   "auto-delete. Precision only; recall unknown.",
    }
    eq_path.write_text(json.dumps(eq, indent=2, ensure_ascii=False))

    print("counts:", counts)
    print(f"dup+near = {dup_near}/{total} ({100*dup_near/total:.0f}%), "
          f"on-topic = {on_topic}/{total} ({100*on_topic/total:.0f}%)")
    print(f"literal top-50 duplicate rate = {literal['duplicate_rate_pct']}%")
    print("wrote dup_review.md, dup_review_zh.md, figures/dup_review.png, "
          "updated evaluation_quant.json")


if __name__ == "__main__":
    main()
