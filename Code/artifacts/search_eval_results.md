# Semantic search evaluation — results

Relevance judged by hand on the pooled top-10 of each method (0=not, 1=somewhat, 2=highly). A result is *relevant* when grade >= 1. Metrics are the IR standards: Precision@5, Precision@10, MRR, nDCG@10.

## Headline: averaged over all queries

| Method | Precision@5 | Precision@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Keyword (current library) | 0.42 | 0.46 | 0.76 | 0.38 |
| TF-IDF (fair lexical baseline) | 0.75 | 0.68 | 0.85 | 0.56 |
| Semantic (ours) | 0.97 | 0.91 | 1.00 | 0.89 |

## Keyword baseline vs our semantic search (per query)

| Query | Method | Top-5 relevant | Precision@5 | Precision@10 | MRR |
|---|---|---:|---:|---:|---:|
| young people not in work | Keyword | 1/5 | 0.20 | 0.10 | 1.00 |
| young people not in work | Our method | 5/5 | 1.00 | 1.00 | 1.00 |
| air pollution near schools | Keyword | 2/5 | 0.40 | 0.40 | 1.00 |
| air pollution near schools | Our method | 5/5 | 1.00 | 1.00 | 1.00 |
| housing affordability by borough | Keyword | 2/5 | 0.40 | 0.60 | 0.50 |
| housing affordability by borough | Our method | 5/5 | 1.00 | 0.90 | 1.00 |
| cycling accidents in London | Keyword | 1/5 | 0.20 | 0.20 | 0.33 |
| cycling accidents in London | Our method | 5/5 | 1.00 | 0.80 | 1.00 |
| energy inefficient homes | Keyword | 2/5 | 0.40 | 0.60 | 0.25 |
| energy inefficient homes | Our method | 4/5 | 0.80 | 0.90 | 1.00 |
| population change in outer London | Keyword | 2/5 | 0.40 | 0.30 | 1.00 |
| population change in outer London | Our method | 5/5 | 1.00 | 0.90 | 1.00 |
| crime around transport stations | Keyword | 4/5 | 0.80 | 0.80 | 1.00 |
| crime around transport stations | Our method | 5/5 | 1.00 | 1.00 | 1.00 |
| access to green space | Keyword | 3/5 | 0.60 | 0.70 | 1.00 |
| access to green space | Our method | 5/5 | 1.00 | 0.80 | 1.00 |

## Full per-query metrics (all three methods)

| Query | Method | P@5 | P@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|
| young people not in work | keyword | 0.20 | 0.10 | 1.00 | 0.12 |
| young people not in work | tfidf | 0.60 | 0.30 | 1.00 | 0.25 |
| young people not in work | semantic | 1.00 | 1.00 | 1.00 | 0.99 |
| air pollution near schools | keyword | 0.40 | 0.40 | 1.00 | 0.40 |
| air pollution near schools | tfidf | 1.00 | 1.00 | 1.00 | 0.82 |
| air pollution near schools | semantic | 1.00 | 1.00 | 1.00 | 1.00 |
| housing affordability by borough | keyword | 0.40 | 0.60 | 0.50 | 0.38 |
| housing affordability by borough | tfidf | 1.00 | 1.00 | 1.00 | 0.95 |
| housing affordability by borough | semantic | 1.00 | 0.90 | 1.00 | 0.90 |
| cycling accidents in London | keyword | 0.20 | 0.20 | 0.33 | 0.09 |
| cycling accidents in London | tfidf | 1.00 | 1.00 | 1.00 | 0.53 |
| cycling accidents in London | semantic | 1.00 | 0.80 | 1.00 | 0.91 |
| energy inefficient homes | keyword | 0.40 | 0.60 | 0.25 | 0.43 |
| energy inefficient homes | tfidf | 0.20 | 0.20 | 0.33 | 0.14 |
| energy inefficient homes | semantic | 0.80 | 0.90 | 1.00 | 0.70 |
| population change in outer London | keyword | 0.40 | 0.30 | 1.00 | 0.28 |
| population change in outer London | tfidf | 1.00 | 0.80 | 1.00 | 0.87 |
| population change in outer London | semantic | 1.00 | 0.90 | 1.00 | 0.80 |
| crime around transport stations | keyword | 0.80 | 0.80 | 1.00 | 0.76 |
| crime around transport stations | tfidf | 0.40 | 0.50 | 0.50 | 0.34 |
| crime around transport stations | semantic | 1.00 | 1.00 | 1.00 | 1.00 |
| access to green space | keyword | 0.60 | 0.70 | 1.00 | 0.60 |
| access to green space | tfidf | 0.80 | 0.60 | 1.00 | 0.59 |
| access to green space | semantic | 1.00 | 0.80 | 1.00 | 0.83 |
