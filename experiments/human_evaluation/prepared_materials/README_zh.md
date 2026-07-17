# 人工评价材料说明

- `human_eval_blind_sheet_en_he_tear_60.csv` 是给标注者使用的盲评表。
- `human_eval_answer_key_en_he_tear_60.csv` 是作者内部使用的答案键，不应发给标注者。
- 当前表格已填入候选译文，但 `source_text` 和 `reference_translation` 为空，因为当前 GitHub zip 中缺原始 WMT source/reference/sample ID。
- 建议优先补齐 source/reference 后再发给标注者；若时间特别紧，可以做 target-side fluency + preference 的临时评价，但说服力低于完整 source-conditioned adequacy 评价。
- 抽样策略：优先抽 MEPRS 与最佳单模型不同的 EN->HE TEaR 句子，目标 60 个 item。