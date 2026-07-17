# 人工评议执行说明

日期：2026-06-24

## 推荐先做哪一份

优先标注：

- `revision_analysis/remaining_gap_audit_outputs/qualitative_error_case_blind_sheet.csv`

这份表包含 35 个 source/reference 已对齐的高分歧案例，每个 item 有 A/B 两个候选。它专门用于句子级 qualitative error analysis，能回应审稿人关于错误案例、BLEURT 误选和人工解释的意见。

可选补充：

- `revision_analysis/human_eval_materials/human_eval_blind_sheet_en_he_tear_aligned_60.csv`

这份是 EN->HE TEaR 的 60 条偏好评价表，但其中 31 条 A/B 候选文本完全相同，信息量较低。可以作为补充材料，不建议作为唯一人工评议材料。

不要发给标注者：

- `qualitative_error_case_answer_key.csv`
- `human_eval_answer_key_en_he_tear_aligned_60.csv`

这两份是作者内部答案键，会暴露 MEPRS / baseline / BLEURT-best 身份。

## 标注字段

每个 item 的 A/B 两行都需要填写：

- `adequacy_1_5`：相对 source/reference 的忠实度，1 最差，5 最好。
- `fluency_1_5`：目标语言流畅度，1 最差，5 最好。
- `overall_preference_A_B_Tie`：只填 `A`、`B` 或 `Tie`。
- `error_type`：可选。建议使用 `incomplete`、`mistranslation`、`omission`、`addition`、`terminology`、`fluency`、`style`、`other`。
- `comments`：可选，简短说明判断原因。

## 最小人力方案

- 最低限度：1 名双语标注者完成 35 个分歧案例。
- 更稳妥：2 名双语标注者独立完成同一份表，之后统计一致性和分歧。
- 若时间非常紧：先完成 `overall_preference_A_B_Tie` 和 `comments`，adequacy/fluency 分数可后补。

## 统计方式

标注完成后运行：

```bash
python revision_analysis/score_human_eval.py \
  --sheet revision_analysis/remaining_gap_audit_outputs/qualitative_error_case_blind_sheet.csv \
  --key revision_analysis/remaining_gap_audit_outputs/qualitative_error_case_answer_key.csv \
  --out_dir revision_analysis/remaining_gap_audit_outputs/human_scored_qualitative_cases
```

输出：

- `human_eval_summary.csv`：MEPRS / BLEURTBest 偏好数量、平均 adequacy/fluency。
- `human_eval_item_preferences.csv`：逐 item 偏好结果。
- `human_eval_joined_rows.csv`：盲评表与答案键合并后的内部分析表。

## 回复信中的保守写法

如果还没收集人工标注：

> We prepared a blind qualitative error-analysis sheet for 35 source/reference-aligned disagreement cases where MEPRS and the sentence-level BLEURT-best candidate select different, non-identical translations. The sheet is ready for human annotation, but manual labels have not yet been collected.

如果完成 1 名标注者：

> As a small qualitative check, one bilingual annotator reviewed 35 source/reference-aligned disagreement cases. We report the preference counts and error categories as exploratory qualitative evidence rather than a full-scale human evaluation.

如果完成 2 名以上标注者：

> Two bilingual annotators independently reviewed 35 source/reference-aligned disagreement cases. We report preference counts, adequacy/fluency scores, and disagreements, and treat this as a focused qualitative error analysis rather than a replacement for a large-scale human evaluation.
