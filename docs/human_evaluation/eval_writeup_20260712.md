# Four-Annotator Human Blind A/B Evaluation Write-up

## 一句话结果

我们用四位人工评价者对主实验可恢复的 aligned public-overlap 子集做了 source/reference-conditioned blind A/B preference evaluation。对 **528 条 MEPRS 与 strongest single-model baseline 输出不完全相同的样本**，四位评价者共给出 **2,112 个 non-identical annotator-item judgments**：

| Item | Count |
|---|---:|
| MEPRS wins | 1,022 |
| Strongest baseline wins | 750 |
| Human Tie | 340 |
| Two-sided sign test excluding ties | 1.117e-10 |

按 item-level majority vote 统计，MEPRS 胜 **270** 条，strongest baseline 胜 **181** 条，Tie/no-majority **77** 条；排除 Tie/no-majority 后，双侧 sign/binomial test 为 **p = 3.234e-05**。

另有 **348 条 aligned items** 中 MEPRS 与 baseline 输出完全相同，不送人工评价，直接计为 automatic Tie。

## 数据口径

正文主实验是：

```text
8 translation directions x 2 prompting strategies x 200 sentences
= 3,200 condition-level items
```

但 A/B 盲评需要 source、reference、Candidate A、Candidate B。当前能从公开 WMT/TEaR overlap 可靠恢复 source/reference 的 aligned subset 是：

```text
876 aligned condition-level items
```

其中：

```text
528 non-identical MEPRS-vs-baseline pairs -> sent to four human annotators
348 exact-identical pairs -> automatic Tie
```

所以论文主结果建议报告 528 条 non-identical human A/B 判断；876 只用于解释 aligned subset 总量和 automatic Tie 处理。

## 评价流程

每条样本包含：

```text
language_pair
strategy
source_text
reference_translation
candidate_A_text
candidate_B_text
```

标注表不显示哪个候选来自 MEPRS，哪个候选来自 strongest baseline。A/B 顺序按 item 随机/确定性打乱。四位评价者分别填写：

```text
preference: A / B / Tie
adequacy_A_1_5
fluency_A_1_5
adequacy_B_1_5
fluency_B_1_5
comments/reason optional
```

回收结果后再根据 hidden mapping 解盲，还原为 MEPRS win、baseline win 或 Tie。显著性检验采用排除 Tie 后的 two-sided sign/binomial test。

## Baseline 定义

baseline 不是固定某一个模型，而是每个 language-pair/prompt condition 下 BLEURT 平均分最高的单模型：

```text
condition-specific strongest single-model baseline
```

16 个 condition 里的 strongest baseline 分布：

| Baseline model | Conditions |
|---|---:|
| Claude-3.5-Sonnet | 10 |
| Claude-3-Opus | 4 |
| GPT-4o | 2 |

## 四位评价者分别结果

| Annotator file | MEPRS wins | Baseline wins | Ties | Two-sided sign test excluding ties |
|---|---:|---:|---:|---:|
| final_result_v2.csv | 283 | 202 | 43 | 2.723e-04 |
| final_result_v3.csv | 226 | 169 | 133 | 4.775e-03 |
| final_result_v4.csv | 273 | 187 | 68 | 7.072e-05 |
| final_result_v5.csv | 240 | 192 | 96 | 2.363e-02 |

## 合并统计

| Metric | Value |
|---|---:|
| Human annotators | 4 |
| Non-identical pairs per annotator | 528 |
| Non-identical annotator-item judgments | 2,112 |
| Human preferences for MEPRS | 1,022 |
| Human preferences for strongest baseline | 750 |
| Human Tie judgments | 340 |
| Two-sided sign test excluding ties | 1.117e-10 |

Item-level majority vote:

| Metric | Value |
|---|---:|
| Non-identical evaluated pairs | 528 |
| Majority favors MEPRS | 270 |
| Majority favors strongest baseline | 181 |
| Tie/no-majority | 77 |
| Two-sided sign test excluding tie/no-majority | 3.234e-05 |

## A/B 位置审计

在 528 条 non-identical pairs 中，MEPRS 作为 Candidate A 出现 **282** 次，作为 Candidate B 出现 **246** 次。按四位评价者合并判断：

| MEPRS position | MEPRS wins | Baseline wins | Ties |
|---|---:|---:|---:|
| MEPRS as A | 589 | 332 | 207 |
| MEPRS as B | 433 | 418 | 133 |

这说明 A/B 顺序已经随机化；MEPRS 在 B 位时仍略多于 baseline，但优势主要来自整体 pooled 和 majority-vote 结果。论文正文不一定需要展开这张表，但回复审稿人时可以说明 system identity was hidden and candidate order was randomized。

## 论文可写英文

```text
To complement the BLEURT-based results with a source/reference-conditioned human preference check, we conducted a four-annotator blind A/B evaluation on the aligned public-overlap subset of the main experiments. For each language-pair and prompting condition, MEPRS was compared against the strongest single-model baseline, defined as the individual model with the highest mean BLEURT score in that condition. Among 876 aligned condition-level items, 348 had text-identical MEPRS and baseline outputs and were therefore counted as automatic ties. The remaining 528 non-identical pairs were independently evaluated by four human annotators using anonymized and order-randomized Candidate A/B translations, together with the source sentence and reference translation. Across 2,112 non-identical annotator-item judgments, the annotators preferred MEPRS in 1,022 cases, preferred the strongest baseline in 750 cases, and returned 340 ties. Excluding ties, the preference for MEPRS was significant under a two-sided sign test (p = 1.12e-10). At the item level, majority voting favored MEPRS in 270 cases and the strongest baseline in 181 cases, with 77 tie or no-majority cases (two-sided sign test excluding tie/no-majority cases, p = 3.23e-05).
```

更保守的限制说明：

```text
This evaluation should be interpreted as a targeted four-annotator human preference check on the recoverable aligned public-overlap subset, not as a full-set human evaluation over all 3,200 condition-level items. Nevertheless, it uses source/reference-conditioned blind A/B comparisons with anonymized and order-randomized candidates across all language directions and both prompting strategies, providing additional evidence beyond BLEURT that MEPRS is more often preferred than the strongest condition-specific single-model baseline on non-identical aligned examples.
```

## 回复审稿人可写

```text
We agree that relying only on BLEURT is insufficient. In the revision, we added a four-annotator human blind A/B evaluation using source/reference-conditioned comparisons. The evaluation was sampled from the aligned public-overlap subset of the main experiments rather than from a separate single-direction set. We compared the MEPRS-selected output against the strongest single-model baseline for each language-pair/prompting condition. System identities were hidden from the annotators, and the A/B order was randomized before annotation. Of 876 aligned condition-level items, 348 had identical outputs from MEPRS and the strongest baseline and were counted as automatic ties. For the remaining 528 non-identical pairs, four annotators produced 2,112 non-identical annotator-item judgments after deblinding: MEPRS was preferred in 1,022 judgments, the strongest baseline in 750 judgments, and 340 judgments were ties. A two-sided sign test excluding ties showed a significant preference for MEPRS (p = 1.12e-10). At the item level, majority voting favored MEPRS in 270 cases and the strongest baseline in 181 cases, with 77 tie/no-majority cases (two-sided sign test excluding tie/no-majority cases, p = 3.23e-05). We report this as a targeted human preference evaluation on the recoverable aligned subset and distinguish it from a full evaluation over all 3,200 condition-level items.
```

## 不建议怎么写

不要再写旧模板或自动评测口径，例如：

```text
assisted automatic preference check
old single-rater human evaluation wording
```

因为当前最终口径是 **four-annotator human blind A/B evaluation**。

不要把这个实验写成完整 full-set 人工评价：

```text
We conducted a full human evaluation on all 3,200 condition-level items.
```

因为实际人工偏好判断的是 528 个非同文 aligned public-overlap pair；348 个完全相同 pair 是 automatic Tie，其余主实验样本没有进入这组人工 A/B 判断。

不要把 876 当主结果写成：

```text
Human annotators judged 876 pairs.
```

因为实际送人工判断的是 528 个非同文 pair；348 个完全相同 pair 是 automatic Tie。

## 本地文件

```text
final_result_v2.csv
final_result_v3.csv
final_result_v4.csv
final_result_v5.csv
final_summary_v2.csv
final_summary_v3.csv
final_summary_v4.csv
final_summary_v5.csv
```
