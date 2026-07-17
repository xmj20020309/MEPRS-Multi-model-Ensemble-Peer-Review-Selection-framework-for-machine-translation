# Latest-model A/B judge experiment

Scope: 528 non-identical aligned public-overlap A/B pairs.

Files:
- `input/model_judge_input_pairs_528.csv`: deblinded A/B input pairs used for judge calls.
- `raw_results/*_llm_ab_judge_results.csv`: raw per-item judge outputs.
- `summaries/latest_model_judge_summary.csv`: win/tie/sign-test summary per model.
- `summaries/human_model_agreement_summary.csv`: non-tie agreement with the four-human majority result.
- `src/latest_model_judges/fast_ab_judge_runner.py`: concurrent OpenAI-compatible judge runner.
- `src/latest_model_judges/api_concurrency_probe.py`: API concurrency probe used before the 528-run.

Do not distribute local `run_528_*.sh` launch scripts; they are intentionally not included.

Summary:
- Claude Opus 4.8: MEPRS 154, baseline 137, tie 237, p=0.348294233315.
- GPT-5.5: MEPRS 230, baseline 169, tie 129, p=0.00262255151397.
- Gemini 3.1 Pro Preview: MEPRS 174, baseline 151, tie 203, p=0.222278204287.

Agreement with human majority:
- Claude Opus 4.8: 241/267 (0.902622).
- GPT-5.5: 307/337 (0.910979).
- Gemini 3.1 Pro Preview: 271/297 (0.912458).
