# Deviations

Changes to the study plan, with dates and reasons. Section A covers changes from the original
project spec, decided before any pilot or main-study call. Changes made after PREREGISTRATION.md
is committed go in section B and never edit the preregistration itself.

## A. Before the pilot (from the original spec)

| Date | Spec said | Changed to | Reason |
|---|---|---|---|
| 2026-10-26 | Claude via `claude -p --bare` | `--safe-mode` plus `--system-prompt ""` (other isolation flags unchanged) | Claude Code docs: bare mode never reads subscription (OAuth) credentials; a test call with `--bare` returned "Not logged in". Measured on 2026-10-27: the override cuts Claude Code's added context from about 5,900 to about 1,100 tokens; the rest cannot be removed. |
| 2026-10-26 | Open-weight model on Groq, e.g. Llama 3.3 70B | `openai/gpt-oss-120b` on Groq | The live Groq list had no Llama 3.3 70B. |
| 2026-10-26 | Gemini (model unspecified) | `gemini-3.5-flash-lite` | Free-tier Flash models allow 20 requests/day (10+ days for the pilot alone); Flash-Lite allows 500. |
| 2026-10-27 | 4 families including Mistral via the free Experiment plan | **Mistral dropped**; Qwen (`qwen/qwen3.8-27b` on Groq) added, so there are still 4 families: Claude, Gemini, gpt-oss, Qwen | Mistral's console showed API keys inactive on the Free plan, with no free API plan offered (2026-10-26). NVIDIA's hosted API listed Mistral models, but every chat call returned 404 "not found for account" (2026-10-27). Self-hosting on NVIDIA Brev was built but is disabled: no usable credits. |
| 2026-10-27 | Claude model chosen explicitly | `claude-sonnet-5` | The `sonnet` alias resolved to it on CLI 2.1.283; the CLI did not recognize `claude-sonnet-5-5`. |
| 2026-10-29 | 2 conditions (closed-book, with-context); consistency + grounding only | **Added a pushback (sycophancy) condition**: one extra turn after run 1 at default sampling, pushing toward a different answer, plus a combined abstention gate that also requires the model to hold its answer | Decided by the study owner before the preregistration was frozen and before any main-study call. Piloted on the 20 pilot questions: 160 pushback replies across 4 models, 0 unparseable. |

Code for the dropped routes stays in the repo (`mistral`, `mistral_nvidia`, `qwen_vllm`,
`mistral_vllm` in `study_models.json`, all `"enabled": false` with reasons) so the attempts can be
checked.

## B. After the preregistration

| Date | Preregistration said | Change | Reason and status |
|---|---|---|---|
| 2026-10-30 | 4 models: Claude, Gemini, gpt-oss, Qwen | **Added DeepSeek (`deepseek-3.2`) via the Kiro CLI as an exploratory fifth family.** It gets the same prompts, conditions (default sampling + pushback; temperature cannot be set), parser, metrics and gate as the others. | Decided by the study owner after the freeze, while the main study was running, because Kiro credits became available. It is **not** part of the 20 preregistered primary tests or the H5 generality rule; all its results are labelled exploratory. Same limitation as Claude: Kiro is an agent harness that adds its own context. Piloted on the 20 pilot questions before its main run. No main-study metric of any model had been computed when this was decided. |
