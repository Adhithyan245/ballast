# Do AI scientists change their minds?

Measuring answer consistency and evidence grounding of LLMs on expert-labeled
biomedical questions (PubMedQA, PQA-L).

> Status: setup (Step 1). No results yet. This is a research exercise on public
> models and public data. It is not clinical advice or a medical evaluation.

## Research questions

1. **Consistency**: asked the same question 5 times, how often does a model's answer change (flip rate)?
2. **Confident errors**: how often is it wrong while stating confidence ≥ 80?
3. **Grounding**: given the abstract and asked to quote its evidence, do the quotes exist in the abstract?
4. **Fix**: does an abstention gate (abstain on run disagreement or missing quotes) reduce wrong answers, and at what coverage?
5. **Generality**: do the patterns hold across five model families?

## Data

- **Dataset**: PubMedQA expert-labeled subset (PQA-L, 1,000 questions; labels yes 552 / no 338 / maybe 110).
- **Source**: official repository [pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa), file `data/ori_pqal.json`,
  pinned to commit `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997` (2023-04-18). The download is verified against the
  file's git blob hash `38db7750761c78950ed32303e7545bdaa513390c`.
- **License**: MIT (repository license).
- **Citation**: Jin, Dhingra, Liu, Cohen, Lu. *PubMedQA: A Dataset for Biomedical Research Question Answering.* EMNLP-IJCNLP 2019.
- Labels (`final_decision`) are used as-is and never modified. The abstract shown to models is the `CONTEXTS`
  paragraphs with their section labels; `LONG_ANSWER` (the conclusion) is excluded because it usually states the answer.
- **Samples** (label-stratified, largest-remainder allocation, Python `random.Random`):
  pilot = 20 questions (seed 20261026; 11 yes / 7 no / 2 maybe), main = 100 questions drawn from the
  remaining 980 (seed 20261027; 55 / 34 / 11). Ids are in `data/samples/`.

## Models and conditions

See `study_models.json` for exact model names, providers, limits and the date each was verified.

| Family | Access | Temperature |
|---|---|---|
| Claude | Claude subscription via Claude Code headless mode (`claude -p`) | cannot be set (default only) |
| Gemini | Google AI Studio free tier | default, and 0 if quota allows |
| gpt-oss (OpenAI open weights) | Groq free plan | default, and 0 if quota allows |
| Mistral (open weights, self-hosted) | vLLM on a rented NVIDIA Brev GPU, paid with Brev credits; weights pinned by Hugging Face revision ([setup](docs/BREV_SETUP.md)) | default (model's generation config) and 0 |
| Qwen (open weights, self-hosted) | same Brev GPU, served one model at a time | default (model's generation config) and 0 |

The spec planned Mistral's free Experiment API plan. On 2026-10-26 the Mistral console showed API keys inactive on
the Free plan, and no free API plan was offered, so Mistral is represented by a self-hosted open-weight model instead.
Its results describe that open-weight model, not Mistral's hosted API.

Conditions: **closed-book** (question only) and **with-context** (question + abstract, must quote verbatim
supporting sentences). 5 runs per question × condition × model. Identical prompt text for every model
(`aisci/prompts.py`), sent as a single user message with no system prompt.

### Claude Code limitation

Claude results come from Claude Code headless mode, not the raw API. Claude Code adds its own system context,
temperature cannot be set, and calls share the subscription's usage limits with interactive use. Every call runs
from an empty directory with all tools, MCP servers, slash commands and session persistence disabled, and the
full JSON response is saved. The Claude Code docs state that `--bare` never reads subscription (OAuth)
credentials, so `--bare` cannot be used without an API key. Instead the calls use `--safe-mode` (no CLAUDE.md,
skills, plugins, hooks, MCP or memory, but normal login) and `--system-prompt` to replace Claude Code's default
system prompt. Whether the replacement took effect is checked from the input token count of the first live call
and reported. See `aisci/providers/claude_code.py` for the exact flags.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # or .venv/bin/pip
cp .env.example .env    # add your own keys; .env is git-ignored
python -m aisci download && python -m aisci sample
python -m aisci smoke                 # one test call per provider
python -m aisci run pilot             # resumable; re-run to continue after a quota stop
python -m aisci status pilot          # calls, missing and unparseable counts
python -m pytest -q
```

Raw responses are cached under `results/cache/<model>/<condition>/<qid>/run<n>.json`; nothing cached is ever
re-requested. Failed calls are logged to `logs/errors.jsonl` and counted as missing, never as answers.
