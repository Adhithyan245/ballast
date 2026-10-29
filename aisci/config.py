"""Study-wide constants. Changing any of these after the main study starts is a
deviation and must be recorded in DEVIATIONS.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_DIR = DATA_DIR / "samples"
CACHE_DIR = ROOT / "results" / "cache"
USAGE_DIR = ROOT / "results" / "usage"
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"
LOG_DIR = ROOT / "logs"
# Empty directory that every `claude -p` call runs from, so no project context loads.
CLAUDE_SANDBOX_DIR = ROOT / ".claude_sandbox"

# PubMedQA expert-labeled subset (PQA-L), pinned to a commit of the official repo.
PUBMEDQA_REPO = "pubmedqa/pubmedqa"
PUBMEDQA_COMMIT = "1cbae8e92f72f20c8d3747cbb3bf5bc53554d997"  # master, 2023-04-18
PUBMEDQA_URL = (
    f"https://raw.githubusercontent.com/{PUBMEDQA_REPO}/{PUBMEDQA_COMMIT}/data/ori_pqal.json"
)
PUBMEDQA_FILE = RAW_DIR / "ori_pqal.json"
# Git blob SHA-1 of data/ori_pqal.json at that commit (from the GitHub contents API).
PUBMEDQA_GIT_BLOB_SHA = "38db7750761c78950ed32303e7545bdaa513390c"

SEED = 20261026
PILOT_SIZE = 20
MAIN_SIZE = 100
LABELS = ("yes", "no", "maybe")

RUNS_PER_ITEM = 5
CONFIDENT_THRESHOLD = 80  # confidence >= this counts as "high confidence"

# Condition keys: <context>_<sampling>
CONTEXTS = ("closed", "context")
SAMPLINGS = ("default", "t0")

# Pushback (sycophancy probe): a second turn after run PUSHBACK_RUN of each question at
# default sampling, stored as condition "<context>_pushback".
PUSHBACK_RUN = 1
