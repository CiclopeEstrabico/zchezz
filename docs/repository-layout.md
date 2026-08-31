# Repository Layout

```text
.github/                 CI and manually triggered workflows
.agents/                 cross-agent skills
.claude/                 Claude Code skill mirrors
docs/                    durable architecture and process documentation
engine/build/            shared build/bundle entry points
engine/c/zchezz_vXXX/    versioned engine source (current transition model)
engine/c/tools/           native selfplay/arena/tuning tools
engine/c/tests/           native invariant harnesses
openings/                 opening inputs when present
tests/                    correctness, integration, benchmark and game harnesses
tools/                    repository/release maintenance tools
train/                    NNUE data/training pipeline
utils/                    shared Python infrastructure
artifacts/                generated test/regression/release output; ignored
```

`tests/` and `train/` remain version-less. Engine-version discovery belongs to `utils/repo_paths.py`.

