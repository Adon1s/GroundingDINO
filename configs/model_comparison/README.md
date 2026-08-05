# Model-comparison profiles

Profiles hold non-secret model and run settings. The comparison script loads the
repository `.env` first without replacing variables already exported by the
shell. Credentials remain environment-only and are never written to reports.

Required cloud settings:

```dotenv
OPENAI_API_KEY=...
MODEL_COMPARISON_JUDGE_MODEL=gpt-5.4
MODEL_COMPARISON_FIXTURE_MODEL=gpt-5.4
```

`MODEL_COMPARISON_FIXTURE_MODEL` may be omitted to reuse the configured judge
model. Local contestant roles default to `unsloth/qwen3.6-27b@q6_k` and use
`LM_STUDIO_URL`. Set `MODEL_COMPARISON_LOCAL_MODEL` to replace that default when
a CLI provider override changes either contestant to `lmstudio`; an explicit
`--model-a` or `--model-b` takes precedence. This local default is never used as
a fallback for an OpenAI role or a failed OpenAI request.

## Common commands

```powershell
# Discover profiles and validate a fully resolved, redacted run.
.\.venv\Scripts\python.exe tools\model_comparison.py --list-profiles
.\.venv\Scripts\python.exe tools\model_comparison.py --profile sol_vs_terra --validate-config

# Premium versus premium.
.\.venv\Scripts\python.exe tools\model_comparison.py --profile sol_vs_terra --property redfin_126224899 --max-images 20

# Reusable mixed-provider profile with a one-off model override.
.\.venv\Scripts\python.exe tools\model_comparison.py --profile local_vs_premium --property redfin_126224899 --model-b gpt-5.6-sol

# Resume the newest checkpoint for the same property and profile.
.\.venv\Scripts\python.exe tools\model_comparison.py --profile sol_vs_terra --property redfin_126224899 --resume

# Run both forced judge positions over identical cached cells.
.\.venv\Scripts\python.exe tools\bias_check.py --profile sol_vs_terra --property redfin_126224899
```

Use `--skills` to select active cells, `--skip-skills` for a temporary exclusion,
and `--model-a`, `--provider-a`, `--model-b`, `--provider-b`, `--fixture-model`,
or `--judge-model` for one-off overrides. The resolved configuration and its
fingerprint are stored in the report; secret values are not.
