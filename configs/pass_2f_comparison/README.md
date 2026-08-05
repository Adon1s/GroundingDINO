# Pass 2f model comparison

This workflow consumes a completed local artifact and never reruns upstream
analysis. First run the normal analyzer with the standard local profile and
Pass 2f disabled:

```powershell
.\\.venv\\Scripts\\python.exe tools\\analyzer_cli.py ... `
  --analysis-profile standard --disable-2f
```

Then validate and compare the saved `photo_intel.json`:

```powershell
.\\.venv\\Scripts\\python.exe tools\\pass_2f_model_comparison.py `
  --profile terra_vs_sol_2f --run PATH_TO_RUN --validate-config

.\\.venv\\Scripts\\python.exe tools\\pass_2f_model_comparison.py `
  --profile terra_vs_sol_2f --run PATH_TO_RUN

.\\.venv\\Scripts\\python.exe tools\\pass_2f_model_comparison.py `
  --profile terra_vs_sol_2f --run PATH_TO_RUN --resume

.\\.venv\\Scripts\\python.exe tools\\pass_2f_model_comparison.py `
  --profile terra_vs_sol_2f --run PATH_TO_RUN --resume --executive-review
```

Multiple artifacts can be validated or compared sequentially with one `--run`
flag. All artifacts are prepared before the first provider call, and each
property retains its own report and checkpoint:

```powershell
.\\.venv\\Scripts\\python.exe tools\\pass_2f_model_comparison.py `
  --profile terra_vs_sol_2f `
  --run PATH_TO_RUN_1 PATH_TO_RUN_2 PATH_TO_RUN_3
```

For multiple runs, `--output` names an output directory. With `--resume`, an
existing checkpoint is reused for each property and properties that have not
started yet receive a new report normally.

Profiles contain model IDs and generation controls only. Credentials and
OpenAI endpoint overrides remain environment-only. The comparison never
patches or backs up the source artifact.
