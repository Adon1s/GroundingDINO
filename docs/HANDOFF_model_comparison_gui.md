# Handoff: Streamlit model-comparison GUI

## Goal

Add a separate Streamlit launcher/viewer for the provider-neutral comparison
engine. The CLI and `tools/model_comparison_config.py` remain the only sources
of truth; the GUI must not duplicate profile resolution, validation, checkpoint,
or comparison logic.

## Proposed implementation

Create `tools/model_comparison_gui.py` and launch it with:

```powershell
.\.venv\Scripts\streamlit.exe run tools\model_comparison_gui.py
```

The app should:

1. Call `list_profiles()`, `load_comparison_profile()`,
   `with_cli_overrides()`, and `ComparisonConfig.validate()` for profile
   selection and redacted preflight.
2. Show non-secret controls for property, image limit, active skills,
   concurrency, confirmation interval, model/provider overrides, output path,
   comprehensive judging, and resume/rejudge mode.
3. Never render, accept, persist, or pass API-key values. Show only whether
   required environment variables are present.
4. Build an argv list and launch `tools/model_comparison.py` with
   `subprocess.Popen`; do not use a shell command string. Merge stdout/stderr
   for a live log panel and retain the process handle in `st.session_state`.
5. Provide Start, Cancel, and Resume buttons. Cancel should send a graceful
   interrupt first, wait briefly, then terminate the child if needed. It must
   leave the CLI checkpoint untouched.
6. Disable configuration controls while a process is active. Surface non-zero
   exits, validation errors, missing checkpoints, and missing report files as
   explicit UI errors.
7. Load the completed JSON report and show:
   - model labels/providers/ids and fixture/judge roles;
   - per-skill wins, ties, and average score metrics;
   - recommendation and coupled-vs-isolated divergence;
   - per-role wall time and token/call statistics;
   - judge parse failures and missing verdicts;
   - an image selector with Model A/Model B cells and blinded verdict details.

Keep this app separate from `tools/artifact_viewer/app.py`: that viewer is tied
to `photo_intel.json` and requires different startup arguments. Shared display
helpers may be extracted later only if real duplication appears.

## Streamlit/runtime details

- Confirm Streamlit is installed in the project environment; add it to
  `requirements.txt` if the existing artifact viewer currently relies on an
  unrecorded global installation.
- Poll the subprocess on Streamlit reruns without blocking the UI thread.
- Store only argv, PID, timestamps, bounded log lines, and report/checkpoint
  paths in session state.
- Quote nothing manually: argv elements are passed directly to `Popen`.
- Run redacted `--validate-config` behavior in-process before enabling Start.

## Tests and acceptance

- Unit-test argv construction, redaction, profile overrides, report parsing,
  checkpoint discovery, and process-state transitions with mocked subprocesses.
- Verify API-key values cannot appear in widgets, argv, logs produced by the
  GUI, or saved UI state.
- Manually exercise OpenAI/OpenAI and LM Studio/OpenAI profiles, cancellation,
  resume, invalid environment configuration, and a completed report.
- Acceptance: a user can select `sol_vs_terra`, validate it, start a run, watch
  progress, safely cancel/resume it, and inspect the final quality and runtime
  comparison without opening a terminal or entering a secret.
