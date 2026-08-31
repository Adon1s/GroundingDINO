"""Streamlit front end for the Pass 2a prompt comparator.

Run:

    .venv\\Scripts\\streamlit.exe run tools\\pass2a_comparator\\app.py

A thin view. All generation happens in a subprocess (`runner.py`), all state on
disk (`storage.py`), all scoring in `review.py`. Nothing here calls a model, and
no prompt text is defined here - the production pair is imported from
`tools.scene_classifier_passes`, because a copy drifts silently and would show a
prompt the run never used.

API-key values are never rendered, stored in session state, or passed on argv;
only their presence is reported.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pass2a_comparator import config as cc  # noqa: E402
from tools.pass2a_comparator import review as rv  # noqa: E402
from tools.pass2a_comparator import runner as run_mod  # noqa: E402
from tools.pass2a_comparator import storage as store  # noqa: E402

LOG_LINES = 40
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

st.set_page_config(page_title="Pass 2a Prompt Comparator", layout="wide")


# ---------------------------------------------------------------------------
# Subprocess control
# ---------------------------------------------------------------------------

def _drain(stream: Any, log: Deque[str], progress: Dict[str, Any]) -> None:
    """Consume one child stream. Runs on a daemon thread.

    Writes only to the deque and dict handed in - never to st.session_state,
    which is not safe to touch from a background thread.
    """
    for line in stream:
        line = line.rstrip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            log.append(line)
            continue
        if event.get("type") == "progress":
            progress.update(event)
        elif event.get("type") == "stage":
            progress.clear()
            progress.update(event)
            log.append(f"-- stage: {event.get('stage')} --")
        else:
            log.append(line)


def start_run(argv: List[str]) -> None:
    """Launch the runner. argv list, never a shell string."""
    env = {**os.environ, **cc.budget_guard_env()}
    proc = subprocess.Popen(
        argv, cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    log: Deque[str] = deque(maxlen=LOG_LINES)
    progress: Dict[str, Any] = {}
    for stream in (proc.stdout, proc.stderr):
        threading.Thread(target=_drain, args=(stream, log, progress), daemon=True).start()
    st.session_state.run = {
        "proc": proc, "argv": argv, "pid": proc.pid,
        "log": log, "progress": progress, "started_at": store.now_iso(),
    }


def active_run() -> Optional[Dict[str, Any]]:
    run = st.session_state.get("run")
    if run and run["proc"].poll() is None:
        return run
    return None


def cancel_run() -> None:
    """Graceful stop first, then force. Checkpoints survive either way."""
    run = st.session_state.get("run")
    if not run:
        return
    proc = run["proc"]
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def runner_argv(experiment_id: Optional[str], baseline_only: bool) -> List[str]:
    argv = [str(VENV_PYTHON), "-m", "tools.pass2a_comparator.runner"]
    if experiment_id:
        argv += ["--experiment", experiment_id]
    if baseline_only:
        argv.append("--baseline-only")
    return argv


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _dataset() -> Any:
    photos, images_root = store.load_dataset()
    return [vars(p) for p in photos], str(images_root)


def load_context() -> Dict[str, Any]:
    raw_photos, images_root = _dataset()
    photos = [store.Photo(**p) for p in raw_photos]
    model_config = cc.build_model_config()
    fingerprint = store.compute_fingerprint(photos, model_config["model"])
    bid = store.baseline_id(fingerprint)
    base_dir = store.baseline_dir(bid)
    return {
        "photos": photos,
        "images_root": Path(images_root),
        "gold": store.load_gold(),
        "model_config": model_config,
        "fingerprint": fingerprint,
        "baseline_id": bid,
        "baseline_dir": base_dir,
        "baseline_done": (
            store.baseline_progress(base_dir, photos, fingerprint)
            if (base_dir / "fingerprint.json").is_file() else 0
        ),
    }


try:
    ctx = load_context()
except Exception as exc:  # noqa: BLE001 - surfaced in the UI, not the terminal
    st.error(f"Comparator cannot start: {type(exc).__name__}: {exc}")
    st.stop()

photos: List[store.Photo] = ctx["photos"]
total_calls = store.expected_call_count(photos)
running = active_run() is not None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Baseline")
    st.caption(f"`{ctx['baseline_id']}`")
    st.progress(ctx["baseline_done"] / total_calls,
                text=f"{ctx['baseline_done']}/{total_calls} production outputs")
    stale = store.stale_baselines(ctx["fingerprint"])
    if stale and ctx["baseline_done"] < total_calls:
        names = ", ".join(f"`{bid}`" for bid, _ in stale[:3])
        differs = sorted({key for _, keys in stale for key in keys})
        st.warning(
            f"The Pass 2a call path changed since {names}. Those baselines are "
            f"preserved, not lost - a new one is required because "
            f"**{', '.join(differs)}** differ."
        )
    if ctx["baseline_done"] < total_calls:
        st.button(
            f"Create fresh baseline - {total_calls - ctx['baseline_done']} calls",
            disabled=running, use_container_width=True,
            on_click=lambda: start_run(runner_argv(None, True)),
        )

    st.divider()
    st.header("Environment")
    st.write("OPENAI_API_KEY:", "present" if os.environ.get("OPENAI_API_KEY") else "**missing**")
    st.write("Budget guard:", "forced on for runs")
    st.caption(
        f"{ctx['model_config']['model']} - {ctx['model_config']['reasoning_effort']} "
        f"reasoning - {ctx['model_config']['max_output_tokens']} max output tokens"
    )
    if os.environ.get(run_mod.STUB_ENV):
        st.warning("STUB MODE - runs will not call a model.")

    st.divider()
    st.header("Experiments")
    experiments = store.list_experiments()
    labels = {
        e["experiment_id"]: (
            f"{e.get('display_name')} ({e['experiment_id'][:16]})"
            + (" [STUB]" if (e.get("fingerprint") or {}).get("stub") else "")
        )
        for e in experiments
    }
    selected_id = st.selectbox(
        "Open experiment", options=list(labels), format_func=lambda k: labels[k],
        index=0 if experiments else None, placeholder="no experiments yet",
    ) if experiments else None
    selected = store.load_experiment(selected_id) if selected_id else None
    if selected and st.button("Clone prompts into a new experiment",
                              use_container_width=True, disabled=running):
        st.session_state.draft_system = selected["candidate_system_prompt"]
        st.session_state.draft_user = selected["candidate_user_prompt"]
        st.session_state.draft_name = f"{selected.get('display_name')} (clone)"
        st.session_state.active_tab_hint = "Prompts"
        st.success("Prompts copied into the Prompts tab. Nothing was run.")


prompts_tab, run_tab, review_tab, report_tab = st.tabs(
    ["Prompts", "Run", "Review", "Report"]
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

with prompts_tab:
    production_system, production_user = cc.production_prompts()
    left, right = st.columns(2)
    with left:
        st.subheader("Production (read-only)")
        st.caption("Imported from `tools/scene_classifier_passes.py`, never copied.")
        st.text_area("System", production_system, height=120, disabled=True,
                     key="prod_system")
        st.text_area("User", production_user, height=160, disabled=True, key="prod_user")
    with right:
        st.subheader("Candidate")
        candidate_system = st.text_area(
            "System", st.session_state.get("draft_system", production_system),
            height=120, key="draft_system",
        )
        candidate_user = st.text_area(
            "User", st.session_state.get("draft_user", production_user),
            height=160, key="draft_user",
        )
        display_name = st.text_input("Name (optional)",
                                     st.session_state.get("draft_name", ""),
                                     key="draft_name")
        st.caption(
            "`{scene}` renders the frozen Pass 1a scene for each photo "
            "(exterior_front, kitchen, bedroom, ...). Costs no Pass 1a calls. "
            "Use `{{` for a literal brace."
        )
        if cc.uses_scene(candidate_system, candidate_user):
            st.warning(
                "Scene-conditional prompt. Production `run_pass_2a` is handed "
                "the scene but ignores it, so a winning result here also needs "
                "that wiring before it can ship."
            )

    unchanged = (candidate_system == production_system
                 and candidate_user == production_user)
    blank = not candidate_system.strip() or not candidate_user.strip()
    pending = total_calls + max(0, total_calls - ctx["baseline_done"])
    st.info(
        f"Starting this experiment issues **{pending} calls** "
        f"({total_calls} candidate"
        + (f" + {total_calls - ctx['baseline_done']} baseline" if ctx["baseline_done"] < total_calls else "")
        + f") across all {len(photos)} photos. Full-set runs only."
    )
    if blank:
        st.error("Both candidate prompts must be non-blank.")
    elif unchanged:
        st.warning("Candidate is identical to production - there is nothing to compare.")
    if st.button("Create experiment", type="primary", disabled=running or blank or unchanged):
        try:
            new_id, _ = store.create_experiment(
                system_prompt=candidate_system, user_prompt=candidate_user,
                display_name=display_name, baseline_id_value=ctx["baseline_id"],
                fingerprint=ctx["fingerprint"], model_config=ctx["model_config"],
            )
            st.success(f"Created `{new_id}`. Open it in the sidebar, then run it.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

with run_tab:
    if not selected:
        st.info("Create or select an experiment first.")
    else:
        exp_dir = store.experiment_dir(selected["experiment_id"])
        done = store.experiment_progress(exp_dir, photos, selected)
        st.subheader(selected.get("display_name") or selected["experiment_id"])
        st.caption(f"`{selected['experiment_id']}` - baseline `{selected['baseline_id']}`")

        cols = st.columns(3)
        cols[0].metric("Baseline outputs", f"{ctx['baseline_done']}/{total_calls}")
        cols[1].metric("Candidate outputs", f"{done}/{total_calls}")
        remaining = (total_calls - ctx["baseline_done"]) + (total_calls - done)
        cols[2].metric("Calls remaining", remaining)

        buttons = st.columns(2)
        label = "Resume" if (0 < done < total_calls or ctx["baseline_done"]) else "Start"
        buttons[0].button(
            f"{label} - {remaining} calls", type="primary",
            disabled=running or remaining == 0, use_container_width=True,
            on_click=lambda: start_run(runner_argv(selected["experiment_id"], False)),
        )
        buttons[1].button("Cancel", disabled=not running, use_container_width=True,
                          on_click=cancel_run)

        run = st.session_state.get("run")
        if run:
            progress = run["progress"]
            if progress.get("total"):
                st.progress(
                    min(1.0, (progress.get("done") or 0) / progress["total"]),
                    text=f"{progress.get('stage', '')} "
                         f"{progress.get('done', 0)}/{progress['total']}"
                         f" - {progress.get('photo', '')}",
                )
            if run["log"]:
                st.code("\n".join(run["log"]), language="text")
            code = run["proc"].poll()
            if code is None:
                st.caption(f"running - pid {run['pid']}")
                time.sleep(1.5)
                st.rerun()
            elif code != 0:
                st.error(f"Runner exited {code}. Checkpoints are intact; resume to continue.")
            else:
                st.success("Run complete.")


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

with review_tab:
    if not selected:
        st.info("Create or select an experiment first.")
    else:
        exp_id = selected["experiment_id"]
        exp_dir = store.experiment_dir(exp_id)
        base_dir = store.baseline_dir(selected["baseline_id"])
        sides = rv.assign_sides(exp_id, photos)
        review = rv.load_review(exp_dir)
        decisions = review["decisions"]
        revealed = rv.is_revealed(review)

        index = st.session_state.get("review_index", 0) % len(photos)
        photo = photos[index]

        head = st.columns([3, 1, 1])
        head[0].progress(len(decisions) / len(photos),
                         text=f"{len(decisions)}/{len(photos)} photos reviewed")
        head[1].button("Previous", use_container_width=True,
                       on_click=lambda: st.session_state.update(
                           review_index=(index - 1) % len(photos)))
        head[2].button("Next", use_container_width=True,
                       on_click=lambda: st.session_state.update(
                           review_index=(index + 1) % len(photos)))

        st.subheader(f"{photo.key}  -  {photo.scene}")
        image_col, gold_col = st.columns([2, 3])
        with image_col:
            path = store.image_path(photo, ctx["images_root"])
            if path.is_file():
                st.image(str(path), use_container_width=True)
            else:
                st.error(f"image missing: {path}")
        with gold_col:
            st.markdown("**Human gold - material conditions**")
            try:
                for entry in store.gold_for(ctx["gold"], photo):
                    st.markdown(f"- {entry['condition']}")
            except store.ComparatorDataError as exc:
                st.error(str(exc))

        try:
            side_a, side_b = rv.side_outputs(photo, sides[photo.key], base_dir, exp_dir)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")
            side_a = side_b = []

        if not any(t.strip() for t in side_a + side_b):
            st.warning("No outputs yet for this photo. Run the experiment first.")
        else:
            out_a, out_b = st.columns(2)
            for column, label, texts in ((out_a, "Side A", side_a), (out_b, "Side B", side_b)):
                with column:
                    st.markdown(f"### {label}"
                                + (f"  -  `{sides[photo.key] if label == 'Side A' else ('baseline' if sides[photo.key] == 'candidate' else 'candidate')}`"
                                   if revealed else ""))
                    for rep, text in enumerate(texts, start=1):
                        with st.expander(f"repeat {rep}", expanded=rep == 1):
                            st.markdown(text or "_no output_")

            existing = decisions.get(photo.key) or {}
            st.divider()
            verdict = st.radio(
                "Which reads better against the gold?", cc.BLIND_VERDICTS,
                index=(list(cc.BLIND_VERDICTS).index(existing["verdict"])
                       if existing.get("verdict") in cc.BLIND_VERDICTS else None),
                horizontal=True, key=f"verdict_{photo.key}",
            )
            tags = st.multiselect("Reason tags (optional)", cc.REASON_TAGS,
                                  default=existing.get("tags") or [],
                                  key=f"tags_{photo.key}")
            note = st.text_input("Note (optional)", existing.get("note") or "",
                                 key=f"note_{photo.key}")
            critical = st.checkbox(
                "Critical regression - the candidate produced something disqualifying",
                value=bool(existing.get("critical_regression")),
                key=f"crit_{photo.key}",
            )
            if st.button("Save verdict", type="primary", disabled=verdict is None):
                rv.record_decision(review, photo.key, verdict=verdict, tags=tags,
                                   note=note, critical_regression=critical)
                rv.save_review(exp_dir, review)
                st.session_state.review_index = (index + 1) % len(photos)
                st.rerun()

        st.divider()
        if revealed:
            st.success(f"Identities revealed {review['revealed_at']}. Verdicts stay "
                       f"editable; the blind record is preserved in the report.")
        elif rv.is_complete(review, photos):
            if st.button("Reveal identities", type="primary"):
                rv.reveal(review, photos)
                rv.save_review(exp_dir, review)
                st.rerun()
        else:
            st.info(
                f"Identities stay hidden until all {len(photos)} photos have a "
                f"verdict ({len(decisions)} so far)."
            )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

with report_tab:
    if not selected:
        st.info("Create or select an experiment first.")
    else:
        exp_id = selected["experiment_id"]
        exp_dir = store.experiment_dir(exp_id)
        review = rv.load_review(exp_dir)
        sides = rv.assign_sides(exp_id, photos)
        summary = rv.summarize(review, sides, photos)

        st.warning("**Not production-equivalent.** " + cc.PRODUCTION_EQUIVALENT_REASON)

        if not rv.is_revealed(review):
            st.info("Complete the review and reveal identities to see the summary.")
        else:
            counts = summary["counts"]
            cols = st.columns(4)
            for col, key in zip(cols, (rv.BETTER, rv.SAME, rv.WORSE, rv.UNCLEAR)):
                col.metric(key, counts[key])
            st.metric("Directional indicator", summary["directional"],
                      help=summary["directional_reason"])
            if summary["critical_regressions"]:
                st.error("Critical regressions: "
                         + ", ".join(summary["critical_regressions"]))
            if summary["post_reveal_revisions"]:
                st.caption("Revised after reveal: "
                           + ", ".join(summary["post_reveal_revisions"])
                           + ". The blind record is preserved in report.json.")

            st.divider()
            st.subheader("Final verdict")
            current = (review.get("final_verdict") or {})
            final = st.radio(
                "Authoritative verdict", cc.FINAL_VERDICTS,
                index=(list(cc.FINAL_VERDICTS).index(current["verdict"])
                       if current.get("verdict") in cc.FINAL_VERDICTS else None),
                horizontal=True,
            )
            final_note = st.text_area("Note (optional)", current.get("note") or "",
                                      height=80)
            if st.button("Save report", type="primary", disabled=final is None):
                rv.record_final_verdict(review, final, final_note)
                rv.save_review(exp_dir, review)
                rv.write_report(exp_dir, selected, review, sides, photos)
                st.success(f"Wrote {exp_dir / 'report.md'}")

            md = exp_dir / "report.md"
            if md.is_file():
                st.divider()
                st.markdown(md.read_text(encoding="utf-8"))
