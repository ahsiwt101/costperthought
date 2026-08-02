"""
Streamlit Pareto-frontier explorer over results/summary.jsonl.

Run: streamlit run dashboard/app.py

Two modes:
  - "I have an accuracy target" -> shows the cheapest config ($/correct) that
    meets it, per benchmark.
  - "I have a $ budget per 1000 queries" -> shows the best accuracy
    achievable within that budget, per benchmark.

This is the reusable infra artifact: point it at any summary.jsonl produced
by harness/run_benchmark.py (any model, any grid) and it works, not just for
this project's specific sweep.
"""
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.jsonl")


@st.cache_data
def load_summary():
    if not os.path.exists(SUMMARY_PATH):
        return pd.DataFrame()
    rows = [json.loads(line) for line in open(SUMMARY_PATH)]
    df = pd.DataFrame(rows)
    if not df.empty:
        df[["weight_precision", "kv_cache_precision", "budget"]] = (
            df["cell"].str.extract(r"w-(.+)_kv-(.+)_budget-(.+)")
        )
    return df


st.set_page_config(page_title="CostPerThought", layout="wide")
st.title("CostPerThought - Pareto frontier explorer")
st.caption(
    "$ per correct answer vs. accuracy, across weight precision x KV-cache "
    "precision x reasoning-budget configs, measured on real GPU hardware."
)

df = load_summary()

if df.empty:
    st.warning(
        "No results yet. Run scripts/run_baseline.sh then scripts/run_sweep.sh "
        "on the GPU host, then re-launch this dashboard."
    )
    st.stop()

benchmarks = sorted(df["benchmark"].unique())
bench = st.selectbox("Benchmark", benchmarks)
sub = df[df["benchmark"] == bench].copy()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pareto frontier: accuracy vs. $/correct answer")

    # Flag degenerate configs (e.g. uncalibrated FP8 KV-cache that emits garbage).
    # Generic heuristic so this works for any model's summary, not just this sweep:
    # a config is "broken" if it produced zero correct answers (NaN $/correct) or
    # scores below half the best accuracy on this benchmark.
    best_acc = sub["accuracy"].max()
    sub["broken"] = sub["usd_per_correct_answer"].isna() | (sub["accuracy"] < 0.5 * best_acc)
    ok = sub[~sub["broken"]].copy()
    bad = sub[sub["broken"]].copy()

    fig = px.scatter(
        ok, x="usd_per_correct_answer", y="accuracy",
        color="weight_precision", symbol="kv_cache_precision",
        hover_data=["budget", "mean_wall_clock_seconds", "n"],
        labels={"usd_per_correct_answer": "$ per correct answer", "accuracy": "Accuracy"},
    )

    # Pareto frontier: sweeping cost low->high, keep only configs that raise accuracy.
    front = ok.sort_values("usd_per_correct_answer")
    keep, running_best = [], -1.0
    for _, r in front.iterrows():
        if r["accuracy"] > running_best:
            keep.append(r)
            running_best = r["accuracy"]
    if keep:
        f = pd.DataFrame(keep)
        fig.add_trace(go.Scatter(
            x=f["usd_per_correct_answer"], y=f["accuracy"],
            mode="lines+markers", name="Pareto frontier",
            line=dict(color="#2ca02c", width=2, dash="dash"),
            marker=dict(color="#2ca02c", size=11, symbol="circle-open"),
            hoverinfo="skip",
        ))

    # Broken configs with a finite cost: mark them unmissably.
    bad_plot = bad[bad["usd_per_correct_answer"].notna()]
    if not bad_plot.empty:
        fig.add_trace(go.Scatter(
            x=bad_plot["usd_per_correct_answer"], y=bad_plot["accuracy"],
            mode="markers+text", name="BROKEN (uncalibrated)",
            marker=dict(color="red", size=13, symbol="x"),
            text=["BROKEN"] * len(bad_plot), textposition="top center",
            textfont=dict(color="red", size=10),
            hovertext=bad_plot["cell"], hoverinfo="text",
        ))

    st.plotly_chart(fig, use_container_width=True)

    n_dead = int(sub["usd_per_correct_answer"].isna().sum())
    if n_dead:
        st.warning(
            f"{n_dead} config(s) produced **zero** correct answers "
            f"(uncalibrated FP8 KV-cache -> degenerate output), so $/correct is "
            f"undefined and they can't be placed on the axis at all - excluded from the plot."
        )

with col2:
    st.subheader("Query by constraint")
    mode = st.radio("I have...", ["an accuracy target", "a $ budget per 1000 queries"])
    if mode == "an accuracy target":
        target = st.slider("Minimum accuracy", 0.0, 1.0, 0.7, 0.01)
        candidates = sub[sub["accuracy"] >= target].sort_values("usd_per_correct_answer")
        if candidates.empty:
            st.error(f"No config in the sweep hits {target:.0%} accuracy on {bench}.")
        else:
            best = candidates.iloc[0]
            st.success(
                f"Cheapest config hitting {target:.0%}+: **{best['cell']}** "
                f"- accuracy {best['accuracy']:.1%}, ${best['usd_per_correct_answer']:.5f}/correct answer"
            )
    else:
        budget = st.number_input("$ per 1000 queries", min_value=0.0, value=10.0, step=1.0)
        sub["usd_per_1000_queries"] = sub["usd_per_query"] * 1000
        candidates = sub[sub["usd_per_1000_queries"] <= budget].sort_values("accuracy", ascending=False)
        if candidates.empty:
            st.error(f"No config in the sweep stays under ${budget:.2f}/1000 queries on {bench}.")
        else:
            best = candidates.iloc[0]
            st.success(
                f"Best accuracy under ${budget:.2f}/1000 queries: **{best['cell']}** "
                f"- accuracy {best['accuracy']:.1%}, ${best['usd_per_1000_queries']:.2f}/1000 queries"
            )

st.subheader("Full results table")
st.dataframe(sub.sort_values("usd_per_correct_answer"), use_container_width=True)
