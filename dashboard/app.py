from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    st.set_page_config(page_title="Camera Surveillance AI", page_icon="📹", layout="wide")
    st.title("Camera Surveillance AI")
    st.caption("Dataset, evaluation, and benchmark dashboard")

    with st.sidebar:
        evaluation_path = Path(st.text_input("Evaluation report", "reports/metrics/evaluation/evaluation_summary.json"))
        benchmark_path = Path(st.text_input("Benchmark report", "reports/metrics/benchmark.json"))
        dataset_path = Path(st.text_input("Dataset report", "reports/metrics/dataset_statistics.json"))

    evaluation = load_json(evaluation_path)
    benchmark = load_json(benchmark_path)
    dataset = load_json(dataset_path)

    if evaluation:
        st.subheader("Model evaluation")
        metrics = evaluation.get("metrics", {})
        columns = st.columns(4)
        columns[0].metric("Precision", f"{metrics.get('precision', 0):.3f}")
        columns[1].metric("Recall", f"{metrics.get('recall', 0):.3f}")
        columns[2].metric("mAP@50", f"{metrics.get('map50', 0):.3f}")
        columns[3].metric("mAP@50–95", f"{metrics.get('map50_95', 0):.3f}")
        if evaluation.get("per_class"):
            st.dataframe(pd.DataFrame(evaluation["per_class"]), use_container_width=True, hide_index=True)

    if benchmark:
        st.divider()
        st.subheader("Runtime benchmark")
        latency = benchmark.get("latency_ms", {})
        columns = st.columns(4)
        columns[0].metric("Mean latency", f"{latency.get('mean', 0):.1f} ms")
        columns[1].metric("P95 latency", f"{latency.get('p95', 0):.1f} ms")
        columns[2].metric("FPS", f"{benchmark.get('throughput_fps', 0):.1f}")
        columns[3].metric("Inferences", benchmark.get("total_inferences", 0))

    if dataset:
        st.divider()
        st.subheader("Dataset statistics")
        rows = []
        for split_name, values in dataset.get("splits", {}).items():
            rows.append({"Split": split_name, "Images": values.get("image_count", 0), "Labels": values.get("label_count", 0), "Annotations": values.get("annotation_count", 0)})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if not any((evaluation, benchmark, dataset)):
        st.warning("No generated reports were found yet.")


if __name__ == "__main__":
    main()
