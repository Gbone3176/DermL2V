import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN

from visualization.plot_utils import draw_kde_envelopes


PALETTE = {"A": "#67a9cf", "B": "#ef8a8a", "C": "#8fc97a"}
OUTPUT_DIR = "/storage/BioMedNLP/llm2vec/visualization/kde_envelope_tests"


def _estimate_eps(coords):
    span = np.ptp(coords, axis=0)
    diag = float(np.linalg.norm(span))
    return max(diag * 0.08, 1e-3)


def _summarize_clusters(df, min_points, cluster_min_samples):
    rows = []
    for label, group in df.groupby("label"):
        coords = group[["x", "y"]].to_numpy(dtype=np.float32, copy=False)
        eps = _estimate_eps(coords)
        cluster_ids = DBSCAN(eps=eps, min_samples=cluster_min_samples).fit_predict(coords)
        unique_ids = sorted(set(cluster_ids.tolist()))
        drawn_clusters = 0
        ignored_small_clusters = 0
        noise_points = int((cluster_ids < 0).sum())
        for cluster_id in unique_ids:
            if cluster_id < 0:
                continue
            cluster_size = int((cluster_ids == cluster_id).sum())
            if cluster_size >= min_points:
                drawn_clusters += 1
            else:
                ignored_small_clusters += 1
        rows.append(
            {
                "label": label,
                "points": int(len(group)),
                "eps": round(eps, 4),
                "dbscan_clusters": int(sum(1 for x in unique_ids if x >= 0)),
                "drawn_kde_clusters": drawn_clusters,
                "ignored_small_clusters": ignored_small_clusters,
                "noise_points": noise_points,
            }
        )
    return pd.DataFrame(rows)


def _plot_case(df, title, output_path, min_points=20, cluster_min_samples=6):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    draw_kde_envelopes(
        ax=ax,
        plot_df=df,
        palette=PALETTE,
        min_points=min_points,
        cluster_min_samples=cluster_min_samples,
        thresh=0.25,
        alpha=0.18,
        bw_adjust=0.9,
    )
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        palette=PALETTE,
        s=65,
        alpha=0.9,
        edgecolor="white",
        linewidth=1.2,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def build_cases():
    rng = np.random.default_rng(42)

    case1_a = rng.normal(loc=[0.0, 0.0], scale=[0.22, 0.18], size=(180, 2))
    case1_b = rng.normal(loc=[1.4, 0.4], scale=[0.25, 0.20], size=(170, 2))
    case1 = pd.DataFrame(np.vstack([case1_a, case1_b]), columns=["x", "y"])
    case1["label"] = ["A"] * len(case1_a) + ["B"] * len(case1_b)

    case2_a_main = rng.normal(loc=[0.0, 0.0], scale=[0.22, 0.18], size=(180, 2))
    case2_b_main = rng.normal(loc=[1.3, 0.25], scale=[0.26, 0.22], size=(175, 2))
    case2_a_outliers = rng.normal(loc=[1.35, 0.22], scale=[0.025, 0.025], size=(6, 2))
    case2 = pd.DataFrame(
        np.vstack([case2_a_main, case2_b_main, case2_a_outliers]),
        columns=["x", "y"],
    )
    case2["label"] = ["A"] * len(case2_a_main) + ["B"] * len(case2_b_main) + ["A"] * len(case2_a_outliers)

    case3_a_1 = rng.normal(loc=[-0.9, 0.8], scale=[0.16, 0.16], size=(90, 2))
    case3_a_2 = rng.normal(loc=[0.1, -0.2], scale=[0.18, 0.16], size=(95, 2))
    case3_a_tiny = rng.normal(loc=[1.2, 0.95], scale=[0.03, 0.03], size=(7, 2))
    case3_b = rng.normal(loc=[1.2, 0.1], scale=[0.23, 0.20], size=(155, 2))
    case3 = pd.DataFrame(
        np.vstack([case3_a_1, case3_a_2, case3_a_tiny, case3_b]),
        columns=["x", "y"],
    )
    case3["label"] = (
        ["A"] * len(case3_a_1)
        + ["A"] * len(case3_a_2)
        + ["A"] * len(case3_a_tiny)
        + ["B"] * len(case3_b)
    )

    return [
        ("case1_clean_two_clusters", "Case 1: clean A/B clusters", case1),
        ("case2_a_outliers_inside_b", "Case 2: A outliers inside B cluster", case2),
        ("case3_a_two_major_islands", "Case 3: A has two major islands and one tiny island", case3),
    ]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = ["# KDE Envelope Test Report", ""]

    for case_name, title, df in build_cases():
        png_path = os.path.join(OUTPUT_DIR, "{}.png".format(case_name))
        csv_path = os.path.join(OUTPUT_DIR, "{}_summary.csv".format(case_name))
        _plot_case(df, title, png_path)
        summary_df = _summarize_clusters(df, min_points=20, cluster_min_samples=6)
        summary_df.to_csv(csv_path, index=False)

        report_lines.append("## {}".format(title))
        report_lines.append("")
        report_lines.append("- Plot: `{}`".format(png_path))
        report_lines.append("- Summary: `{}`".format(csv_path))
        report_lines.append("")
        report_lines.append(summary_df.to_csv(index=False).strip())
        report_lines.append("")

    report_path = os.path.join(OUTPUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(report_path)


if __name__ == "__main__":
    main()
