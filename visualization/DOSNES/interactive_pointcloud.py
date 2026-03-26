#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
]


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      background: #0f1115;
      color: #f4f4f4;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .page {{
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid #232733;
      padding: 20px 18px;
      background: #151923;
      overflow: auto;
    }}
    .sidebar h1 {{
      font-size: 18px;
      margin: 0 0 8px;
    }}
    .sidebar p {{
      font-size: 13px;
      line-height: 1.5;
      color: #b8bfcc;
    }}
    .sidebar .meta {{
      margin-top: 18px;
      font-size: 12px;
      color: #95a0b4;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 8px 0;
      font-size: 13px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex: none;
    }}
    #plot {{
      width: 100%;
      height: 100vh;
    }}
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <h1>{title}</h1>
      <p>Interactive 3D DOSNES point cloud. Rotate, zoom, and hover points to inspect labels and metadata.</p>
      <div class="meta">
        <div><strong>Input:</strong> {input_csv}</div>
        <div><strong>Points:</strong> {point_count}</div>
        <div><strong>Labels:</strong> {label_count}</div>
      </div>
      <div style="margin-top: 20px;">
        {legend_html}
      </div>
    </aside>
    <main>
      <div id="plot"></div>
    </main>
  </div>

  <script>
    const traces = {traces_json};
    const layout = {{
      paper_bgcolor: "#0f1115",
      plot_bgcolor: "#0f1115",
      font: {{ color: "#f4f4f4" }},
      margin: {{ l: 0, r: 0, b: 0, t: 40 }},
      title: {{ text: {title_json}, x: 0.03 }},
      legend: {{
        bgcolor: "rgba(0,0,0,0)",
        font: {{ size: 12 }}
      }},
      scene: {{
        bgcolor: "#0f1115",
        xaxis: {{ title: "", showbackground: false, showticklabels: false, zeroline: false }},
        yaxis: {{ title: "", showbackground: false, showticklabels: false, zeroline: false }},
        zaxis: {{ title: "", showbackground: false, showticklabels: false, zeroline: false }},
        camera: {{
          eye: {{ x: 1.5, y: 1.5, z: 1.25 }}
        }}
      }}
    }};
    const config = {{
      responsive: true,
      displaylogo: false,
      scrollZoom: true
    }};
    Plotly.newPlot("plot", traces, layout, config);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive 3D HTML point cloud from a DOSNES CSV.")
    parser.add_argument("--input_csv", type=Path, required=True, help="CSV file with x/y/z coordinates.")
    parser.add_argument("--output_html", type=Path, default=None, help="Output HTML path.")
    parser.add_argument("--title", type=str, default=None, help="Plot title.")
    parser.add_argument("--x_col", type=str, default="x", help="X column name.")
    parser.add_argument("--y_col", type=str, default="y", help="Y column name.")
    parser.add_argument("--z_col", type=str, default="z", help="Z column name.")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name.")
    parser.add_argument(
        "--hover_cols",
        type=str,
        nargs="*",
        default=None,
        help="Extra columns to show in hover. Defaults to all non-coordinate columns.",
    )
    parser.add_argument("--point_size", type=int, default=4, help="Marker size.")
    parser.add_argument("--opacity", type=float, default=0.85, help="Marker opacity.")
    return parser.parse_args()


def build_hover_text(row: pd.Series, columns: List[str]) -> str:
    parts = []
    for col in columns:
        value = row[col]
        if isinstance(value, str) and len(value) > 220:
            value = value[:217] + "..."
        parts.append(f"<b>{col}</b>: {value}")
    return "<br>".join(parts)


def build_traces(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], str]:
    traces = []
    legend_parts = []

    labels = df[args.label_col].astype(str).fillna("NA")
    unique_labels = labels.unique().tolist()
    color_map = {label: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, label in enumerate(unique_labels)}

    hover_cols = args.hover_cols
    if hover_cols is None:
        hover_cols = [col for col in df.columns if col not in {args.x_col, args.y_col, args.z_col}]

    for label in unique_labels:
        subset = df[labels == label]
        hover_text = [build_hover_text(row, hover_cols) for _, row in subset.iterrows()]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": label,
                "x": subset[args.x_col].tolist(),
                "y": subset[args.y_col].tolist(),
                "z": subset[args.z_col].tolist(),
                "text": hover_text,
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {
                    "size": args.point_size,
                    "opacity": args.opacity,
                    "color": color_map[label],
                },
            }
        )
        legend_parts.append(
            f'<div class="legend-item"><span class="swatch" style="background:{color_map[label]}"></span><span>{label} ({len(subset)})</span></div>'
        )

    return traces, "\n".join(legend_parts)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    required = [args.x_col, args.y_col, args.z_col, args.label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    title = args.title or f"Interactive DOSNES Point Cloud: {args.input_csv.stem}"
    output_html = args.output_html or args.input_csv.with_suffix(".html")
    output_html.parent.mkdir(parents=True, exist_ok=True)

    traces, legend_html = build_traces(df, args)
    html = HTML_TEMPLATE.format(
        title=title,
        title_json=json.dumps(title),
        input_csv=str(args.input_csv),
        point_count=len(df),
        label_count=df[args.label_col].nunique(),
        legend_html=legend_html,
        traces_json=json.dumps(traces, ensure_ascii=False),
    )

    output_html.write_text(html, encoding="utf-8")
    print(f"Wrote interactive HTML to: {output_html}")


if __name__ == "__main__":
    main()
