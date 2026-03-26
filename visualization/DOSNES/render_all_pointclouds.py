#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid #232733;
      padding: 22px 20px;
      background: #151923;
      overflow: auto;
    }}
    .sidebar h1 {{
      font-size: 20px;
      margin: 0 0 10px;
    }}
    .sidebar p {{
      margin: 0 0 16px;
      font-size: 13px;
      line-height: 1.5;
      color: #b8bfcc;
    }}
    .meta {{
      margin-top: 16px;
      padding: 12px;
      border: 1px solid #232733;
      border-radius: 10px;
      background: #10141d;
      font-size: 12px;
      color: #95a0b4;
      line-height: 1.6;
    }}
    .control-label {{
      display: block;
      margin: 18px 0 8px;
      font-size: 12px;
      color: #95a0b4;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    select {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #2a3140;
      background: #0f1115;
      color: #f4f4f4;
      font-size: 13px;
    }}
    .legend {{
      margin-top: 18px;
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
      <p>Interactive 3D DOSNES gallery. Use the selector to switch between all previously generated CSV results in a single HTML page.</p>

      <label class="control-label" for="dataset-select">Experiment</label>
      <select id="dataset-select"></select>

      <div class="meta">
        <div><strong>CSV:</strong> <span id="meta-path"></span></div>
        <div><strong>Points:</strong> <span id="meta-points"></span></div>
        <div><strong>Labels:</strong> <span id="meta-labels"></span></div>
      </div>

      <div class="legend" id="legend"></div>
    </aside>
    <main>
      <div id="plot"></div>
    </main>
  </div>

  <script>
    const datasets = {datasets_json};
    const selectEl = document.getElementById("dataset-select");
    const legendEl = document.getElementById("legend");
    const metaPathEl = document.getElementById("meta-path");
    const metaPointsEl = document.getElementById("meta-points");
    const metaLabelsEl = document.getElementById("meta-labels");

    datasets.forEach((dataset, idx) => {{
      const option = document.createElement("option");
      option.value = String(idx);
      option.textContent = dataset.name;
      selectEl.appendChild(option);
    }});

    function renderLegend(items) {{
      legendEl.innerHTML = items.map((item) =>
        `<div class="legend-item"><span class="swatch" style="background:${{item.color}}"></span><span>${{item.label}} (${{item.count}})</span></div>`
      ).join("");
    }}

    function renderDataset(index) {{
      const dataset = datasets[index];
      metaPathEl.textContent = dataset.path;
      metaPointsEl.textContent = dataset.point_count;
      metaLabelsEl.textContent = dataset.label_count;
      renderLegend(dataset.legend);

      const layout = {{
        paper_bgcolor: "#0f1115",
        plot_bgcolor: "#0f1115",
        font: {{ color: "#f4f4f4" }},
        margin: {{ l: 0, r: 0, b: 0, t: 40 }},
        title: {{ text: dataset.name, x: 0.03 }},
        legend: {{ bgcolor: "rgba(0,0,0,0)", font: {{ size: 12 }} }},
        scene: {{
          bgcolor: "#0f1115",
          xaxis: {{ title: "", showbackground: false, showticklabels: false, zeroline: false }},
          yaxis: {{ title: "", showbackground: false, showticklabels: false, zeroline: false }},
          zaxis: {{ title: "", showbackground: false, showticklabels: false, zeroline: false }},
          camera: {{ eye: {{ x: 1.5, y: 1.5, z: 1.25 }} }}
        }}
      }};

      Plotly.newPlot("plot", dataset.traces, layout, {{
        responsive: true,
        displaylogo: false,
        scrollZoom: true
      }});
    }}

    selectEl.addEventListener("change", (event) => {{
      renderDataset(Number(event.target.value));
    }});

    renderDataset(0);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render all DOSNES CSVs into a single interactive HTML.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/storage/BioMedNLP/llm2vec/visualization"),
        help="Root directory to search for DOSNES CSV outputs.",
    )
    parser.add_argument(
        "--output_html",
        type=Path,
        default=Path("/storage/BioMedNLP/llm2vec/visualization/DOSNES/all_dosnes_pointclouds.html"),
        help="Output HTML path.",
    )
    parser.add_argument("--title", type=str, default="All DOSNES Point Clouds", help="HTML title.")
    parser.add_argument("--point_size", type=int, default=4, help="Marker size.")
    parser.add_argument("--opacity", type=float, default=0.85, help="Marker opacity.")
    return parser.parse_args()


def discover_csvs(root: Path) -> List[Path]:
    csvs = []
    for path in sorted(root.rglob("*.csv")):
      if path.name == "datas.csv":
          continue
      if "dosnes" not in path.as_posix().lower():
          continue
      csvs.append(path)
    return csvs


def shorten_name(path: Path) -> str:
    parent = path.parent.name
    stem = path.stem
    return "{}/{}".format(parent, stem)


def build_hover_text(row: pd.Series, hover_cols: List[str]) -> str:
    parts = []
    for col in hover_cols:
        value = row[col]
        if isinstance(value, str) and len(value) > 220:
            value = value[:217] + "..."
        parts.append("<b>{}</b>: {}".format(col, value))
    return "<br>".join(parts)


def build_dataset_payload(csv_path: Path, point_size: int, opacity: float) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    required = ["x", "y", "z", "label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns in {}: {}".format(csv_path, missing))

    labels = df["label"].astype(str).fillna("NA")
    unique_labels = labels.unique().tolist()
    color_map = {label: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, label in enumerate(unique_labels)}
    hover_cols = [col for col in df.columns if col not in {"x", "y", "z"}]

    traces = []
    legend = []
    for label in unique_labels:
        subset = df[labels == label]
        hover_text = [build_hover_text(row, hover_cols) for _, row in subset.iterrows()]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": label,
                "x": subset["x"].tolist(),
                "y": subset["y"].tolist(),
                "z": subset["z"].tolist(),
                "text": hover_text,
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {
                    "size": point_size,
                    "opacity": opacity,
                    "color": color_map[label],
                },
            }
        )
        legend.append({"label": label, "count": int(len(subset)), "color": color_map[label]})

    return {
        "name": shorten_name(csv_path),
        "path": str(csv_path),
        "point_count": int(len(df)),
        "label_count": int(df["label"].nunique()),
        "legend": legend,
        "traces": traces,
    }


def main() -> None:
    args = parse_args()
    csv_paths = discover_csvs(args.root)
    if not csv_paths:
        raise ValueError("No DOSNES CSV files found under {}".format(args.root))

    datasets = [build_dataset_payload(path, args.point_size, args.opacity) for path in csv_paths]
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.format(title=args.title, datasets_json=json.dumps(datasets, ensure_ascii=False))
    args.output_html.write_text(html, encoding="utf-8")
    print("Wrote combined HTML to: {}".format(args.output_html))


if __name__ == "__main__":
    main()
