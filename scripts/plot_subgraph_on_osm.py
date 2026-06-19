"""Plot multiple subgraph NPZ files on an OSM map.

Example:
  python scripts/plot_subgraphs_on_osm.py \
    --input-files \
      /data/yuzhang_fei/Urban_Traffic_Benchmark/subgraphs_city_m/speed/city_traffic_m_speed__category__1_0.npz \
      /data/yuzhang_fei/Urban_Traffic_Benchmark/subgraphs_city_m/speed/city_traffic_m_speed__category__2_0.npz \
      /data/yuzhang_fei/Urban_Traffic_Benchmark/subgraphs_city_m/speed/city_traffic_m_speed__category__3_0.npz \
      /data/yuzhang_fei/Urban_Traffic_Benchmark/subgraphs_city_m/speed/city_traffic_m_speed__category__4_0.npz \
      /data/yuzhang_fei/Urban_Traffic_Benchmark/subgraphs_city_m/speed/city_traffic_m_speed__category__5_0.npz \
    --output-html /data/yuzhang_fei/Urban_Traffic_Benchmark/subgraphs_city_m/speed/city_m_speed_categories_1to5_map.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import folium
import numpy as np


PALETTE = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
]


@dataclass
class LayerData:
    label: str
    color: str
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    num_nodes: int
    num_edges: int
    num_anchor_nodes: int


def detect_coordinate_indices(z: Dict[str, np.ndarray]) -> Dict[str, int]:
    # Prefer explicit metadata exported by the splitter script.
    if (
        "subgraph_coordinate_feature_names" in z
        and "subgraph_coordinate_feature_indices" in z
    ):
        names = [str(v) for v in z["subgraph_coordinate_feature_names"].tolist()]
        idxs = [int(v) for v in z["subgraph_coordinate_feature_indices"].tolist()]
        return dict(zip(names, idxs))

    feature_names = [str(v) for v in z["spatial_node_feature_names"].tolist()]
    required = [
        "x_coordinate_start",
        "y_coordinate_start",
        "x_coordinate_end",
        "y_coordinate_end",
    ]
    out: Dict[str, int] = {}
    for name in required:
        if name not in feature_names:
            raise ValueError(
                f"Missing coordinate feature '{name}'. Available: {feature_names}"
            )
        out[name] = feature_names.index(name)
    return out


def load_subgraph_segments(
    npz_path: Path,
    color: str,
    anchor_only: bool,
    max_segments: int,
) -> LayerData:
    with np.load(npz_path, allow_pickle=True) as z:
        coord_idx = detect_coordinate_indices(z)

        spatial = np.asarray(z["spatial_node_features"])
        if spatial.ndim == 3:
            spatial = spatial[0]

        x1 = spatial[:, coord_idx["x_coordinate_start"]]
        y1 = spatial[:, coord_idx["y_coordinate_start"]]
        x2 = spatial[:, coord_idx["x_coordinate_end"]]
        y2 = spatial[:, coord_idx["y_coordinate_end"]]

        node_indices = np.arange(spatial.shape[0], dtype=np.int64)
        if anchor_only and "subgraph_anchor_node_mask" in z:
            anchor_mask = np.asarray(z["subgraph_anchor_node_mask"]).astype(bool)
            node_indices = node_indices[anchor_mask]

        if max_segments > 0 and node_indices.shape[0] > max_segments:
            # Deterministic subsample for stable visualization size.
            node_indices = node_indices[:max_segments]

        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for i in node_indices.tolist():
            # Folium expects (lat, lon) => (y, x)
            segments.append(((float(y1[i]), float(x1[i])), (float(y2[i]), float(x2[i]))))

        feature_name = str(z["subgraph_road_type_feature_name"]) if "subgraph_road_type_feature_name" in z else "group"
        feature_value = str(z["subgraph_road_type_feature_value"]) if "subgraph_road_type_feature_value" in z else npz_path.stem
        label = f"{feature_name}={feature_value}"

        num_nodes = int(spatial.shape[0])
        num_edges = int(np.asarray(z["edges"]).shape[0]) if "edges" in z else 0
        if "subgraph_anchor_node_ids_in_original_graph" in z:
            num_anchor_nodes = int(np.asarray(z["subgraph_anchor_node_ids_in_original_graph"]).shape[0])
        else:
            num_anchor_nodes = num_nodes

        return LayerData(
            label=label,
            color=color,
            segments=segments,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_anchor_nodes=num_anchor_nodes,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multiple subgraph NPZ files on OSM map.")
    parser.add_argument("--input-files", type=Path, nargs="+", required=True, help="Subgraph NPZ files")
    parser.add_argument("--output-html", type=Path, required=True, help="Output HTML map path")
    parser.add_argument("--zoom-start", type=int, default=11, help="Initial map zoom")
    parser.add_argument("--line-weight", type=float, default=2.0, help="Polyline width")
    parser.add_argument("--line-opacity", type=float, default=0.85, help="Polyline opacity")
    parser.add_argument("--anchor-only", action="store_true", help="Plot only anchor (selected-type) nodes")
    parser.add_argument(
        "--max-segments-per-graph",
        type=int,
        default=0,
        help="If >0, cap rendered segments per subgraph to keep HTML light",
    )
    args = parser.parse_args()

    for p in args.input_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    layers: List[LayerData] = []
    all_points: List[Tuple[float, float]] = []

    for i, p in enumerate(args.input_files):
        color = PALETTE[i % len(PALETTE)]
        layer = load_subgraph_segments(
            npz_path=p,
            color=color,
            anchor_only=args.anchor_only,
            max_segments=args.max_segments_per_graph,
        )
        layers.append(layer)
        for a, b in layer.segments:
            all_points.append(a)
            all_points.append(b)

    if not all_points:
        raise ValueError("No segments to plot.")

    center_lat = float(np.mean([p[0] for p in all_points]))
    center_lon = float(np.mean([p[1] for p in all_points]))

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=args.zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    legend_rows = []
    for layer in layers:
        fg = folium.FeatureGroup(name=layer.label, show=True)
        for start, end in layer.segments:
            folium.PolyLine(
                locations=[start, end],
                color=layer.color,
                weight=args.line_weight,
                opacity=args.line_opacity,
            ).add_to(fg)
        fg.add_to(fmap)

        legend_rows.append(
            (
                layer.color,
                f"{layer.label}: nodes={layer.num_nodes}, anchor_nodes={layer.num_anchor_nodes}, "
                f"edges={layer.num_edges}, rendered_segments={len(layer.segments)}",
            )
        )

    folium.LayerControl(collapsed=False).add_to(fmap)

    legend_html = """
    <div style="
      position: fixed;
      bottom: 20px;
      left: 20px;
      z-index: 9999;
      background: white;
      border: 1px solid #333;
      border-radius: 6px;
      padding: 10px;
      max-width: 520px;
      font-size: 12px;
      line-height: 1.3;
    ">
      <div style="font-weight: 700; margin-bottom: 6px;">Subgraph Layers</div>
      {rows}
    </div>
    """
    rows_html = "".join(
        [
            f"<div style='margin-bottom:4px;'><span style='display:inline-block;width:10px;height:10px;background:{c};margin-right:6px;'></span>{t}</div>"
            for c, t in legend_rows
        ]
    )
    fmap.get_root().html.add_child(folium.Element(legend_html.format(rows=rows_html)))

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(args.output_html))
    print(f"Saved map: {args.output_html}")


if __name__ == "__main__":
    main()