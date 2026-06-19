"""
Визуализация предсказаний модели скорости на карте OSM.
Сравнение предсказанных значений с реальными для валидационных данных.



python visualize_predictions.py \
    --dataset /path/to/dataset.npz \
    --predictions /path/to/val_predictions.pt \
    --targets /path/to/val_targets.pt \
    --output-dir ./visualization_output \


"""
"""
Простая визуализация предсказаний скорости на карте OSM.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import folium
from folium import plugins
import branca.colormap as branca_cm


class SimpleTrafficVisualizer:
    """Простой визуализатор предсказаний скорости трафика."""
    
    def __init__(
        self,
        dataset_npz_path: Path,
        predictions_path: Path,
        targets_path: Path,
        output_dir: Path,
        target_index: int = 0
    ):
        self.dataset_path = Path(dataset_npz_path)
        self.predictions_path = Path(predictions_path)
        self.targets_path = Path(targets_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_index = target_index
        
        self._load_data()
        
    def _load_data(self):
        """Загружает необходимые данные."""
        print("Loading data...")
        
        # Загружаем датасет
        with np.load(self.dataset_path, allow_pickle=True) as data:
            self.spatial_features = data['spatial_node_features'].astype(np.float32)
            self.edge_index = data['edges']
            
            if self.spatial_features.ndim == 3:
                self.num_nodes = self.spatial_features.shape[1]
            else:
                self.num_nodes = self.spatial_features.shape[0]
            
            # Определяем индексы координат
            feature_names = [str(v) for v in data['spatial_node_feature_names'].tolist()]
            self._detect_coordinate_indices(feature_names)
            
            self.val_timestamps = data['val_timestamps']
        
        # Загружаем предсказания и таргеты
        self.predictions = torch.load(self.predictions_path, map_location='cpu')
        self.targets = torch.load(self.targets_path, map_location='cpu')
        
        # Обработка 3D данных
        if self.predictions.ndim == 3:
            self.predictions = self.predictions[:, :, self.target_index]
        if self.targets.ndim == 3:
            self.targets = self.targets[:, :, self.target_index]
        
        # Обрезаем до нужного размера
        if self.predictions.shape[1] > self.num_nodes:
            self.predictions = self.predictions[:, :self.num_nodes]
            self.targets = self.targets[:, :self.num_nodes]
        
        # Получаем координаты
        self.node_coords = self._get_node_coordinates()
        
        print(f"✅ Data loaded: {self.num_nodes} nodes, {self.targets.shape[0]} timestamps")
        
    def _detect_coordinate_indices(self, feature_names):
        """Определяет индексы координат."""
        self.coord_indices = {}
        
        coord_mappings = {
            'x': ['x_coordinate_start', 'x_coordinate', 'lon', 'longitude', 'x', 'x_start', 'start_lon'],
            'y': ['y_coordinate_start', 'y_coordinate', 'lat', 'latitude', 'y', 'y_start', 'start_lat']
        }
        
        for coord_name, alternatives in coord_mappings.items():
            found = False
            for alt in alternatives:
                if alt in feature_names:
                    self.coord_indices[coord_name] = feature_names.index(alt)
                    found = True
                    break
            if not found:
                # Fallback индексы
                self.coord_indices[coord_name] = 23 if coord_name == 'y' else 22
        
    def _get_node_coordinates(self) -> np.ndarray:
        """Извлекает координаты узлов."""
        spatial = self.spatial_features
        if spatial.ndim == 3:
            spatial = spatial[0]
            
        x_idx = self.coord_indices.get('x', 22)
        y_idx = self.coord_indices.get('y', 23)
        
        coords = np.zeros((self.num_nodes, 2))
        coords[:, 0] = spatial[:, y_idx]  # latitude
        coords[:, 1] = spatial[:, x_idx]  # longitude
        
        return coords
    
    def visualize_timestamp(
        self,
        timestamp_idx: int = 0,
        show_errors: bool = True,
        save_html: bool = True
    ) -> folium.Map:
        """
        Создает карту для указанной временной метки.
        """
        if timestamp_idx >= self.targets.shape[0]:
            timestamp_idx = self.targets.shape[0] - 1
        
        # Извлекаем данные
        preds = self.predictions[timestamp_idx].numpy()
        targets = self.targets[timestamp_idx].numpy()
        
        # Маска валидных данных (не NaN)
        mask = ~np.isnan(targets)
        valid_count = np.sum(mask)
        print(f"Timestamp {timestamp_idx}: {valid_count} valid nodes")
        
        if valid_count == 0:
            print("❌ No valid data found!")
            return folium.Map(location=[0, 0], zoom_start=2)
        
        # Ошибки
        errors = np.abs(preds - targets)
        errors[~mask] = np.nan
        
        # Центр карты
        valid_coords = self.node_coords[mask]
        center_lat = float(np.mean(valid_coords[:, 0]))
        center_lon = float(np.mean(valid_coords[:, 1]))
        
        # Создаем карту
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        # Добавляем мини-карту
        plugins.MiniMap().add_to(m)
        
        # Определяем цветовые диапазоны
        valid_speeds = targets[mask]
        vmin_speed = max(0, np.nanmin(valid_speeds) - 5)
        vmax_speed = np.nanmax(valid_speeds) + 5
        
        # Цветовая карта для скорости
        speed_cmap = branca_cm.LinearColormap(
            colors=['green', 'yellow', 'red'],
            vmin=vmin_speed,
            vmax=vmax_speed,
            caption='Speed (km/h)'
        )
        
        # Создаем слои
        fg_speed = folium.FeatureGroup(name='Real Speed', show=True)
        fg_pred = folium.FeatureGroup(name='Predicted Speed', show=False)
        fg_error = folium.FeatureGroup(name='Error', show=False)
        fg_roads = folium.FeatureGroup(name='Roads', show=True)
        
        # Добавляем дороги (первые 1000 ребер для производительности)
        if self.edge_index is not None and len(self.edge_index) > 0:
            max_edges = min(1000, len(self.edge_index))
            for i in range(max_edges):
                u, v = int(self.edge_index[i][0]), int(self.edge_index[i][1])
                if u < self.num_nodes and v < self.num_nodes:
                    start = self.node_coords[u]
                    end = self.node_coords[v]
                    if np.isfinite(start).all() and np.isfinite(end).all():
                        folium.PolyLine(
                            locations=[(start[0], start[1]), (end[0], end[1])],
                            color='#888888',
                            weight=1,
                            opacity=0.3
                        ).add_to(fg_roads)
        
        # Добавляем узлы
        for i in range(self.num_nodes):
            if not mask[i]:
                continue
                
            lat, lon = self.node_coords[i]
            if not np.isfinite(lat) or not np.isfinite(lon):
                continue
            
            true_speed = targets[i]
            pred_speed = preds[i]
            error = errors[i]
            
            if np.isnan(true_speed) or np.isnan(pred_speed):
                continue
            
            # Popup с информацией
            popup_text = f"""
            <b>Node {i}</b><br>
            <span style="color:blue;">Real Speed: {true_speed:.1f} km/h</span><br>
            <span style="color:red;">Pred Speed: {pred_speed:.1f} km/h</span><br>
            <span style="color:orange;">Error: {error:.1f} km/h</span>
            """
            
            # Реальная скорость (крупные точки)
            folium.CircleMarker(
                location=(lat, lon),
                radius=7,
                color=speed_cmap(true_speed),
                fill=True,
                fill_color=speed_cmap(true_speed),
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=250)
            ).add_to(fg_speed)
            
            # Предсказанная скорость (маленькие точки внутри)
            folium.CircleMarker(
                location=(lat, lon),
                radius=3,
                color='white' if np.abs(true_speed - pred_speed) < 10 else 'black',
                fill=True,
                fill_color=speed_cmap(pred_speed),
                fill_opacity=1.0,
                popup=folium.Popup(popup_text, max_width=250)
            ).add_to(fg_pred)
            
            # Ошибка (цветной кружок вокруг)
            if show_errors:
                error_cmap = branca_cm.LinearColormap(
                    colors=['green', 'yellow', 'red'],
                    vmin=0,
                    vmax=np.percentile(errors[mask], 95)
                )
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=9,
                    color=error_cmap(error),
                    fill=False,
                    weight=3,
                    opacity=0.5,
                    popup=folium.Popup(f"Error: {error:.1f} km/h", max_width=200)
                ).add_to(fg_error)
        
        # Добавляем слои на карту
        fg_roads.add_to(m)
        fg_speed.add_to(m)
        fg_pred.add_to(m)
        if show_errors:
            fg_error.add_to(m)
        
        # Добавляем цветовую шкалу
        speed_cmap.add_to(m)
        
        # Управление слоями
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Добавляем информацию о таймстампе
        if hasattr(self, 'val_timestamps') and len(self.val_timestamps) > timestamp_idx:
            timestamp_info = f"Timestamp: {self.val_timestamps[timestamp_idx]}"
        else:
            timestamp_info = f"Timestamp: {timestamp_idx}"
        
        folium.Marker(
            location=[center_lat, center_lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:14px;font-weight:bold;color:white;background:rgba(0,0,0,0.7);padding:5px 10px;border-radius:5px;">{timestamp_info}</div>'
            )
        ).add_to(m)
        
        if save_html:
            html_filename = self.output_dir / f'traffic_map_timestamp_{timestamp_idx}.html'
            m.save(str(html_filename))
            print(f"✅ Map saved to {html_filename}")
        
        return m
    
    def visualize_all_timestamps(self, num_timestamps: int = 5, save_html: bool = True):
        """Создает карты для нескольких временных меток."""
        total = self.targets.shape[0]
        if num_timestamps > total:
            num_timestamps = total
            
        indices = np.linspace(0, total - 1, num_timestamps, dtype=int)
        print(f"Creating maps for timestamps: {indices}")
        
        for idx in indices:
            self.visualize_timestamp(
                timestamp_idx=idx,
                show_errors=True,
                save_html=save_html
            )
    
    def visualize_difference_map(self, timestamp_idx: int = 0, save_html: bool = True):
        """
        Создает карту, показывающую разницу между предсказанием и реальностью.
        """
        if timestamp_idx >= self.targets.shape[0]:
            timestamp_idx = self.targets.shape[0] - 1
        
        preds = self.predictions[timestamp_idx].numpy()
        targets = self.targets[timestamp_idx].numpy()
        mask = ~np.isnan(targets)
        
        if np.sum(mask) == 0:
            print("❌ No valid data!")
            return folium.Map(location=[0, 0], zoom_start=2)
        
        differences = preds - targets
        differences[~mask] = np.nan
        
        valid_coords = self.node_coords[mask]
        center_lat = float(np.mean(valid_coords[:, 0]))
        center_lon = float(np.mean(valid_coords[:, 1]))
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Цветовая карта для разницы
        max_diff = max(abs(np.nanmin(differences)), abs(np.nanmax(differences)))
        diff_cmap = branca_cm.LinearColormap(
            colors=['red', 'white', 'green'],
            vmin=-max_diff,
            vmax=max_diff,
            caption='Prediction - Real (km/h)'
        )
        
        # Добавляем узлы
        for i in range(self.num_nodes):
            if not mask[i]:
                continue
                
            lat, lon = self.node_coords[i]
            if not np.isfinite(lat) or not np.isfinite(lon):
                continue
            
            diff = differences[i]
            if np.isnan(diff):
                continue
            
            popup_text = f"""
            <b>Node {i}</b><br>
            Real: {targets[i]:.1f} km/h<br>
            Pred: {preds[i]:.1f} km/h<br>
            <b>Diff: {diff:+.1f} km/h</b>
            """
            
            # Размер точки зависит от величины разницы
            radius = 4 + min(abs(diff) / 5, 8)
            
            folium.CircleMarker(
                location=(lat, lon),
                radius=radius,
                color=diff_cmap(diff),
                fill=True,
                fill_color=diff_cmap(diff),
                fill_opacity=0.8,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(m)
        
        diff_cmap.add_to(m)
        
        if save_html:
            html_filename = self.output_dir / f'difference_map_timestamp_{timestamp_idx}.html'
            m.save(str(html_filename))
            print(f"✅ Difference map saved to {html_filename}")
        
        return m


def main():
    parser = argparse.ArgumentParser(
        description="Simple OSM visualization of traffic speed predictions"
    )
    parser.add_argument(
        "--dataset", 
        type=Path, 
        required=True,
        help="Path to dataset NPZ file"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions PT file"
    )
    parser.add_argument(
        "--targets",
        type=Path,
        required=True,
        help="Path to targets PT file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./osm_visualization",
        help="Output directory for HTML maps"
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=0,
        help="Target index to visualize"
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=0,
        help="Timestamp index to visualize"
    )
    parser.add_argument(
        "--num-timestamps",
        type=int,
        default=5,
        help="Number of timestamps to visualize"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Visualize all timestamps"
    )
    parser.add_argument(
        "--difference",
        action="store_true",
        help="Create difference map"
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("Simple OSM Traffic Visualizer")
    print("="*50)
    
    visualizer = SimpleTrafficVisualizer(
        dataset_npz_path=args.dataset,
        predictions_path=args.predictions,
        targets_path=args.targets,
        output_dir=args.output_dir,
        target_index=args.target_index
    )
    
    if args.all:
        # Визуализируем все таймстампы (максимум 20)
        num = min(20, visualizer.targets.shape[0])
        visualizer.visualize_all_timestamps(num_timestamps=num)
    elif args.difference:
        visualizer.visualize_difference_map(timestamp_idx=args.timestamp)
    else:
        # Одиночная карта
        visualizer.visualize_timestamp(timestamp_idx=args.timestamp)
    
    print("\n" + "="*50)
    print(f"✅ All maps saved to: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()