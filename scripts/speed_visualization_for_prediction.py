"""
Визуализация предсказаний модели скорости на карте OSM.
Сравнение предсказанных значений с реальными для валидационных данных.



python visualize_predictions.py \
    --dataset /path/to/dataset.npz \
    --predictions /path/to/val_predictions.pt \
    --targets /path/to/val_targets.pt \
    --output-dir ./visualization_output \
    --create-plots \
    --create-animation
"""

"""
Визуализация предсказаний модели скорости на карте OSM.
Сравнение предсказанных значений с реальными для валидационных данных.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass

import numpy as np
import torch
import folium
from folium import plugins
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import colormaps
import branca.colormap as branca_cm


class TrafficPredictionVisualizer:
    """Визуализатор предсказаний скорости трафика."""
    
    # Палитра цветов для разных категорий
    PALETTE = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", 
        "#ff7f00", "#a65628", "#f781bf", "#999999"
    ]
    
    def __init__(
        self,
        dataset_npz_path: Path,
        predictions_path: Path,
        targets_path: Path,
        output_dir: Path,
        nan_mask_path: Optional[Path] = None
    ):
        """
        Args:
            dataset_npz_path: Путь к .npz файлу с данными датасета
            predictions_path: Путь к .pt файлу с предсказаниями
            targets_path: Путь к .pt файлу с реальными значениями
            output_dir: Директория для сохранения результатов
            nan_mask_path: Путь к .pt файлу с маской NaN (опционально)
        """
        self.dataset_path = Path(dataset_npz_path)
        self.predictions_path = Path(predictions_path)
        self.targets_path = Path(targets_path)
        self.nan_mask_path = Path(nan_mask_path) if nan_mask_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем данные
        self._load_data()
        
    def _load_data(self):
        """Загружает все необходимые данные."""
        print("Loading data...")
        
        # Загружаем датасет
        with np.load(self.dataset_path, allow_pickle=True) as data:
            print(f"Available keys: {list(data.keys())}")
            
            self.spatial_features = data['spatial_node_features'].astype(np.float32)
            self.edge_index = data['edges']
            
            print(f"spatial_features shape: {self.spatial_features.shape}")
            print(f"edges shape: {self.edge_index.shape}")
            
            # Получаем количество узлов
            # spatial_features имеет форму (1, num_nodes, num_features) или (num_nodes, num_features)
            if self.spatial_features.ndim == 3:
                self.num_nodes = self.spatial_features.shape[1]
            else:
                self.num_nodes = self.spatial_features.shape[0]
            
            # Определяем индексы координат
            self._detect_coordinate_indices(data)
            print(f"Coordinate indices: {self.coord_indices}")
            
            # Загружаем временные метки для валидации
            self.val_timestamps = data['val_timestamps']
            print(f"val_timestamps shape: {self.val_timestamps.shape}")
            
        # Загружаем предсказания и таргеты
        self.predictions = torch.load(self.predictions_path, map_location='cpu')
        self.targets = torch.load(self.targets_path, map_location='cpu')
        
        print(f"predictions shape: {self.predictions.shape}")
        print(f"targets shape: {self.targets.shape}")
        
        if self.nan_mask_path and self.nan_mask_path.exists():
            self.nan_mask = torch.load(self.nan_mask_path, map_location='cpu')
            print(f"nan_mask shape: {self.nan_mask.shape}")
        else:
            self.nan_mask = torch.isnan(self.targets)
            print("Created nan_mask from targets")
            
        print(f"Number of nodes: {self.num_nodes}")
        
        # Получаем координаты всех узлов
        self.node_coords = self._get_node_coordinates()
        print(f"node_coords shape: {self.node_coords.shape}")
        print(f"node_coords sample (first 5 nodes):\n{self.node_coords[:5]}")
        
    def _detect_coordinate_indices(self, data):
        """Определяет индексы координатных признаков."""
        feature_names = [str(v) for v in data['spatial_node_feature_names'].tolist()]
        print(f"Available feature names: {feature_names}")
        
        # Ищем координатные признаки
        self.coord_indices = {}
        
        # Пробуем найти стандартные имена
        coord_mappings = {
            'x_coordinate_start': ['x_coordinate_start', 'x_coordinate', 'lon', 'longitude', 'x', 'x_start', 'start_lon'],
            'y_coordinate_start': ['y_coordinate_start', 'y_coordinate', 'lat', 'latitude', 'y', 'y_start', 'start_lat'],
            'x_coordinate_end': ['x_coordinate_end', 'lon_end', 'x_end', 'end_lon'],
            'y_coordinate_end': ['y_coordinate_end', 'lat_end', 'y_end', 'end_lat']
        }
        
        for coord_name, alternatives in coord_mappings.items():
            found = False
            for alt in alternatives:
                if alt in feature_names:
                    self.coord_indices[coord_name] = feature_names.index(alt)
                    found = True
                    print(f"Found {coord_name} as '{alt}' at index {feature_names.index(alt)}")
                    break
            
            if not found:
                # Если не нашли, используем разумные значения по умолчанию
                if coord_name == 'x_coordinate_start':
                    self.coord_indices[coord_name] = 0
                    print(f"Using fallback: {coord_name} -> index 0")
                elif coord_name == 'y_coordinate_start':
                    self.coord_indices[coord_name] = 1
                    print(f"Using fallback: {coord_name} -> index 1")
                elif coord_name == 'x_coordinate_end':
                    self.coord_indices[coord_name] = 0
                    print(f"Using fallback: {coord_name} -> index 0")
                elif coord_name == 'y_coordinate_end':
                    self.coord_indices[coord_name] = 1
                    print(f"Using fallback: {coord_name} -> index 1")
        
    def _get_node_coordinates(self) -> np.ndarray:
        """Извлекает координаты узлов из пространственных признаков."""
        spatial = self.spatial_features
        if spatial.ndim == 3:
            spatial = spatial[0]
            
        x_start_idx = self.coord_indices.get('x_coordinate_start', 0)
        y_start_idx = self.coord_indices.get('y_coordinate_start', 1)
        
        print(f"Using indices: x_start={x_start_idx}, y_start={y_start_idx}")
        
        # Проверяем, что индексы валидны
        if x_start_idx >= spatial.shape[1] or y_start_idx >= spatial.shape[1]:
            print(f"WARNING: Invalid indices! spatial shape: {spatial.shape}")
            # Используем первые два признака
            x_start_idx = 0
            y_start_idx = 1
        
        # Возвращаем координаты (широта, долгота) - folium ожидает (lat, lon)
        coords = np.zeros((self.num_nodes, 2))
        coords[:, 0] = spatial[:, y_start_idx]  # latitude
        coords[:, 1] = spatial[:, x_start_idx]  # longitude
        
        return coords
    
    def _get_timestamp_for_visualization(self) -> int:
        """
        Выбирает временную метку для визуализации.
        Использует среднюю метку из валидационного набора.
        """
        if len(self.val_timestamps) == 0:
            return 0
        
        # Выбираем среднюю временную метку для визуализации
        mid_idx = len(self.val_timestamps) // 2
        return int(self.val_timestamps[mid_idx])
    
    def prepare_data_for_timestamp(self, timestamp_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Подготавливает данные для указанной временной метки.
        
        Returns:
            preds: Предсказания для всех узлов
            targets: Реальные значения для всех узлов
            mask: Маска валидных значений
        """
        # Проверяем, что индекс валиден
        if timestamp_idx >= self.targets.shape[0]:
            timestamp_idx = self.targets.shape[0] - 1
            
        preds = self.predictions[timestamp_idx].numpy() if torch.is_tensor(self.predictions) else self.predictions[timestamp_idx]
        targets = self.targets[timestamp_idx].numpy() if torch.is_tensor(self.targets) else self.targets[timestamp_idx]
        mask = ~(self.nan_mask[timestamp_idx].numpy() if torch.is_tensor(self.nan_mask) else self.nan_mask[timestamp_idx])
        
        # Проверяем, что размеры совпадают
        if len(preds) != self.num_nodes:
            print(f"WARNING: predictions length ({len(preds)}) != num_nodes ({self.num_nodes})")
            # Обрезаем или дополняем
            if len(preds) > self.num_nodes:
                preds = preds[:self.num_nodes]
                targets = targets[:self.num_nodes]
                mask = mask[:self.num_nodes]
            else:
                # Дополняем нулями
                preds = np.pad(preds, (0, self.num_nodes - len(preds)))
                targets = np.pad(targets, (0, self.num_nodes - len(targets)))
                mask = np.pad(mask, (0, self.num_nodes - len(mask)), constant_values=False)
        
        print(f"Timestamp {timestamp_idx}: {np.sum(mask)} valid nodes out of {self.num_nodes}")
        
        return preds, targets, mask
    
    def create_speed_color_map(self, values: np.ndarray) -> branca_cm.LinearColormap:
        """Создает цветовую карту для значений скорости."""
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        
        # Добавляем небольшой отступ
        vmin = max(0, vmin - 5)
        vmax = vmax + 5
        
        return branca_cm.LinearColormap(
            colors=[(0, 0.4, 0), (0.8, 0.8, 0), (0.8, 0.2, 0.2)],
            vmin=vmin,
            vmax=vmax,
            caption='Speed (km/h)'
        )
    
    def create_error_color_map(self, errors: np.ndarray) -> branca_cm.LinearColormap:
        """Создает цветовую карту для ошибок предсказания."""
        vmin = 0
        vmax = np.percentile(errors[~np.isnan(errors)], 95) if np.any(~np.isnan(errors)) else 10
        
        return branca_cm.LinearColormap(
            colors=['green', 'yellow', 'red'],
            vmin=vmin,
            vmax=vmax,
            caption='Prediction Error (km/h)'
        )
    
    def visualize_predictions_at_timestamp(
        self,
        timestamp_idx: Optional[int] = None,
        save_html: bool = True,
        show_errors: bool = True,
        max_nodes: int = 1000
    ) -> folium.Map:
        """
        Создает карту с визуализацией предсказаний для указанной временной метки.
        
        Args:
            timestamp_idx: Индекс временной метки (если None - выбирается автоматически)
            save_html: Сохранять ли HTML файл
            show_errors: Отображать ли ошибки предсказания отдельным слоем
            max_nodes: Максимальное количество узлов для отображения (для производительности)
        """
        if timestamp_idx is None:
            timestamp_idx = self._get_timestamp_for_visualization()
            
        print(f"Visualizing timestamp {timestamp_idx}...")
        
        # Получаем данные для выбранной временной метки
        preds, targets, mask = self.prepare_data_for_timestamp(timestamp_idx)
        
        # Проверяем, что есть валидные данные
        if not np.any(mask):
            print("WARNING: No valid data found!")
            return folium.Map(location=[0, 0], zoom_start=2)
        
        # Вычисляем ошибки
        errors = np.abs(preds - targets)
        errors[~mask] = np.nan
        
        # Определяем центр карты
        valid_coords = self.node_coords[mask]
        if len(valid_coords) == 0:
            print("WARNING: No valid coordinates found!")
            return folium.Map(location=[0, 0], zoom_start=2)
            
        center_lat = float(np.mean(valid_coords[:, 0]))
        center_lon = float(np.mean(valid_coords[:, 1]))
        print(f"Map center: ({center_lat}, {center_lon})")
        
        # Создаем карту
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        # Создаем цветовые карты
        valid_speeds = targets[mask]
        valid_preds = preds[mask]
        
        if len(valid_speeds) == 0:
            print("WARNING: No valid speeds found!")
            return m
            
        all_speeds = np.concatenate([valid_speeds, valid_preds])
        speed_cmap = self.create_speed_color_map(all_speeds)
        error_cmap = self.create_error_color_map(errors[mask])
        
        # Создаем группы слоев
        fg_targets = folium.FeatureGroup(name='Real Speed', show=True)
        fg_predictions = folium.FeatureGroup(name='Predicted Speed', show=False)
        fg_errors = folium.FeatureGroup(name='Prediction Error', show=False)
        fg_links = folium.FeatureGroup(name='Road Links', show=True)
        
        # Добавляем дорожные сегменты (связи между узлами)
        if self.edge_index is not None and len(self.edge_index) > 0:
            # Ограничиваем количество отображаемых ребер для производительности
            max_edges = 2000
            edges_to_plot = self.edge_index[:max_edges] if len(self.edge_index) > max_edges else self.edge_index
            print(f"Plotting {len(edges_to_plot)} edges...")
            
            edge_count = 0
            for i, (u, v) in enumerate(edges_to_plot):
                u, v = int(u), int(v)
                if u < self.num_nodes and v < self.num_nodes and mask[u] and mask[v]:
                    start = self.node_coords[u]
                    end = self.node_coords[v]
                    folium.PolyLine(
                        locations=[(start[0], start[1]), (end[0], end[1])],
                        color='#888888',
                        weight=1,
                        opacity=0.3
                    ).add_to(fg_links)
                    edge_count += 1
            print(f"Added {edge_count} edges to map")
        
        # Добавляем узлы с реальными значениями
        nodes_added = 0
        # Ограничиваем количество узлов для отображения
        nodes_to_plot = min(self.num_nodes, max_nodes)
        
        for i in range(nodes_to_plot):
            if not mask[i]:
                continue
                
            lat, lon = self.node_coords[i]
            true_speed = targets[i]
            pred_speed = preds[i]
            error = errors[i]
            
            # Создаем маркер с реальным значением
            popup_text = f"""
            <b>Node {i}</b><br>
            Real Speed: {true_speed:.1f} km/h<br>
            Pred Speed: {pred_speed:.1f} km/h<br>
            Error: {error:.1f} km/h
            """
            
            # Маркер для реального значения
            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                color=speed_cmap(true_speed),
                fill=True,
                fill_color=speed_cmap(true_speed),
                fill_opacity=0.8,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(fg_targets)
            
            # Маркер для предсказания
            folium.CircleMarker(
                location=(lat, lon),
                radius=3,
                color=speed_cmap(pred_speed),
                fill=True,
                fill_color=speed_cmap(pred_speed),
                fill_opacity=0.8,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(fg_predictions)
            
            # Маркер для ошибки (если включено)
            if show_errors:
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=4,
                    color=error_cmap(error),
                    fill=True,
                    fill_color=error_cmap(error),
                    fill_opacity=0.8,
                    popup=folium.Popup(f"Error: {error:.1f} km/h", max_width=200)
                ).add_to(fg_errors)
            
            nodes_added += 1
            
        print(f"Added {nodes_added} nodes to map")
        
        # Добавляем группы слоев на карту
        fg_links.add_to(m)
        fg_targets.add_to(m)
        fg_predictions.add_to(m)
        if show_errors:
            fg_errors.add_to(m)
        
        # Добавляем цветовые шкалы
        speed_cmap.add_to(m)
        
        # Добавляем контроллер слоев
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Сохраняем HTML
        if save_html:
            html_filename = self.output_dir / f'predictions_timestamp_{timestamp_idx}.html'
            try:
                m.save(str(html_filename))
                print(f"✅ Map saved to {html_filename}")
                print(f"File size: {html_filename.stat().st_size / 1024:.1f} KB")
            except Exception as e:
                print(f"❌ Error saving map: {e}")
            
        return m
    
    def visualize_comparison_grid(
        self,
        num_timestamps: int = 4,
        save_html: bool = True
    ) -> None:
        """
        Создает сетку карт для нескольких временных меток.
        """
        # Выбираем равномерно распределенные временные метки
        total_timestamps = self.targets.shape[0]
        if num_timestamps > total_timestamps:
            num_timestamps = total_timestamps
            
        indices = np.linspace(0, total_timestamps - 1, num_timestamps, dtype=int)
        print(f"Visualizing timestamps: {indices}")
        
        for idx in indices:
            self.visualize_predictions_at_timestamp(
                timestamp_idx=idx,
                save_html=save_html,
                show_errors=True
            )
    
    def create_animation_data(self, output_dir: Optional[Path] = None) -> None:
        """
        Создает данные для анимации изменения скорости во времени.
        """
        if output_dir is None:
            output_dir = self.output_dir / 'animation_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Подготавливаем данные для каждого временного шага
        all_data = []
        
        # Ограничиваем количество временных шагов для анимации
        max_timestamps = min(self.targets.shape[0], 100)
        print(f"Creating animation for {max_timestamps} timestamps...")
        
        # Для анимации используем только узлы с валидными координатами
        valid_nodes = np.all(np.isfinite(self.node_coords), axis=1)
        node_indices = np.where(valid_nodes)[0]
        
        for t in range(max_timestamps):
            preds, targets, mask = self.prepare_data_for_timestamp(t)
            
            # Создаем список данных для каждого узла
            timestamp_data = []
            for i in node_indices[:1000]:  # Ограничиваем количество узлов для анимации
                if mask[i]:
                    lat, lon = self.node_coords[i]
                    timestamp_data.append({
                        'node_id': int(i),
                        'lat': float(lat),
                        'lon': float(lon),
                        'true_speed': float(targets[i]),
                        'pred_speed': float(preds[i]),
                        'error': float(abs(preds[i] - targets[i]))
                    })
            
            all_data.append({
                'timestamp': t,
                'nodes': timestamp_data
            })
        
        # Сохраняем как JSON
        json_path = output_dir / 'animation_data.json'
        with open(json_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        print(f"✅ Animation data saved to {json_path}")
        print(f"File size: {json_path.stat().st_size / 1024:.1f} KB")
        
        # Создаем HTML шаблон для анимации
        self._create_animation_html(output_dir)
    
    def _create_animation_html(self, output_dir: Path):
        """Создает HTML шаблон для анимации."""
        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Traffic Speed Animation</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { margin:0; padding:0; font-family: Arial, sans-serif; }
        #map { position: absolute; top: 0; bottom: 0; width: 100%; }
        #controls {
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            text-align: center;
            min-width: 300px;
        }
        #controls input { width: 80%; margin: 10px 0; }
        #controls label { display: inline-block; margin: 0 5px; }
        #legend {
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            font-size: 12px;
        }
        .legend-item { display: flex; align-items: center; margin: 2px 0; }
        .legend-color { width: 20px; height: 10px; margin-right: 5px; border-radius: 2px; }
        #info {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 1000;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="info"><strong>Timestamp: <span id="timestamp-label">0</span></strong></div>
    <div id="legend">
        <div class="legend-item"><div class="legend-color" style="background: #4CAF50;"></div><span>Low Speed (0-30 km/h)</span></div>
        <div class="legend-item"><div class="legend-color" style="background: #FFC107;"></div><span>Medium Speed (30-60 km/h)</span></div>
        <div class="legend-item"><div class="legend-color" style="background: #F44336;"></div><span>High Speed (60+ km/h)</span></div>
    </div>
    <div id="controls">
        <label>Play</label>
        <button id="play-btn">▶</button>
        <br>
        <label>Timestamp: <span id="timestamp-display">0</span></label>
        <input type="range" id="timeline" min="0" max="0" value="0" step="1">
    </div>
    <script>
        let animationData = [];
        fetch('animation_data.json')
            .then(response => response.json())
            .then(data => { animationData = data; initializeMap(); })
            .catch(error => { console.error('Error loading data:', error); });
        
        let map, markers, currentIndex = 0, isPlaying = false, playInterval = null;
        
        function getSpeedColor(speed) {
            if (speed < 30) return '#4CAF50';
            if (speed < 60) return '#FFC107';
            return '#F44336';
        }
        
        function initializeMap() {
            if (animationData.length === 0) return;
            const firstData = animationData[0];
            const nodes = firstData.nodes;
            if (nodes.length === 0) return;
            
            let latSum = 0, lonSum = 0;
            nodes.forEach(n => { latSum += n.lat; lonSum += n.lon; });
            const centerLat = latSum / nodes.length;
            const centerLon = lonSum / nodes.length;
            
            map = L.map('map').setView([centerLat, centerLon], 11);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap'
            }).addTo(map);
            
            markers = [];
            nodes.forEach((node, i) => {
                const circle = L.circleMarker([node.lat, node.lon], {
                    radius: 6,
                    color: getSpeedColor(node.true_speed),
                    fillColor: getSpeedColor(node.true_speed),
                    fillOpacity: 0.8,
                    weight: 2
                }).addTo(map);
                circle.bindPopup(`<b>Node ${node.node_id}</b><br>Real Speed: ${node.true_speed.toFixed(1)} km/h<br>Pred Speed: ${node.pred_speed.toFixed(1)} km/h<br>Error: ${node.error.toFixed(1)} km/h`);
                markers.push(circle);
            });
            
            document.getElementById('timeline').max = animationData.length - 1;
            document.getElementById('timeline').value = 0;
            updateTimestamp(0);
            
            document.getElementById('timeline').addEventListener('input', function() {
                updateTimestamp(parseInt(this.value));
            });
            document.getElementById('play-btn').addEventListener('click', togglePlay);
        }
        
        function updateTimestamp(idx) {
            if (!animationData || idx >= animationData.length) return;
            currentIndex = idx;
            const data = animationData[idx];
            document.getElementById('timestamp-display').textContent = idx;
            document.getElementById('timestamp-label').textContent = idx;
            document.getElementById('timeline').value = idx;
            data.nodes.forEach((node, i) => {
                if (i < markers.length) {
                    const color = getSpeedColor(node.true_speed);
                    markers[i].setStyle({ color: color, fillColor: color });
                }
            });
        }
        
        function togglePlay() {
            isPlaying = !isPlaying;
            document.getElementById('play-btn').textContent = isPlaying ? '⏸' : '▶';
            if (isPlaying) {
                playInterval = setInterval(() => {
                    let nextIdx = currentIndex + 1;
                    if (nextIdx >= animationData.length) nextIdx = 0;
                    updateTimestamp(nextIdx);
                }, 500);
            } else {
                clearInterval(playInterval);
            }
        }
    </script>
</body>
</html>'''
        
        html_path = output_dir / 'index.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        print(f"✅ Animation HTML saved to {html_path}")
        print(f"File size: {html_path.stat().st_size / 1024:.1f} KB")
    
    def create_histogram_comparison(self, save_plot: bool = True) -> None:
        """
        Создает гистограммы сравнения предсказаний и реальных значений.
        """
        print("Creating histogram comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Подготавливаем все данные
        all_preds = []
        all_targets = []
        all_errors = []
        
        # Ограничиваем количество временных шагов для анализа
        max_timestamps = min(self.targets.shape[0], 200)
        
        for t in range(max_timestamps):
            preds, targets, mask = self.prepare_data_for_timestamp(t)
            if np.any(mask):
                all_preds.extend(preds[mask])
                all_targets.extend(targets[mask])
                all_errors.extend(np.abs(preds[mask] - targets[mask]))
        
        if len(all_targets) == 0:
            print("WARNING: No data points collected!")
            return
            
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_errors = np.array(all_errors)
        
        print(f"Collected {len(all_targets)} data points")
        
        # 1. Гистограмма скоростей
        axes[0, 0].hist(all_targets, bins=50, alpha=0.5, label='Real', color='blue')
        axes[0, 0].hist(all_preds, bins=50, alpha=0.5, label='Predicted', color='red')
        axes[0, 0].set_xlabel('Speed (km/h)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Speed Distribution Comparison')
        axes[0, 0].legend()
        
        # 2. Scatter plot
        axes[0, 1].scatter(all_targets, all_preds, alpha=0.3, s=1)
        axes[0, 1].plot([0, max(all_targets)], [0, max(all_targets)], 'k--', label='Perfect Prediction')
        axes[0, 1].set_xlabel('Real Speed (km/h)')
        axes[0, 1].set_ylabel('Predicted Speed (km/h)')
        axes[0, 1].set_title('Prediction vs Reality')
        axes[0, 1].legend()
        
        # 3. Гистограмма ошибок
        axes[1, 0].hist(all_errors, bins=50, color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Absolute Error (km/h)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Error Distribution')
        axes[1, 0].axvline(np.mean(all_errors), color='red', linestyle='--', label=f'Mean: {np.mean(all_errors):.2f}')
        axes[1, 0].legend()
        
        # 4. Ошибка по временным шагам
        errors_by_time = []
        for t in range(max_timestamps):
            preds, targets, mask = self.prepare_data_for_timestamp(t)
            if np.any(mask):
                errors_by_time.append(np.mean(np.abs(preds[mask] - targets[mask])))
            else:
                errors_by_time.append(np.nan)
        
        axes[1, 1].plot(errors_by_time, marker='o', markersize=2)
        axes[1, 1].set_xlabel('Timestamp')
        axes[1, 1].set_ylabel('Mean Absolute Error (km/h)')
        axes[1, 1].set_title('Error Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'prediction_analysis.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved analysis plot to {plot_path}")
        
        plt.close()
        print("Histogram comparison completed")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize traffic speed predictions on OSM map"
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
        "--nan-mask",
        type=Path,
        default=None,
        help="Path to NaN mask PT file (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./visualization_output",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=None,
        help="Specific timestamp index to visualize (default: middle timestamp)"
    )
    parser.add_argument(
        "--num-timestamps",
        type=int,
        default=4,
        help="Number of timestamps for grid visualization"
    )
    parser.add_argument(
        "--create-animation",
        action="store_true",
        help="Create animation data for all timestamps"
    )
    parser.add_argument(
        "--create-plots",
        action="store_true",
        help="Create statistical analysis plots"
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("Traffic Prediction Visualizer")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Predictions: {args.predictions}")
    print(f"Targets: {args.targets}")
    print(f"Output dir: {args.output_dir}")
    print("="*50)
    
    # Создаем визуализатор
    visualizer = TrafficPredictionVisualizer(
        dataset_npz_path=args.dataset,
        predictions_path=args.predictions,
        targets_path=args.targets,
        output_dir=args.output_dir,
        nan_mask_path=args.nan_mask
    )
    
    # Визуализируем
    if args.timestamp is not None:
        visualizer.visualize_predictions_at_timestamp(
            timestamp_idx=args.timestamp,
            save_html=True,
            show_errors=True
        )
    else:
        # Создаем сетку карт для нескольких временных меток
        visualizer.visualize_comparison_grid(
            num_timestamps=args.num_timestamps,
            save_html=True
        )
    
    # Создаем анимацию (если запрошено)
    if args.create_animation:
        visualizer.create_animation_data()
    
    # Создаем статистические графики (если запрошено)
    if args.create_plots:
        visualizer.create_histogram_comparison()
    
    print("\n" + "="*50)
    print("✅ All visualizations completed!")
    print(f"📁 Output directory: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()