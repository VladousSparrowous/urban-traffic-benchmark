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
            self.spatial_features = data['spatial_node_features'].astype(np.float32)
            self.edge_index = data['edges']
            
            # Получаем количество узлов из формы spatial_features
            # spatial_features может иметь форму (1, num_nodes, num_features) или (num_nodes, num_features)
            if self.spatial_features.ndim == 3:
                self.num_nodes = self.spatial_features.shape[1]
            else:
                self.num_nodes = self.spatial_features.shape[0]
            
            # Определяем индексы координат
            self._detect_coordinate_indices(data)
            
            # Загружаем временные метки для валидации
            self.val_timestamps = data['val_timestamps']
            
        # Загружаем предсказания и таргеты
        self.predictions = torch.load(self.predictions_path, map_location='cpu')
        self.targets = torch.load(self.targets_path, map_location='cpu')
        
        if self.nan_mask_path and self.nan_mask_path.exists():
            self.nan_mask = torch.load(self.nan_mask_path, map_location='cpu')
        else:
            self.nan_mask = torch.isnan(self.targets)
            
        print(f"Loaded predictions shape: {self.predictions.shape}")
        print(f"Loaded targets shape: {self.targets.shape}")
        print(f"Number of nodes: {self.num_nodes}")
        
        # Получаем координаты всех узлов
        self.node_coords = self._get_node_coordinates()
        
    def _detect_coordinate_indices(self, data):
        """Определяет индексы координатных признаков."""
        feature_names = [str(v) for v in data['spatial_node_feature_names'].tolist()]
        
        # Пытаемся найти индексы координат
        coord_names = ['x_coordinate_start', 'y_coordinate_start', 
                       'x_coordinate_end', 'y_coordinate_end']
        self.coord_indices = {}
        
        for name in coord_names:
            if name in feature_names:
                self.coord_indices[name] = feature_names.index(name)
            else:
                # Пробуем альтернативные имена
                alt_names = {
                    'x_coordinate_start': ['x_coordinate', 'lon', 'longitude', 'x'],
                    'y_coordinate_start': ['y_coordinate', 'lat', 'latitude', 'y'],
                    'x_coordinate_end': ['x_coordinate_end', 'lon_end', 'x_end'],
                    'y_coordinate_end': ['y_coordinate_end', 'lat_end', 'y_end']
                }
                found = False
                for alt in alt_names.get(name, []):
                    if alt in feature_names:
                        self.coord_indices[name] = feature_names.index(alt)
                        found = True
                        break
                if not found:
                    # Если координаты не найдены, пробуем использовать первые два признака как координаты
                    if name == 'x_coordinate_start':
                        self.coord_indices[name] = 0
                    elif name == 'y_coordinate_start':
                        self.coord_indices[name] = 1
                    elif name == 'x_coordinate_end':
                        self.coord_indices[name] = 0
                    elif name == 'y_coordinate_end':
                        self.coord_indices[name] = 1
                    else:
                        raise ValueError(f"Coordinate feature '{name}' not found. Available: {feature_names}")
        
    def _get_node_coordinates(self) -> np.ndarray:
        """Извлекает координаты узлов из пространственных признаков."""
        spatial = self.spatial_features
        if spatial.ndim == 3:
            spatial = spatial[0]
            
        x_start_idx = self.coord_indices['x_coordinate_start']
        y_start_idx = self.coord_indices['y_coordinate_start']
        
        # Возвращаем координаты (долгота, широта) - folium ожидает (lat, lon)
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
        # Валидационные данные имеют форму [num_timestamps, num_nodes]
        if timestamp_idx >= self.targets.shape[0]:
            timestamp_idx = self.targets.shape[0] - 1
            
        preds = self.predictions[timestamp_idx].numpy() if torch.is_tensor(self.predictions) else self.predictions[timestamp_idx]
        targets = self.targets[timestamp_idx].numpy() if torch.is_tensor(self.targets) else self.targets[timestamp_idx]
        mask = ~(self.nan_mask[timestamp_idx].numpy() if torch.is_tensor(self.nan_mask) else self.nan_mask[timestamp_idx])
        
        return preds, targets, mask
    
    def create_speed_color_map(self, values: np.ndarray, cmap_name: str = 'RdYlGn_r') -> branca_cm.LinearColormap:
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
        show_errors: bool = True
    ) -> folium.Map:
        """
        Создает карту с визуализацией предсказаний для указанной временной метки.
        
        Args:
            timestamp_idx: Индекс временной метки (если None - выбирается автоматически)
            save_html: Сохранять ли HTML файл
            show_errors: Отображать ли ошибки предсказания отдельным слоем
            
        Returns:
            folium.Map: Карта с визуализацией
        """
        if timestamp_idx is None:
            timestamp_idx = self._get_timestamp_for_visualization()
            
        print(f"Visualizing timestamp {timestamp_idx}...")
        
        # Получаем данные для выбранной временной метки
        preds, targets, mask = self.prepare_data_for_timestamp(timestamp_idx)
        
        # Вычисляем ошибки
        errors = np.abs(preds - targets)
        errors[~mask] = np.nan
        
        # Определяем центр карты
        center_lat = float(np.mean(self.node_coords[:, 0]))
        center_lon = float(np.mean(self.node_coords[:, 1]))
        
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
            max_edges = 5000
            edges_to_plot = self.edge_index[:max_edges] if len(self.edge_index) > max_edges else self.edge_index
            
            for i, (u, v) in enumerate(edges_to_plot):
                u, v = int(u), int(v)
                if u < self.num_nodes and v < self.num_nodes:
                    start = self.node_coords[u]
                    end = self.node_coords[v]
                    folium.PolyLine(
                        locations=[(start[0], start[1]), (end[0], end[1])],
                        color='#888888',
                        weight=1,
                        opacity=0.3
                    ).add_to(fg_links)
        
        # Добавляем узлы с реальными значениями
        for i in range(self.num_nodes):
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
            m.save(str(html_filename))
            print(f"Map saved to {html_filename}")
            
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
        
        for idx in indices:
            self.visualize_predictions_at_timestamp(
                timestamp_idx=idx,
                save_html=save_html,
                show_errors=True
            )
    
    def create_animation_data(self, output_dir: Optional[Path] = None) -> None:
        """Создает данные для анимации."""
        if output_dir is None:
            output_dir = self.output_dir / 'animation_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_data = []
        max_timestamps = min(self.targets.shape[0], 100)
        print(f"Creating animation for {max_timestamps} timestamps...")
        
        for t in range(max_timestamps):
            preds, targets, mask = self.prepare_data_for_timestamp(t)
            
            timestamp_data = []
            for i in range(self.num_nodes):
                if mask[i]:
                    lat, lon = self.node_coords[i]
                    if np.isfinite(lat) and np.isfinite(lon):
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
        
        json_path = output_dir / 'animation_data.json'
        with open(json_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        print(f"✅ Animation data saved to {json_path}")
        
        # Создаем HTML с встроенными данными
        self._create_animation_html(output_dir)
    
    def _create_animation_html(self, output_dir: Path):
        """Создает HTML шаблон для анимации с встроенными данными."""
        
        # Загружаем данные из JSON
        json_path = output_dir / 'animation_data.json'
        if not json_path.exists():
            print(f"❌ animation_data.json not found at {json_path}")
            return
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Преобразуем данные в JSON строку для встраивания
        json_data = json.dumps(data)
        
        html_content = f'''<!DOCTYPE html>
    <html>
    <head>
        <title>Traffic Speed Animation</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body {{ margin:0; padding:0; font-family: Arial, sans-serif; }}
            #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
            #controls {{
                position: absolute;
                bottom: 30px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 1000;
                background: white;
                padding: 15px 25px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                text-align: center;
                min-width: 350px;
            }}
            #controls input {{ width: 80%; margin: 10px 0; cursor: pointer; }}
            #controls label {{ display: inline-block; margin: 0 5px; font-size: 14px; }}
            #controls button {{
                padding: 5px 20px;
                font-size: 16px;
                cursor: pointer;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                margin: 5px 0;
            }}
            #controls button:hover {{ background: #45a049; }}
            #legend {{
                position: absolute;
                bottom: 100px;
                right: 20px;
                z-index: 1000;
                background: white;
                padding: 12px 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                font-size: 12px;
                min-width: 150px;
            }}
            .legend-item {{ display: flex; align-items: center; margin: 3px 0; }}
            .legend-color {{ width: 20px; height: 12px; margin-right: 8px; border-radius: 3px; }}
            #info {{
                position: absolute;
                top: 20px;
                left: 20px;
                z-index: 1000;
                background: rgba(255,255,255,0.9);
                padding: 10px 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                font-size: 14px;
            }}
            #stats {{
                position: absolute;
                top: 70px;
                left: 20px;
                z-index: 1000;
                background: rgba(255,255,255,0.9);
                padding: 10px 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                font-size: 12px;
                max-width: 200px;
            }}
            .speed-indicator {{
                position: absolute;
                bottom: 110px;
                left: 20px;
                z-index: 1000;
                background: rgba(255,255,255,0.9);
                padding: 8px 12px;
                border-radius: 5px;
                font-size: 13px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <div id="info">
            <strong>⏱ Timestamp: <span id="timestamp-label">0</span></strong>
        </div>
        <div id="stats">
            <div>📍 Nodes: <span id="node-count">0</span></div>
            <div>📊 Avg Error: <span id="avg-error">0.0</span> km/h</div>
        </div>
        <div class="speed-indicator">
            🔵 Real Speed &nbsp;|&nbsp; 🔴 Pred Speed
        </div>
        <div id="legend">
            <div style="font-weight:bold;margin-bottom:5px;">Speed Legend</div>
            <div class="legend-item"><div class="legend-color" style="background: #4CAF50;"></div><span>0 - 30 km/h</span></div>
            <div class="legend-item"><div class="legend-color" style="background: #FFC107;"></div><span>30 - 60 km/h</span></div>
            <div class="legend-item"><div class="legend-color" style="background: #F44336;"></div><span>60+ km/h</span></div>
        </div>
        <div id="controls">
            <div style="margin-bottom:5px;">
                <button id="play-btn">▶ Play</button>
                <button id="reset-btn">⟲ Reset</button>
            </div>
            <div>
                <span style="font-size:12px;">Speed: <span id="play-speed-label">1x</span></span>
                <button id="speed-down-btn" style="padding:2px 8px;font-size:12px;">-</button>
                <button id="speed-up-btn" style="padding:2px 8px;font-size:12px;">+</button>
            </div>
            <br>
            <label>Timestamp: <span id="timestamp-display">0</span> / <span id="max-timestamp">0</span></label>
            <input type="range" id="timeline" min="0" max="0" value="0" step="1">
        </div>
        <script>
            // Встроенные данные
            const animationData = {json_data};
            
            let map, markers = [], realMarkers = [], predMarkers = [];
            let currentIndex = 0, isPlaying = false, playInterval = null;
            let playSpeed = 500; // ms between frames
            
            function getSpeedColor(speed) {{
                if (speed < 30) return '#4CAF50';
                if (speed < 60) return '#FFC107';
                return '#F44336';
            }}
            
            function getSpeedColorWithOpacity(speed, opacity=0.8) {{
                const color = getSpeedColor(speed);
                // Convert hex to rgba
                const r = parseInt(color.slice(1,3), 16);
                const g = parseInt(color.slice(3,5), 16);
                const b = parseInt(color.slice(5,7), 16);
                return `rgba(${{r}},${{g}},${{b}},${{opacity}})`;
            }}
            
            function initializeMap() {{
                if (animationData.length === 0) return;
                const firstData = animationData[0];
                const nodes = firstData.nodes;
                if (nodes.length === 0) return;
                
                // Calculate center
                let latSum = 0, lonSum = 0;
                nodes.forEach(n => {{ latSum += n.lat; lonSum += n.lon; }});
                const centerLat = latSum / nodes.length;
                const centerLon = lonSum / nodes.length;
                
                map = L.map('map').setView([centerLat, centerLon], 11);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© OpenStreetMap'
                }}).addTo(map);
                
                // Create two layers: one for real speed, one for predicted
                const realLayer = L.layerGroup().addTo(map);
                const predLayer = L.layerGroup().addTo(map);
                
                // Store markers in two separate arrays
                nodes.forEach((node, i) => {{
                    // Real speed marker (larger, outer)
                    const realCircle = L.circleMarker([node.lat, node.lon], {{
                        radius: 8,
                        color: getSpeedColor(node.true_speed),
                        fillColor: getSpeedColor(node.true_speed),
                        fillOpacity: 0.6,
                        weight: 2
                    }}).addTo(realLayer);
                    realCircle.bindPopup(createPopup(node, 'Real'));
                    
                    // Predicted speed marker (smaller, inner)
                    const predCircle = L.circleMarker([node.lat, node.lon], {{
                        radius: 4,
                        color: getSpeedColor(node.pred_speed),
                        fillColor: getSpeedColor(node.pred_speed),
                        fillOpacity: 0.9,
                        weight: 1
                    }}).addTo(predLayer);
                    predCircle.bindPopup(createPopup(node, 'Predicted'));
                    
                    realMarkers.push(realCircle);
                    predMarkers.push(predCircle);
                }});
                
                document.getElementById('max-timestamp').textContent = animationData.length - 1;
                document.getElementById('timeline').max = animationData.length - 1;
                document.getElementById('timeline').value = 0;
                updateTimestamp(0);
                
                // Event listeners
                document.getElementById('timeline').addEventListener('input', function() {{
                    if (isPlaying) togglePlay();
                    updateTimestamp(parseInt(this.value));
                }});
                document.getElementById('play-btn').addEventListener('click', togglePlay);
                document.getElementById('reset-btn').addEventListener('click', function() {{
                    if (isPlaying) togglePlay();
                    document.getElementById('timeline').value = 0;
                    updateTimestamp(0);
                }});
                document.getElementById('speed-down-btn').addEventListener('click', function() {{
                    playSpeed = Math.min(2000, playSpeed + 200);
                    updatePlaySpeedLabel();
                    if (isPlaying) {{
                        clearInterval(playInterval);
                        playInterval = setInterval(playNext, playSpeed);
                    }}
                }});
                document.getElementById('speed-up-btn').addEventListener('click', function() {{
                    playSpeed = Math.max(100, playSpeed - 200);
                    updatePlaySpeedLabel();
                    if (isPlaying) {{
                        clearInterval(playInterval);
                        playInterval = setInterval(playNext, playSpeed);
                    }}
                }});
            }}
            
            function createPopup(node, type) {{
                return `<b>Node ${{node.node_id}}</b><br>
                        <b>${{type}} Speed:</b> ${{type === 'Real' ? node.true_speed.toFixed(1) : node.pred_speed.toFixed(1)}} km/h<br>
                        <b>Error:</b> ${{node.error.toFixed(1)}} km/h`;
            }}
            
            function updateTimestamp(idx) {{
                if (!animationData || idx >= animationData.length) return;
                currentIndex = idx;
                const data = animationData[idx];
                
                document.getElementById('timestamp-display').textContent = idx;
                document.getElementById('timestamp-label').textContent = idx;
                document.getElementById('timeline').value = idx;
                document.getElementById('node-count').textContent = data.nodes.length;
                
                // Calculate average error
                let totalError = 0;
                data.nodes.forEach((node, i) => {{
                    totalError += node.error;
                    if (i < realMarkers.length) {{
                        const realColor = getSpeedColor(node.true_speed);
                        const predColor = getSpeedColor(node.pred_speed);
                        realMarkers[i].setStyle({{ color: realColor, fillColor: realColor }});
                        predMarkers[i].setStyle({{ color: predColor, fillColor: predColor }});
                    }}
                }});
                const avgError = data.nodes.length > 0 ? totalError / data.nodes.length : 0;
                document.getElementById('avg-error').textContent = avgError.toFixed(1);
            }}
            
            function playNext() {{
                let nextIdx = currentIndex + 1;
                if (nextIdx >= animationData.length) {{
                    nextIdx = 0;
                }}
                updateTimestamp(nextIdx);
            }}
            
            function togglePlay() {{
                isPlaying = !isPlaying;
                document.getElementById('play-btn').textContent = isPlaying ? '⏸ Pause' : '▶ Play';
                if (isPlaying) {{
                    playInterval = setInterval(playNext, playSpeed);
                }} else {{
                    clearInterval(playInterval);
                }}
            }}
            
            function updatePlaySpeedLabel() {{
                const speed = (1000 / playSpeed).toFixed(1);
                document.getElementById('play-speed-label').textContent = speed + 'x';
            }}
            
            // Initialize when page loads
            document.addEventListener('DOMContentLoaded', initializeMap);
        </script>
    </body>
    </html>'''
        
        html_path = output_dir / 'index.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ Animation HTML saved to {html_path}")
        print(f"📊 Total timestamps: {len(data)}")
        print(f"📍 Total nodes in first frame: {len(data[0]['nodes']) if data else 0}")
    
    def create_histogram_comparison(self, save_plot: bool = True) -> None:
        """
        Создает гистограммы сравнения предсказаний и реальных значений.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Подготавливаем все данные
        all_preds = []
        all_targets = []
        all_errors = []
        
        # Ограничиваем количество временных шагов для анализа
        max_timestamps = min(self.targets.shape[0], 200)
        
        for t in range(max_timestamps):
            preds, targets, mask = self.prepare_data_for_timestamp(t)
            all_preds.extend(preds[mask])
            all_targets.extend(targets[mask])
            all_errors.extend(np.abs(preds[mask] - targets[mask]))
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_errors = np.array(all_errors)
        
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
            errors_by_time.append(np.mean(np.abs(preds[mask] - targets[mask])))
        
        axes[1, 1].plot(errors_by_time, marker='o', markersize=2)
        axes[1, 1].set_xlabel('Timestamp')
        axes[1, 1].set_ylabel('Mean Absolute Error (km/h)')
        axes[1, 1].set_title('Error Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'prediction_analysis.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved analysis plot to {plot_path}")
        
        plt.show()