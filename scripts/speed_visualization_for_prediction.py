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


@dataclass
class PredictionData:
    """Данные для визуализации предсказаний."""
    node_id: int
    timestamp_idx: int
    lat: float
    lon: float
    true_speed: float
    pred_speed: float
    error: float


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
            self.num_nodes = data['num_nodes'].item()
            
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
                    'x_coordinate_start': ['x_coordinate', 'lon', 'longitude'],
                    'y_coordinate_start': ['y_coordinate', 'lat', 'latitude'],
                    'x_coordinate_end': ['x_coordinate_end', 'lon_end'],
                    'y_coordinate_end': ['y_coordinate_end', 'lat_end']
                }
                found = False
                for alt in alt_names.get(name, []):
                    if alt in feature_names:
                        self.coord_indices[name] = feature_names.index(alt)
                        found = True
                        break
                if not found:
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
            for i, (u, v) in enumerate(self.edge_index):
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
        """
        Создает данные для анимации изменения скорости во времени.
        Сохраняет как JSON для использования в JavaScript анимациях.
        """
        if output_dir is None:
            output_dir = self.output_dir / 'animation_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Подготавливаем данные для каждого временного шага
        all_data = []
        
        for t in range(self.targets.shape[0]):
            preds, targets, mask = self.prepare_data_for_timestamp(t)
            
            # Создаем список данных для каждого узла
            timestamp_data = []
            for i in range(self.num_nodes):
                if mask[i]:
                    lat, lon = self.node_coords[i]
                    timestamp_data.append({
                        'node_id': i,
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
        with open(output_dir / 'animation_data.json', 'w') as f:
            json.dump(all_data, f, indent=2)
            
        print(f"Animation data saved to {output_dir / 'animation_data.json'}")
        
        # Создаем HTML шаблон для анимации
        self._create_animation_html(output_dir)
    
    def _create_animation_html(self, output_dir: Path):
        """Создает HTML шаблон для анимации."""
        html_content = '''
        <!DOCTYPE html>
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
                #controls input {
                    width: 80%;
                    margin: 10px 0;
                }
                #controls label {
                    display: inline-block;
                    margin: 0 5px;
                }
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
                .legend-item {
                    display: flex;
                    align-items: center;
                    margin: 2px 0;
                }
                .legend-color {
                    width: 20px;
                    height: 10px;
                    margin-right: 5px;
                    border-radius: 2px;
                }
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
            <div id="info">
                <strong>Timestamp: <span id="timestamp-label">0</span></strong>
            </div>
            <div id="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #4CAF50;"></div>
                    <span>Low Speed (0-30 km/h)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FFC107;"></div>
                    <span>Medium Speed (30-60 km/h)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #F44336;"></div>
                    <span>High Speed (60+ km/h)</span>
                </div>
            </div>
            <div id="controls">
                <label>Play</label>
                <button id="play-btn">▶</button>
                <br>
                <label>Timestamp: <span id="timestamp-display">0</span></label>
                <input type="range" id="timeline" min="0" max="0" value="0" step="1">
            </div>
            
            <script src="animation_data/animation_data.json"></script>
            <script>
                // Загрузка данных
                let animationData = [];
                
                fetch('animation_data/animation_data.json')
                    .then(response => response.json())
                    .then(data => {
                        animationData = data;
                        initializeMap();
                    })
                    .catch(error => {
                        console.error('Error loading data:', error);
                        document.getElementById('info').innerHTML = '<strong>Error loading data</strong>';
                    });
                
                let map, markers, currentIndex = 0;
                let isPlaying = false;
                let playInterval = null;
                
                function getSpeedColor(speed) {
                    if (speed < 30) return '#4CAF50';
                    if (speed < 60) return '#FFC107';
                    return '#F44336';
                }
                
                function initializeMap() {
                    if (animationData.length === 0) return;
                    
                    const firstData = animationData[0];
                    const nodes = firstData.nodes;
                    
                    if (nodes.length === 0) {
                        document.getElementById('info').innerHTML = '<strong>No data available</strong>';
                        return;
                    }
                    
                    // Вычисляем центр
                    let latSum = 0, lonSum = 0;
                    nodes.forEach(n => { latSum += n.lat; lonSum += n.lon; });
                    const centerLat = latSum / nodes.length;
                    const centerLon = lonSum / nodes.length;
                    
                    map = L.map('map').setView([centerLat, centerLon], 11);
                    
                    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                        attribution: '© OpenStreetMap'
                    }).addTo(map);
                    
                    // Создаем маркеры
                    markers = [];
                    nodes.forEach((node, i) => {
                        const circle = L.circleMarker([node.lat, node.lon], {
                            radius: 6,
                            color: getSpeedColor(node.true_speed),
                            fillColor: getSpeedColor(node.true_speed),
                            fillOpacity: 0.8,
                            weight: 2
                        }).addTo(map);
                        
                        circle.bindPopup(`
                            <b>Node ${node.node_id}</b><br>
                            Real Speed: ${node.true_speed.toFixed(1)} km/h<br>
                            Pred Speed: ${node.pred_speed.toFixed(1)} km/h<br>
                            Error: ${node.error.toFixed(1)} km/h
                        `);
                        
                        markers.push(circle);
                    });
                    
                    // Настраиваем слайдер
                    document.getElementById('timeline').max = animationData.length - 1;
                    document.getElementById('timeline').value = 0;
                    
                    // Обновляем отображение
                    updateTimestamp(0);
                    
                    // Добавляем обработчики
                    document.getElementById('timeline').addEventListener('input', function() {
                        const idx = parseInt(this.value);
                        updateTimestamp(idx);
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
                    
                    // Обновляем маркеры
                    data.nodes.forEach((node, i) => {
                        if (i < markers.length) {
                            const color = getSpeedColor(node.true_speed);
                            markers[i].setStyle({
                                color: color,
                                fillColor: color
                            });
                        }
                    });
                }
                
                function togglePlay() {
                    isPlaying = !isPlaying;
                    document.getElementById('play-btn').textContent = isPlaying ? '⏸' : '▶';
                    
                    if (isPlaying) {
                        playInterval = setInterval(() => {
                            let nextIdx = currentIndex + 1;
                            if (nextIdx >= animationData.length) {
                                nextIdx = 0;
                            }
                            updateTimestamp(nextIdx);
                        }, 500);
                    } else {
                        clearInterval(playInterval);
                    }
                }
            </script>
        </body>
        </html>
        '''
        
        with open(output_dir / 'index.html', 'w') as f:
            f.write(html_content)
        print(f"Animation HTML saved to {output_dir / 'index.html'}")
    
    def create_histogram_comparison(self, save_plot: bool = True) -> None:
        """
        Создает гистограммы сравнения предсказаний и реальных значений.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Подготавливаем все данные
        all_preds = []
        all_targets = []
        all_errors = []
        
        for t in range(self.targets.shape[0]):
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
        for t in range(self.targets.shape[0]):
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
    
    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("Open the HTML files in your browser to view the maps.")


if __name__ == "__main__":
    main()