import os
import typing as tp
import yaml
from pathlib import Path
import numpy as np
import torch
from time import perf_counter
import pickle
from typing import Any, Dict, Optional
TorchStateDict = tp.Mapping[str, torch.FloatTensor]


class LocalLogger:
 
    def __init__(self, save_dir: str, metric: str = 'MAE', do_not_evaluate_on_test: bool = False):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.do_not_evaluate_on_test = do_not_evaluate_on_test
        
        self.val_metrics = []
        self.test_metrics = [] if not do_not_evaluate_on_test else None
        self.best_steps = []
        self.best_epochs = []
        self.num_runs = 0
        self.current_run = 0
        self.elapsed_time = 0
        self.max_memory_allocated = 0
        
        # Для сохранения предсказаний
        self.val_predictions = None
        self.val_targets = None
        self.test_predictions = None
        self.test_targets = None
        
    def _to_python_type(self, value):
        """Рекурсивно преобразует numpy/torch типы в Python типы."""
        if isinstance(value, (np.ndarray, torch.Tensor)):
            return value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            return float(value)
        elif isinstance(value, (np.int32, np.int64)):
            return int(value)
        elif isinstance(value, dict):
            return {k: self._to_python_type(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._to_python_type(v) for v in value]
        elif isinstance(value, (np.bool_)):
            return bool(value)
        else:
            return value
    
    def get_state(self):
        """Возвращает состояние для сохранения в чекпоинт."""
        return {
            'val_metrics': self._to_python_type(self.val_metrics),
            'test_metrics': self._to_python_type(self.test_metrics),
            'best_steps': self._to_python_type(self.best_steps),
            'best_epochs': self._to_python_type(self.best_epochs),
            'num_runs': self.num_runs,
            'current_run': self.current_run,
            'elapsed_time': self.elapsed_time,
            'max_memory_allocated': self.max_memory_allocated,
        }
    
    def set_state(self, state):
        """Восстанавливает состояние из чекпоинта."""
        self.val_metrics = state.get('val_metrics', [])
        self.test_metrics = state.get('test_metrics', [])
        self.best_steps = state.get('best_steps', [])
        self.best_epochs = state.get('best_epochs', [])
        self.num_runs = state.get('num_runs', 0)
        self.current_run = state.get('current_run', 0)
        self.elapsed_time = state.get('elapsed_time', 0)
        self.max_memory_allocated = state.get('max_memory_allocated', 0)
    
    def start_run(self, run):
        self.current_run = run
        self.val_metrics.append(None)
        if not self.do_not_evaluate_on_test:
            self.test_metrics.append(None)
        self.best_steps.append(None)
        self.best_epochs.append(None)
        print(f"Starting run {run}...")
        
    def update_metrics(self, metrics, step, epoch):
        current_val = metrics[f'val {self.metric}']
        if self.val_metrics[-1] is None or current_val < self.val_metrics[-1]:
            self.val_metrics[-1] = float(current_val)  # Приводим к float
            if not self.do_not_evaluate_on_test:
                self.test_metrics[-1] = float(metrics[f'test {self.metric}'])
            self.best_steps[-1] = int(step)
            self.best_epochs[-1] = int(epoch)
    
    def save_predictions(self, predictions, targets, split='val'):
        """Сохраняет предсказания."""
        if split == 'val':
            self.val_predictions = predictions
            self.val_targets = targets
            torch.save(predictions, self.save_dir / 'val_predictions.pt')
            torch.save(targets, self.save_dir / 'val_targets.pt')
        else:
            self.test_predictions = predictions
            self.test_targets = targets
            torch.save(predictions, self.save_dir / 'test_predictions.pt')
            torch.save(targets, self.save_dir / 'test_targets.pt')
            
    def finish_run(self):
        self.save_metrics()
        self.num_runs += 1
        print(f"Finished run {self.current_run}. Best {self.metric}: {self.val_metrics[-1]:.4f}")
        
    def save_metrics(self):
        metrics_file = self.save_dir / 'metrics.yaml'
        
        # Приводим все значения к Python типам
        val_metrics = self._to_python_type(self.val_metrics)
        val_metrics_filtered = [m for m in val_metrics if m is not None]
        
        metrics = {
            'num_runs': len(val_metrics_filtered),
            f'val_{self.metric}_mean': float(np.mean(val_metrics_filtered)) if val_metrics_filtered else None,
            f'val_{self.metric}_std': float(np.std(val_metrics_filtered, ddof=1)) if len(val_metrics_filtered) > 1 else None,
            f'val_{self.metric}_values': val_metrics_filtered,
            'best_steps': self._to_python_type(self.best_steps),
            'best_epochs': self._to_python_type(self.best_epochs),
        }
        
        if not self.do_not_evaluate_on_test and self.test_metrics is not None:
            test_metrics = self._to_python_type(self.test_metrics)
            test_metrics_filtered = [m for m in test_metrics if m is not None]
            
            if test_metrics_filtered:
                metrics[f'test_{self.metric}_mean'] = float(np.mean(test_metrics_filtered))
                metrics[f'test_{self.metric}_std'] = float(np.std(test_metrics_filtered, ddof=1)) if len(test_metrics_filtered) > 1 else None
                metrics[f'test_{self.metric}_values'] = test_metrics_filtered
                metrics['best_test_metric'] = float(np.min(test_metrics_filtered))
        
        if val_metrics_filtered:
            metrics['best_val_metric'] = float(np.min(val_metrics_filtered))
        
        metrics['elapsed_time'] = float(self.elapsed_time)
        metrics['max_memory_allocated_mb'] = int(self.max_memory_allocated // (2 ** 20))
        
        # Сохраняем YAML
        with open(metrics_file, 'w') as f:
            yaml.safe_dump(metrics, f, sort_keys=False, default_flow_style=False)
            
        print(f"Metrics saved to {metrics_file}")
            
    def print_metrics_summary(self):
        metrics_file = self.save_dir / 'metrics.yaml'
        if not metrics_file.exists():
            print("No metrics file found.")
            return
            
        with open(metrics_file, 'r') as f:
            metrics = yaml.safe_load(f)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE - METRICS SUMMARY")
        print("="*50)
        print(f"Finished {metrics['num_runs']} runs.")
        print(f"Val {self.metric} mean: {metrics.get(f'val_{self.metric}_mean', 'N/A'):.4f}")
        print(f"Val {self.metric} std: {metrics.get(f'val_{self.metric}_std', 'N/A'):.4f}")
        
        if not self.do_not_evaluate_on_test and f'test_{self.metric}_mean' in metrics:
            print(f"Test {self.metric} mean: {metrics[f'test_{self.metric}_mean']:.4f}")
            print(f"Test {self.metric} std: {metrics.get(f'test_{self.metric}_std', 'N/A'):.4f}")
        
        print(f"Best val {self.metric}: {metrics.get('best_val_metric', 'N/A'):.4f}")
        if 'best_test_metric' in metrics:
            print(f"Best test {self.metric}: {metrics['best_test_metric']:.4f}")
        print(f"Elapsed time: {metrics.get('elapsed_time', 0):.2f} seconds")
        print(f"Max memory allocated: {metrics.get('max_memory_allocated_mb', 0)} MB")
        print("="*50)


class LocalStateHandler:

    
    def __init__(self, checkpoint_dir: Path, checkpoint_steps_interval: int = 1000):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.pt"
        self.checkpoint_steps_interval = checkpoint_steps_interval
        
        # Состояние тренировки
        self.steps_after_run_start = 0
        self.epochs_finished = 0
        self.optimizer_steps_done = 0
        self.loss = 0.0
        self.num_runs_completed = 0
        self.current_run = 0
        
        # Компоненты
        self.model = None
        self.optimizer = None
        self.grad_scaler = None
        self.logger = None
        
    def load_checkpoint(self) -> bool:
        """Загружает чекпоинт если существует."""
        if not self.checkpoint_file.exists():
            print("No checkpoint found, starting from scratch.")
            return False
            
        try:
            checkpoint = torch.load(self.checkpoint_file, map_location='cpu')
            
            self.steps_after_run_start = checkpoint.get('steps_after_run_start', 0)
            self.epochs_finished = checkpoint.get('epochs_finished', 0)
            self.optimizer_steps_done = checkpoint.get('optimizer_steps_done', 0)
            self.loss = checkpoint.get('loss', 0.0)
            self.num_runs_completed = checkpoint.get('num_runs_completed', 0)
            self.current_run = checkpoint.get('current_run', 0)
            
            # Восстанавливаем состояния компонентов если они уже добавлены
            if self.model is not None and 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            if self.optimizer is not None and 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if self.grad_scaler is not None and 'scaler_state' in checkpoint:
                self.grad_scaler.load_state_dict(checkpoint['scaler_state'])
                
            print(f"Loaded checkpoint from {self.checkpoint_file}")
            print(f"Resuming from run {self.current_run}, steps: {self.steps_after_run_start}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def save_checkpoint(self, finish_run: bool = False):
        """Сохраняет текущее состояние."""
        checkpoint = {
            'steps_after_run_start': 0 if finish_run else self.steps_after_run_start,
            'epochs_finished': 0 if finish_run else self.epochs_finished,
            'optimizer_steps_done': 0 if finish_run else self.optimizer_steps_done,
            'loss': 0.0 if finish_run else self.loss,
            'num_runs_completed': self.num_runs_completed,
            'current_run': self.current_run,
        }
        
        # Сохраняем состояния только если не завершаем run
        if not finish_run:
            if self.model is not None:
                checkpoint['model_state'] = self.model.state_dict()
            if self.optimizer is not None:
                checkpoint['optimizer_state'] = self.optimizer.state_dict()
            if self.grad_scaler is not None:
                checkpoint['scaler_state'] = self.grad_scaler.state_dict()
        
        # Сохраняем состояние логгера если он есть
        if self.logger is not None:
            checkpoint['logger_state'] = self.logger.get_state()
        
        torch.save(checkpoint, self.checkpoint_file)
        print(f"Checkpoint saved to {self.checkpoint_file}")
    
    def add_model(self, model):
        self.model = model
        
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_grad_scaler(self, scaler):
        self.grad_scaler = scaler
        
    def add_logger(self, logger):
        self.logger = logger
        
    def step(self):
        """Вызывается после каждого шага оптимизации."""
        self.steps_after_run_start += 1
        if self.steps_after_run_start % self.checkpoint_steps_interval == 0:
            self.save_checkpoint()
            
    def finish_epoch(self):
        """Вызывается после завершения эпохи."""
        self.epochs_finished += 1
        self.save_checkpoint()
        
    def finish_run(self):
        """Вызывается после завершения запуска."""
        self.num_runs_completed += 1
        self.current_run += 1
        self.steps_after_run_start = 0
        self.epochs_finished = 0
        self.optimizer_steps_done = 0
        self.save_checkpoint(finish_run=True)

        self.model = None
        self.optimizer = None
        self.grad_scaler = None

def getitem_wrapper(func: tp.Callable[[int | torch.Tensor], torch.Tensor]):
    def _inner_func(idx: int | torch.Tensor):
        print(f"Accessing {idx=}")

        result = func(idx)

        return result

    return _inner_func


class DummyHandler(LocalStateHandler):
    pass


class TensorMemmapAdapter:
    """
    Wraps memmap numpy object and supports
    """
    def __init__(self, memmap_object: torch.Tensor) -> None:
        self._inner_memmap: torch.Tensor = memmap_object

    def __repr__(self) -> str:
        return repr(self._inner_memmap)

    @getitem_wrapper
    def __getitem__(self, idx: int | torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self._inner_memmap[idx])


def get_tensor_or_wrap_memmap(array_or_memmap: np.ndarray | torch.Tensor | np.memmap) -> torch.Tensor | TensorMemmapAdapter:
    """
    Either returns tensor or wraps tensor logic aroung numpy memmap file
    """
    if isinstance(array_or_memmap, np.ndarray):
        return torch.from_numpy(array_or_memmap)
    elif isinstance(array_or_memmap, np.memmap):
        return torch.from_numpy(array_or_memmap)
    else:
        return array_or_memmap  # for debug can be replaced with TensorMemmapWrapper


def read_memmap(filepath: str,
                shape: tuple[int, ...],
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    number_of_elements = np.prod(shape)

    _, file_extension = os.path.splitext(filepath)
    if file_extension == '.pt':
        return torch.load(f=filepath, weights_only=True)
    return torch.from_file(
        filename=filepath, size=number_of_elements, dtype=dtype, shared=False
    ).reshape(shape)
    # return torch.tensor(np.memmap(filename=filepath, dtype="float32", mode="r", shape=shape))


def get_parameter_groups(model):
    no_weight_decay_names = ['bias', 'normalization', 'frequencies']

    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters()
                       if not any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)]
        },
        {
            'params': [param for name, param in model.named_parameters()
                       if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)],
            'weight_decay': 0
        },
    ]

    return parameter_groups


def _check_dim_and_num_heads_consistency(dim, num_heads):
    if dim % num_heads != 0:
        raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')