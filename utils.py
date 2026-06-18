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


import os
import yaml
from pathlib import Path
import numpy as np

class LocalLogger:
    """Простой локальный логгер без зависимостей от Nirvana."""
    
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
        
    def get_state(self):
        """Возвращает состояние для сохранения в чекпоинт."""
        return {
            'val_metrics': self.val_metrics,
            'test_metrics': self.test_metrics,
            'best_steps': self.best_steps,
            'best_epochs': self.best_epochs,
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
            self.val_metrics[-1] = current_val
            if not self.do_not_evaluate_on_test:
                self.test_metrics[-1] = metrics[f'test {self.metric}']
            self.best_steps[-1] = step
            self.best_epochs[-1] = epoch
            
    def finish_run(self):
        self.save_metrics()
        self.num_runs += 1
        print(f"Finished run {self.current_run}. Best {self.metric}: {self.val_metrics[-1]:.4f}")
        
    def save_metrics(self):
        metrics_file = self.save_dir / 'metrics.yaml'
        metrics = {
            'num_runs': len(self.val_metrics),
            f'val_{self.metric}_mean': np.mean(self.val_metrics),
            f'val_{self.metric}_std': np.std(self.val_metrics, ddof=1) if len(self.val_metrics) > 1 else np.nan,
            f'val_{self.metric}_values': self.val_metrics,
            'best_steps': self.best_steps,
            'best_epochs': self.best_epochs,
        }
        
        if not self.do_not_evaluate_on_test and self.test_metrics is not None:
            metrics[f'test_{self.metric}_mean'] = np.mean(self.test_metrics)
            metrics[f'test_{self.metric}_std'] = np.std(self.test_metrics, ddof=1) if len(self.test_metrics) > 1 else np.nan
            metrics[f'test_{self.metric}_values'] = self.test_metrics
            metrics['best_test_metric'] = np.min(self.test_metrics)
        
        metrics['best_val_metric'] = np.min(self.val_metrics)
        metrics['elapsed_time'] = self.elapsed_time
        metrics['max_memory_allocated_mb'] = self.max_memory_allocated // (2 ** 20)
        
        with open(metrics_file, 'w') as f:
            yaml.safe_dump(metrics, f, sort_keys=False)
            
    def print_metrics_summary(self):
        with open(self.save_dir / 'metrics.yaml', 'r') as f:
            metrics = yaml.safe_load(f)
        
        print(f"Finished {metrics['num_runs']} runs.")
        print(f"Val {self.metric} mean: {metrics[f'val_{self.metric}_mean']:.4f}")
        print(f"Val {self.metric} std: {metrics[f'val_{self.metric}_std']:.4f}")
        
        if not self.do_not_evaluate_on_test and f'test_{self.metric}_mean' in metrics:
            print(f"Test {self.metric} mean: {metrics[f'test_{self.metric}_mean']:.4f}")
            print(f"Test {self.metric} std: {metrics[f'test_{self.metric}_std']:.4f}")


class Logger:
    
    def __init__(self, args, start_from_scratch=True):
        if args.dataset.endswith('.npz'):
            dataset_name = os.path.splitext(os.path.basename(args.dataset))[0].replace('_', '-')
        else:
            dataset_name = args.dataset

        self.metric = args.metric
        self.do_not_evaluate_on_test = args.do_not_evaluate_on_test
        self.val_metrics = []
        self.test_metrics = None if args.do_not_evaluate_on_test else []
        self.best_steps = []
        self.best_epochs = []
        self.num_runs = args.num_runs
        self.cur_run = None


        if start_from_scratch:
            self.save_dir = self.get_save_dir(base_dir=args.save_dir, dataset_name=dataset_name, experiment_name=args.name)

            print(f'Results will be saved to {self.save_dir}.')
            with open(os.path.join(self.save_dir, 'args.yaml'), 'w') as file:
                yaml.safe_dump(vars(args), file, sort_keys=False)
            self.current_run_already_started: bool | None = False
            self.elapsed_time = 0
            self.max_memory_allocated = 0
        else:
            self.save_dir = None  # Will be set during restarting
            self.current_run_already_started = None
            self.elapsed_time = None
            self.max_memory_allocated = None

        self._start_time = perf_counter()

    def set_parameters_from_restarted_job(self, val_metrics, test_metrics, cur_run, best_steps, best_epochs, save_dir, current_run_already_started, elapsed_time, max_memory_allocated):
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.cur_run = cur_run
        self.best_steps = best_steps
        self.best_epochs = best_epochs
        self.save_dir = save_dir
        self.current_run_already_started = current_run_already_started
        self.elapsed_time = elapsed_time
        self.max_memory_allocated = max_memory_allocated
        print(f"Logging will be resumed at save directory {self.save_dir}")

    def _update_timer_and_torch_monitor(self):
        # elapsed is updated_here:
        time_spent_after_last_elapse = perf_counter() - self._start_time
        self.elapsed_time += time_spent_after_last_elapse
        self._start_time = perf_counter()

        self.max_memory_allocated = max(self.max_memory_allocated, torch.cuda.max_memory_allocated())

    def get_parameters_for_checkpoint(self) -> dict[str, tp.Any]:
        self._update_timer_and_torch_monitor()
        return dict(
            val_metrics=self.val_metrics,
            test_metrics=self.test_metrics,
            cur_run=self.cur_run,
            best_steps=self.best_steps,
            best_epochs=self.best_epochs,
            save_dir=self.save_dir,
            current_run_already_started=self.current_run_already_started,
            elapsed_time=self.elapsed_time,
            max_memory_allocated=self.max_memory_allocated,
        )

    def start_run(self, run):
        assert self.current_run_already_started is not None
        self._start_time = perf_counter()

        if not self.current_run_already_started:
            self.current_run_already_started = True
            self.cur_run = run

            self.val_metrics.append(None)
            if not self.do_not_evaluate_on_test:
                self.test_metrics.append(None)

            self.best_steps.append(None)
            self.best_epochs.append(None)

            print(f'Starting run {run}/{self.num_runs}...')
        else:
            print(f'Resuming run {run}/{self.num_runs}...')

    def update_metrics(self, metrics, step, epoch):
        if self.val_metrics[-1] is None or metrics[f'val {self.metric}'] < self.val_metrics[-1]:
            self.val_metrics[-1] = metrics[f'val {self.metric}']
            if not self.do_not_evaluate_on_test:
                self.test_metrics[-1] = metrics[f'test {self.metric}']

            self.best_steps[-1] = step
            self.best_epochs[-1] = epoch

    def finish_run(self):
        self.save_metrics()
        self.current_run_already_started = False

        if self.do_not_evaluate_on_test:
            print(f'Finished run {self.cur_run}. '
                  f'Best val {self.metric}: {self.val_metrics[-1]:.4f} '
                  f'(step {self.best_steps[-1]}, epoch {self.best_epochs[-1]}).\n')

        else:
            print(f'Finished run {self.cur_run}. '
                  f'Best val {self.metric}: {self.val_metrics[-1]:.4f}, '
                  f'corresponding test {self.metric}: {self.test_metrics[-1]:.4f} '
                  f'(step {self.best_steps[-1]}, epoch {self.best_epochs[-1]}).\n')

    def save_metrics(self):
        self._update_timer_and_torch_monitor()
        num_runs = len(self.val_metrics)

        val_metric_mean = np.mean(self.val_metrics).item()
        val_metric_std = np.std(self.val_metrics, ddof=1).item() if len(self.val_metrics) > 1 else np.nan
        best_val_metric = np.min(self.val_metrics).item()

        if not self.do_not_evaluate_on_test:
            test_metric_mean = np.mean(self.test_metrics).item()
            test_metric_std = np.std(self.test_metrics, ddof=1).item() if len(self.test_metrics) > 1 else np.nan
            best_test_metric = np.min(self.test_metrics).item()

            metrics = {
                'num runs': num_runs,
                f'val {self.metric} mean': val_metric_mean,
                f'val {self.metric} std': val_metric_std,
                f'test {self.metric} mean': test_metric_mean,
                f'test {self.metric} std': test_metric_std,
                f'val {self.metric} values': self.val_metrics,
                f'test {self.metric} values': self.test_metrics,
                'elapsed_time': self.elapsed_time,
                'best steps': self.best_steps,
                'best epochs': self.best_epochs,
                'max_memory_allocated': self.max_memory_allocated,
                'max_memory_allocated_mb': self.max_memory_allocated // 2 ** 20,
                'best_val_metric': best_val_metric,
                'best_test_metric': best_test_metric,
            }

        else:
            metrics = {
                'num runs': num_runs,
                f'val {self.metric} mean': val_metric_mean,
                f'val {self.metric} std': val_metric_std,
                f'val {self.metric} values': self.val_metrics,
                'best steps': self.best_steps,
                'best epochs': self.best_epochs,
                'elapsed_time': self.elapsed_time,
                'max_memory_allocated': self.max_memory_allocated,
                'max_memory_allocated_mb': self.max_memory_allocated // 2 ** 20,
                'best_val_metric': best_val_metric,
            }

        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'w') as file:
            yaml.safe_dump(metrics, file, sort_keys=False)

    def print_metrics_summary(self):
        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'r') as file:
            metrics = yaml.safe_load(file)

        print(f'Finished {metrics["num runs"]} runs.')
        print(f'Val {self.metric} mean: {metrics[f"val {self.metric} mean"]:.4f}')
        print(f'Val {self.metric} std: {metrics[f"val {self.metric} std"]:.4f}')

        if not self.do_not_evaluate_on_test:
            print(f'Test {self.metric} mean: {metrics[f"test {self.metric} mean"]:.4f}')
            print(f'Test {self.metric} std: {metrics[f"test {self.metric} std"]:.4f}')

        print(f'Elapsed time: {self.elapsed_time}')
        print(f'Max memory allocated: {self.max_memory_allocated} bytes')
        print(f'Max memory allocated: {self.max_memory_allocated // 2 ** 20} megabytes')

    @staticmethod
    def get_save_dir(base_dir, dataset_name, experiment_name):
        idx = 1
        save_dir = os.path.join(base_dir, dataset_name, f'{experiment_name}_{idx:02d}')
        while os.path.exists(save_dir):
            idx += 1
            save_dir = os.path.join(base_dir, dataset_name, f'{experiment_name}_{idx:02d}')

        os.makedirs(save_dir)

        return save_dir


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


class StateHandler:
    def __init__(self, checkpoint_file_path: Path, checkpoint_dir: Path) -> None:
        self.checkpoint_file_path = checkpoint_file_path
        self.checkpoint_dir = checkpoint_dir

        self.num_runs_completed: int = 0
        self.epochs_finished: int = 0
        self.steps_after_run_start: int = 0
        self.optimizer_steps_done: int = 0
        self.loss: float = 0.0

        self.model: torch.nn.Module = ...
        self.optimizer: torch.optim.Optimizer = ...
        self.grad_scaler: torch.amp.GradScaler = ...
        self.logger: Logger = ...

        self._model_state: TorchStateDict | None = None
        self._optimizer_state: TorchStateDict | None = None
        self._grad_scaler_state: TorchStateDict | None = None
        self._logger_state: dict[str, tp.Any] | None = None

    def load_checkpoint(self, initial_loading: bool = False) -> None:
        pass

    def add_logger(self, logger: Logger) -> None:
        assert self.logger is Ellipsis
        self.logger = logger

        if self._logger_state is not None:
            self.logger.set_parameters_from_restarted_job(**self._logger_state)
            del self._logger_state

    def add_model(self, model: torch.nn.Module) -> None:
        assert self.model is Ellipsis
        self.model = model

        if self._model_state is not None:
            self.model.load_state_dict(self._model_state)
            del self._model_state

    def add_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        assert self.optimizer is Ellipsis
        self.optimizer = optimizer

        if self._optimizer_state is not None:
            self.optimizer.load_state_dict(self._optimizer_state)
            del self._optimizer_state

    def add_grad_scaler(self, scaler: torch.amp.GradScaler) -> None:
        assert self.grad_scaler is Ellipsis
        self.grad_scaler = scaler

        if self._grad_scaler_state is not None:
            self.grad_scaler.load_state_dict(self._grad_scaler_state)
            del self._grad_scaler_state

    def step(self) -> None:
        pass
    
    def finish_epoch(self) -> None:
        pass

    def finish_run(self) -> None:
        del self.model
        del self.optimizer
        del self.grad_scaler

        self.model = ...
        self.optimizer = ...
        self.grad_scaler = ...

    def save_checkpoint(self, finish_run: bool = False) -> None:
        pass


class DummyHandler(StateHandler):
    pass


class LocalStateHandler:
    """Простой локальный обработчик чекпоинтов без зависимостей от Nirvana."""
    
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
