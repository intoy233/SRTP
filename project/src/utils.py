#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Bridge VIV Risk Assessment
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import yaml
import json
import pandas as pd
import numpy as np

def setup_logging(level: str = 'INFO',
                 log_file: Optional[str] = None,
                 console: bool = True,
                 format_string: Optional[str] = None) -> None:
    """
    Setup logging configuration

    Args:
        level: Log level
        log_file: Log file path
        console: Enable console output
        format_string: Custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ['data', 'tasks', 'models']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section missing: {section}")

    return config

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def create_output_directories(directories: List[Union[str, Path]]) -> None:
    """
    Create output directories if they don't exist

    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_data(data_path: Union[str, Path],
             encoding: str = 'utf-8-sig',
             **kwargs) -> pd.DataFrame:
    """
    Load data from CSV file with encoding handling

    Args:
        data_path: Path to data file
        encoding: File encoding
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame with loaded data
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Try different encodings
    encodings = [encoding, 'utf-8', 'gbk', 'latin-1']

    for enc in encodings:
        try:
            df = pd.read_csv(data_path, encoding=enc, **kwargs)
            logging.info(f"Successfully loaded data with encoding: {enc}")
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Failed to load data with any encoding: {encodings}")

def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file

    Args:
        data: Data to save
        file_path: Path to save file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    converted_data = convert_numpy(data)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Set seeds for ML libraries if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.0f}s"

def format_memory(bytes_size: int) -> str:
    """
    Format memory size in bytes to human readable format

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration and return list of issues

    Args:
        config: Configuration dictionary

    Returns:
        List of validation issues
    """
    issues = []

    # Check required sections
    required_sections = ['data', 'tasks', 'models', 'preprocessing']
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required section: {section}")

    # Check data configuration
    if 'data' in config:
        data_config = config['data']
        if 'dataset_path' not in data_config:
            issues.append("Missing dataset_path in data configuration")
        elif not Path(data_config['dataset_path']).exists():
            issues.append(f"Dataset file not found: {data_config['dataset_path']}")

    # Check task configuration
    if 'tasks' in config:
        tasks_config = config['tasks']
        valid_tasks = ['amplitude', 'risk_class', 'viv_occurrence', 'all']
        target_tasks = tasks_config.get('target_tasks', [])

        if not target_tasks:
            issues.append("No target tasks specified")
        else:
            for task in target_tasks:
                if task not in valid_tasks:
                    issues.append(f"Invalid task: {task}. Valid tasks: {valid_tasks}")

    # Check model configuration
    if 'models' in config:
        models_config = config['models']
        if 'baseline_models' not in models_config:
            issues.append("Missing baseline_models in models configuration")

    return issues

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for logging

    Returns:
        Dictionary with system information
    """
    import platform
    import psutil

    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': format_memory(psutil.virtual_memory().total),
        'memory_available': format_memory(psutil.virtual_memory().available)
    }

    # GPU information if available
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu_available'] = True
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
        else:
            info['gpu_available'] = False
    except ImportError:
        info['gpu_available'] = False

    return info

def create_experiment_name(config: Dict[str, Any]) -> str:
    """
    Create experiment name based on configuration

    Args:
        config: Configuration dictionary

    Returns:
        Experiment name string
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks = "_".join(config['tasks'].get('target_tasks', ['unknown']))
    mode = "baseline" if len(config['models']['baseline_models']) <= 5 else "full"

    return f"bridge_viv_{tasks}_{mode}_{timestamp}"

def filter_warnings():
    """Filter common warnings to reduce noise"""
    import warnings

    # Scikit-learn warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

    # XGBoost warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

    # LightGBM warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

    # Matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

    # Pandas warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

class ProgressBar:
    """Simple progress bar for command line"""

    def __init__(self, total: int, width: int = 50, desc: str = ""):
        self.total = total
        self.width = width
        self.desc = desc
        self.current = 0

    def update(self, amount: int = 1):
        """Update progress bar"""
        self.current += amount
        self._display()

    def _display(self):
        """Display progress bar"""
        if self.total == 0:
            return

        progress = self.current / self.total
        filled = int(progress * self.width)
        bar = '█' * filled + '-' * (self.width - filled)

        percent = progress * 100
        print(f'\r{self.desc} |{bar}| {percent:.1f}% ({self.current}/{self.total})', end='')

        if self.current >= self.total:
            print()  # New line when complete

def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are available

    Returns:
        Dictionary with dependency availability
    """
    dependencies = {}

    # Core dependencies
    try:
        import pandas
        dependencies['pandas'] = True
    except ImportError:
        dependencies['pandas'] = False

    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        dependencies['numpy'] = False

    try:
        import sklearn
        dependencies['sklearn'] = True
    except ImportError:
        dependencies['sklearn'] = False

    # ML libraries
    try:
        import xgboost
        dependencies['xgboost'] = True
    except ImportError:
        dependencies['xgboost'] = False

    try:
        import lightgbm
        dependencies['lightgbm'] = True
    except ImportError:
        dependencies['lightgbm'] = False

    try:
        import catboost
        dependencies['catboost'] = True
    except ImportError:
        dependencies['catboost'] = False

    # Deep learning
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        dependencies['torch'] = False

    # Visualization
    try:
        import matplotlib
        dependencies['matplotlib'] = True
    except ImportError:
        dependencies['matplotlib'] = False

    try:
        import seaborn
        dependencies['seaborn'] = True
    except ImportError:
        dependencies['seaborn'] = False

    # Experiment tracking
    try:
        import mlflow
        dependencies['mlflow'] = True
    except ImportError:
        dependencies['mlflow'] = False

    try:
        import wandb
        dependencies['wandb'] = True
    except ImportError:
        dependencies['wandb'] = False

    # Hyperparameter optimization
    try:
        import optuna
        dependencies['optuna'] = True
    except ImportError:
        dependencies['optuna'] = False

    # Interpretability
    try:
        import shap
        dependencies['shap'] = True
    except ImportError:
        dependencies['shap'] = False

    return dependencies

def main():
    """Test utility functions"""
    # Test configuration loading
    config = {
        'data': {'dataset_path': 'test.csv'},
        'tasks': {'target_tasks': ['amplitude']},
        'models': {'baseline_models': ['linear']}
    }

    print("Testing utility functions...")

    # Test progress bar
    print("Progress bar test:")
    pbar = ProgressBar(10, desc="Testing")
    for i in range(10):
        import time
        time.sleep(0.1)
        pbar.update()

    # Test dependencies
    deps = check_dependencies()
    print(f"\nDependencies: {sum(deps.values())}/{len(deps)} available")

    # Test system info
    info = get_system_info()
    print(f"System: {info['platform']}")

    print("All tests completed!")

if __name__ == "__main__":
    main()