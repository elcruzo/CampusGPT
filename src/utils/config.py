"""
Configuration management utilities
Load and validate configuration files
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        _validate_config(config, config_path)
        
        # Expand environment variables
        config = _expand_env_vars(config)
        
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {str(e)}") from e


def _validate_config(config: Dict[str, Any], config_path: str) -> None:
    """Validate configuration structure"""
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a dictionary: {config_path}")
    
    # Check for required sections based on config type
    if 'model' in config and 'training' in config:
        # Training config validation
        required_sections = ['model', 'data', 'lora', 'training']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in {config_path}")
    
    elif 'model' in config and 'generation' in config:
        # Inference config validation
        required_sections = ['model', 'generation']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in {config_path}")


def _expand_env_vars(config: Any) -> Any:
    """Recursively expand environment variables in configuration"""
    if isinstance(config, dict):
        return {k: _expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)
    else:
        return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., 'model.base_model')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def validate_paths(config: Dict[str, Any]) -> None:
    """
    Validate that all file paths in configuration exist
    
    Args:
        config: Configuration dictionary
        
    Raises:
        FileNotFoundError: If required paths don't exist
    """
    path_keys = [
        'data.train_file',
        'data.validation_file',
        'model.cache_dir'
    ]
    
    for path_key in path_keys:
        path = get_config_value(config, path_key)
        if path and not Path(path).exists():
            # Create directories if they don't exist
            if 'dir' in path_key:
                Path(path).mkdir(parents=True, exist_ok=True)
            # Only warn for missing data files, don't fail
            elif 'data' in path_key:
                print(f"Warning: Data file not found: {path}")