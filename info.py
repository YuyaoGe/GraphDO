"""
Model information and mapping configuration for GraphDO.
This file contains the mapping between model shortcuts and their actual paths.
"""

import os

# Model mapping configuration
MODEL_MAPPING = {
    # Local HuggingFace models
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "/shared/models/DeepSeek-R1-Distill-Qwen-1.5B",
    "Llama-2-7b-chat-hf": "/shared/models/Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf": "/shared/models/Llama-2-13b-chat-hf",
    "Meta-Llama-3-8B": "/shared/models/Meta-Llama-3-8B",
    "Mistral-7B-v0.3": "/shared/models/Mistral-7B-v0.3",
    "Qwen2-7B": "/shared/models/Qwen2-7B",
    "vicuna-7b-v1.5": "/shared/models/vicuna-7b-v1.5",
}

def get_model_path(model_name):
    """
    Get the actual path for a model given its shortcut name.

    Args:
        model_name (str): Model shortcut name

    Returns:
        str: Actual model path or the original name if not found
    """
    if model_name in MODEL_MAPPING:
        return MODEL_MAPPING[model_name]
    else:
        # Try to use as direct path/name
        return model_name

def is_local_model(model_name):
    """
    Check if a model is a registered local HuggingFace model.

    Args:
        model_name (str): Model name

    Returns:
        bool: True if it's a registered local model, False otherwise
    """
    return model_name in MODEL_MAPPING

def list_available_models():
    """
    List all available models.

    Returns:
        dict: Dictionary with local models
    """
    return {
        "local_models": list(MODEL_MAPPING.keys()),
    }

def validate_model_path(model_name):
    """
    Validate if a local model path exists.

    Args:
        model_name (str): Model name

    Returns:
        bool: True if path exists
    """
    model_path = get_model_path(model_name)
    return os.path.exists(model_path)
