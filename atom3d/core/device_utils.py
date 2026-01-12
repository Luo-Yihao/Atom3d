"""
Device utilities for Atom3D

Provides unified device resolution across all modules.
Priority: input tensors > explicit device arg > default ('cuda:0' or 'cpu')
"""

from typing import Optional, Union
import torch


def resolve_device(
    *tensors: Optional[torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    default: str = 'cuda'
) -> torch.device:
    """
    Resolve device with priority: input tensors > explicit device > default.
    
    This ensures that when users pass tensors on a specific device (e.g., cuda:3),
    the computation stays on that device instead of unexpectedly moving to cuda:0.
    
    Args:
        *tensors: Input tensors to infer device from (first non-None wins)
        device: Explicitly specified device (overrides tensor inference if not None)
        default: Default device if no tensors and no explicit device
    
    Returns:
        torch.device: Resolved device
    
    Examples:
        >>> # Infer from tensor
        >>> t = torch.randn(3, device='cuda:2')
        >>> resolve_device(t)  # Returns device('cuda:2')
        
        >>> # Explicit override
        >>> resolve_device(t, device='cuda:0')  # Returns device('cuda:0')
        
        >>> # Default fallback
        >>> resolve_device(None, None)  # Returns device('cuda') or device('cpu')
    """
    # Priority 1: Explicit device argument (if provided)
    if device is not None:
        if isinstance(device, torch.device):
            return device
        return torch.device(device)
    
    # Priority 2: Infer from input tensors
    for tensor in tensors:
        if tensor is not None and isinstance(tensor, torch.Tensor):
            return tensor.device
    
    # Priority 3: Default device
    if default == 'cuda' and not torch.cuda.is_available():
        return torch.device('cpu')
    
    return torch.device(default)


def ensure_same_device(*tensors: torch.Tensor, target_device: Optional[torch.device] = None) -> tuple:
    """
    Ensure all tensors are on the same device.
    
    Args:
        *tensors: Tensors to check/move
        target_device: Target device (if None, uses first tensor's device)
    
    Returns:
        Tuple of tensors on the same device
    
    Raises:
        ValueError: If no tensors provided and no target_device
    """
    if not tensors:
        raise ValueError("At least one tensor required")
    
    if target_device is None:
        target_device = tensors[0].device
    
    return tuple(t.to(target_device) if t.device != target_device else t for t in tensors)


def get_default_cuda_device() -> torch.device:
    """
    Get the default CUDA device.
    
    Returns cuda:0 if CUDA is available, otherwise cpu.
    """
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')
