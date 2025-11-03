"""
Custom collate function to replace mmcv.parallel.collate
Handles DataContainer objects and delegates the rest to PyTorch's default_collate
"""

import torch
from torch.utils.data.dataloader import default_collate as torch_default_collate


def collate(batch, samples_per_gpu=1):
    """Custom collate function that handles DataContainer objects.
    
    Args:
        batch: A list of samples from the dataset
        samples_per_gpu: Number of samples per GPU (used for batching)
        
    Returns:
        Collated batch
    """
    # Import here to avoid circular dependency
    from .data_container import DataContainer
    
    if not isinstance(batch, list):
        raise TypeError(f'batch must be a list, but got {type(batch)}')
    
    if len(batch) == 0:
        return batch
    
    # Check if batch contains DataContainer
    if isinstance(batch[0], dict):
        # Handle dict batches
        result = {}
        for key in batch[0]:
            samples = [d[key] for d in batch]
            
            # Check if this field contains DataContainer
            if isinstance(samples[0], DataContainer):
                # Extract data from DataContainer and keep metadata
                cpu_only = samples[0].cpu_only
                data_list = [s.data for s in samples]
                
                # Don't collate if cpu_only is True, just keep as list
                if cpu_only:
                    result[key] = DataContainer(data_list, cpu_only=True)
                else:
                    # Try to collate the data
                    try:
                        collated_data = torch_default_collate(data_list)
                        result[key] = DataContainer(collated_data, cpu_only=False)
                    except:
                        # If collation fails, keep as list in DataContainer
                        result[key] = DataContainer(data_list, cpu_only=True)
            else:
                # Normal field, use default collate
                result[key] = torch_default_collate(samples)
        
        return result
    else:
        # Not a dict batch, use default collate
        return torch_default_collate(batch)
