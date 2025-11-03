"""
Custom DataContainer to replace mmcv.parallel.DataContainer
Minimal implementation with only the features used in LaneDet
"""


class DataContainer:
    """A container for data that can be passed through DataLoader.
    
    Args:
        data: The data to be contained.
        cpu_only (bool): Whether the data should only be kept on CPU.
    """
    
    def __init__(self, data, cpu_only=False):
        self.data = data
        self.cpu_only = cpu_only
