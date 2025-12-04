from typing import Dict, Callable, List
import torch
import torch.nn as nn

class InternalsObserver:
    """
    A system to register hooks and observe internal states of the model.
    """
    def __init__(self):
        self.hooks = []
        self.observations = {}

    def register_hook(self, module: nn.Module, name: str, fn: Callable):
        """
        Registers a forward hook on a module.
        fn signature: (module, input, output) -> None
        """
        def hook_wrapper(module, input, output):
            self.observations[name] = fn(module, input, output)
            
        handle = module.register_forward_hook(hook_wrapper)
        self.hooks.append(handle)

    def clear(self):
        self.observations.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# Global observer instance
observer = InternalsObserver()
