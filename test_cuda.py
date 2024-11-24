import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() > 0
assert torch.cuda.current_device() == 0