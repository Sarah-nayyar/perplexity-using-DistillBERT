import os
import torch

# This forces Python to look at the internal torch libs
torch_bin = os.path.join(os.path.dirname(torch.__file__), 'bin')
os.add_dll_directory(torch_bin)

print(torch.cuda.is_available())