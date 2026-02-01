import torch

# 检查CUDA是否可用
print("CUDA可用:", torch.cuda.is_available())
# 查看GPU型号（应显示40HX对应的TU106核心）
print("GPU型号:", torch.cuda.get_device_name(0))
# 查看PyTorch和CUDA版本匹配情况
print("PyTorch版本:", torch.__version__)
print("CUDA版本:", torch.version.cuda)