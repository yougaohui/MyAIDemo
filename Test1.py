import torch

print("是否支持 CUDA: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 设备数: ", torch.cuda.device_count())
    print("当前 CUDA 设备: ", torch.cuda.current_device())
    print("CUDA 设备名称: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA 不可用。")
