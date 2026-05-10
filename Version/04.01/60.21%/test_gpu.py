import torch

print(f"当前 PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"系统检测到 {gpu_count} 张显卡。")
    for i in range(gpu_count):
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
        # 顺便看看显存容量 (粗略值)
        vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   显存大小: {vram:.2f} GB")
else:
    print("糟糕，没有检测到 GPU，请检查 PyTorch 安装！")