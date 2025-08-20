import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import onnxruntime as ort
from openvino.runtime import Core

# -------------------------------
# 1. 定义 PyTorch 模型结构
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.fc1 = nn.Linear(5408, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------
# 2. 加载 PyTorch 模型权重（可选）
# -------------------------------
device = "cpu"
model = SimpleCNN().to(device)
# model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# -------------------------------
# 3. 准备 MNIST 测试数据
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# -------------------------------
# 4. PyTorch 推理
# -------------------------------
def evaluate_pytorch(model, loader):
    correct = 0
    total_time = 0
    with torch.no_grad():
        for data, target in loader:
            start = time.time()
            output = model(data)
            end = time.time()
            total_time += (end - start)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = correct / len(loader.dataset)
    avg_time = total_time / len(loader)
    return acc, avg_time

pytorch_acc, pytorch_time = evaluate_pytorch(model, test_loader)
print(f"PyTorch: Accuracy={pytorch_acc:.4f}, Avg Batch Inference Time={pytorch_time*1000:.2f} ms")

# -------------------------------
# 5. ONNX Runtime 推理
# -------------------------------
ort_session = ort.InferenceSession("mnist_cnn.onnx", providers=["CPUExecutionProvider"])
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

def evaluate_onnxruntime(session, loader):
    correct = 0
    total_time = 0
    for data, target in loader:
        input_data = data.numpy().astype(np.float32)
        start = time.time()
        outputs = session.run([output_name], {input_name: input_data})
        end = time.time()
        total_time += (end - start)
        pred = np.argmax(outputs[0], axis=1)
        correct += (pred == target.numpy()).sum().item()
    acc = correct / len(loader.dataset)
    avg_time = total_time / len(loader)
    return acc, avg_time

onnx_acc, onnx_time = evaluate_onnxruntime(ort_session, test_loader)
print(f"ONNX Runtime: Accuracy={onnx_acc:.4f}, Avg Batch Inference Time={onnx_time*1000:.2f} ms")

# -------------------------------
# 6. OpenVINO 推理
# -------------------------------
core = Core()
model_ov = core.read_model("mnist_cnn.onnx")
compiled_model = core.compile_model(model_ov, "CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def evaluate_openvino(compiled_model, loader):
    correct = 0
    total_time = 0
    for data, target in loader:
        inputs = {input_layer: data.numpy()}
        start = time.time()
        outputs = compiled_model(inputs)[output_layer]
        end = time.time()
        total_time += (end - start)
        pred = outputs.argmax(axis=1)
        correct += (pred == target.numpy()).sum().item()
    acc = correct / len(loader.dataset)
    avg_time = total_time / len(loader)
    return acc, avg_time

ov_acc, ov_time = evaluate_openvino(compiled_model, test_loader)
print(f"OpenVINO: Accuracy={ov_acc:.4f}, Avg Batch Inference Time={ov_time*1000:.2f} ms")
