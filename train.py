import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

# ------------------------------
# 1. 定义模型
# ------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(9216, 128)  # 64*12*12 = 9216
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # -> [batch, 32, 26, 26]
        x = F.relu(self.conv2(x))   # -> [batch, 64, 24, 24]
        x = F.max_pool2d(x, 2)      # -> [batch, 64, 12, 12]
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------------------
# 2. 训练模型
# ------------------------------
def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 与推理保持一致
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1, 2):  # 只训练 1 epoch 演示
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}")

    # ------------------------------
    # 3. 导出模型 (ONNX)
    # ------------------------------
    dummy_input = torch.randn(1, 1, 28, 28)  # 输入必须与训练一致
    torch.onnx.export(
        model, dummy_input, "model/mnist_cnn.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print("✅ 模型已导出为 model/mnist_cnn.onnx")

if __name__ == "__main__":
    train()
