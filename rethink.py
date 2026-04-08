import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import os

class rethinkable_mlp(nn.Module):
    def __init__(self,input_dim=768,hidden_dim=512,output_dim=10):
        super().__init__()
        self.fc_forward1=nn.Linear(input_dim,hidden_dim)
        self.fc_adjust1=nn.Linear(hidden_dim,input_dim)
        self.relu=nn.ReLU()
        self.fc_forward2=nn.Linear(hidden_dim,output_dim)
        self.fc_adjust2=nn.Linear(output_dim,hidden_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)                 # 展平 (B, 784)
        hidden_output = self.relu(self.fc_forward1(x))
        hidden_rethink = self.fc_adjust1(hidden_output)       # 用于重建输入
        ground_output = self.fc_forward2(hidden_output)       # 分类 logits
        ground_rethink = self.fc_adjust2(ground_output)       # 映射回隐藏层维度
        return ground_output, ground_rethink, hidden_output, hidden_rethink
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
epochs = 20
learning_rate = 1e-3

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 标准化参数
])

# 下载并加载 MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型、损失函数、优化器
model = rethinkable_mlp(input_dim=784, hidden_dim=512, output_dim=10).to(device)
criterion_ce = nn.CrossEntropyLoss()          # 分类损失
criterion_mse = nn.MSELoss()                  # 用于 L2 范数平方（MSE 即 ||·||^2 / n）

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------- 训练循环 --------------------
def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        batch_size_curr = images.size(0)

        # 前向传播
        ground_output, ground_rethink, hidden_output, hidden_rethink = model(images)

        # 计算三项损失
        loss_ce = criterion_ce(ground_output, labels)
        # 输入重建损失：||x - hidden_rethink||^2
        loss_recon = criterion_mse(hidden_rethink, images.view(batch_size_curr, -1))
        # 隐藏层一致性损失：||hidden_output - ground_rethink||^2  （修正维度匹配问题）
        loss_hidden = criterion_mse(ground_rethink, hidden_output)

        # 总损失 = CE + 0.25 * 重建损失 + 0.25 * 隐藏一致性损失
        loss = loss_ce + 0.25 * loss_recon + 0.25 * loss_hidden

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size_curr
        _, predicted = torch.max(ground_output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100.0 * correct / total
    print(f'Epoch [{epoch+1:2d}]   Loss: {avg_loss:.4f}   Acc: {accuracy:.2f}%')
    return avg_loss, accuracy

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            batch_size_curr = images.size(0)

            ground_output, ground_rethink, hidden_output, hidden_rethink = model(images)

            loss_ce = criterion_ce(ground_output, labels)
            loss_recon = criterion_mse(hidden_rethink, images.view(batch_size_curr, -1))
            loss_hidden = criterion_mse(ground_rethink, hidden_output)
            loss = loss_ce + 0.25 * loss_recon + 0.25 * loss_hidden

            total_loss += loss.item() * batch_size_curr
            _, predicted = torch.max(ground_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# 开始训练
print("Start training on MNIST...")
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, epoch)
    test_loss, test_acc = evaluate(model, test_loader)
    print(f'        Test Loss: {test_loss:.4f}   Test Acc: {test_acc:.2f}%\n')

print("Training finished.")
save_path = "rethinkable_mlp_mnist.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
