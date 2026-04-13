import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

class RethinkableResNet(nn.Module):
    def __init__(self, num_classes=10,hidden_dim=512,recon_weight=0.1,consist_weight=0.1):
        super().__init__()
        self.recon_weight = recon_weight
        self.consist_weight = consist_weight

        backbone=torchvision.models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.pool=nn.AdaptiveAvgPool2d((1, 1))        
        self.classifier=nn.Linear(512,num_classes)

        # 反思分支1：特征图重建原始图像（转置卷积解码器）
        self.recon_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, 2, 2)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (B, 64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (B, 32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # (B, 3, 32, 32) 匹配 CIFAR-10 输入尺寸
            nn.Tanh()  # 输出范围 [-1, 1]，与归一化后的图像一致
        )

        self.consistency_head = nn.Linear(num_classes, 512)
    def forward(self,x):
        features=self.encoder(x)
        pooled=self.pool(features).flatten(1)
        logits=self.classifier(pooled)

        recon_img = self.recon_decoder(features)
        consist_feat = self.consistency_head(logits)
        return logits, recon_img, pooled, consist_feat
    def compute_loss(self, logits, recon_img, pooled, consist_feat, images, labels):
        """计算联合损失"""
        ce_loss = nn.functional.cross_entropy(logits, labels)
        recon_loss = nn.functional.mse_loss(recon_img, images)
        consist_loss = nn.functional.mse_loss(consist_feat, pooled)
        total_loss = ce_loss + self.recon_weight * recon_loss + self.consist_weight * consist_loss
        return total_loss, ce_loss, recon_loss, consist_loss
def get_cifar10_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

# -------------------- 训练与评估 --------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        logits, recon_img, pooled, consist_feat = model(images)
        loss, ce_loss, recon_loss, consist_loss = model.compute_loss(logits, recon_img, pooled, consist_feat, images, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = 100.0 * correct / total
    return avg_loss, acc

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits, recon_img, pooled, consist_feat = model(images)
            loss, ce_loss, recon_loss, consist_loss = model.compute_loss(logits, recon_img, pooled, consist_feat, images, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = 100.0 * correct / total
    return avg_loss, acc

# -------------------- 主程序 --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 30
    lr = 1e-3

    print("Preparing CIFAR-10 data...")
    train_loader, test_loader = get_cifar10_loaders(batch_size)

    model = RethinkableResNet(num_classes=10, recon_weight=0.1, consist_weight=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("Start training...")
    best_acc = 0.0
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "rethinkable_resnet_cifar10.pth")
            print(f"  --> Best model saved (Acc: {best_acc:.2f}%)")

    print(f"Training finished. Best test accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()