"""Train CIFAR10 with PyTorch."""
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler

import torchvision
import torchvision.transforms as transforms
# from model_tokendrop import CIFARViT
from model_tokendrop_graphcut import CIFARViT


import wandb
import random
# from fvcore.nn import FlopCountAnalysis, flop_count_table


wandb.init(
    entity="cppr",
    # set the wandb project where this run will be logged
    project="groupingvit"
)

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

# Optimizer options
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")

parser.add_argument("--eta", default=1.0, type=float, help="learning rate")
parser.add_argument("--bs", default=128, type=int, help="batch size")

parser.add_argument(
    "--epochs", default=200, type=int, help="number of classes in the dataset"
)

# Transformer options
parser.add_argument(
    "--embed_dim",
    default=256,
    type=int,
    help="embedding dimension of the transformer",
)
parser.add_argument(
    "--n_head", default=8, type=int, help="number of heads in the transformer"
)
parser.add_argument(
    "--depth", default=4, type=int, help="number of transformer blocks"
)
parser.add_argument(
    "--patch_size", default=(4, 4), help="patch size in patchification"
)

parser.add_argument(
    "--share_mini",
    default=False,
    type=bool,
    help="whether to share mini across layers",
)
parser.add_argument("--image_size", default=(32, 32), help="input image size")
parser.add_argument(
    "--num_classes",
    default=10,
    type=int,
    help="number of classes in the dataset",
)


args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

# Will downloaded and save the dataset if needed
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.bs, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.bs, shuffle=False, num_workers=2
)

model = CIFARViT(
    embed_dim=args.embed_dim,
    n_head=args.n_head,
    depth=args.depth,
    patch_size=args.patch_size,
    image_size=args.image_size,
    num_classes=args.num_classes,
    keep_eta = args.eta,
    # share_mini = args.share_mini,
)

model = model.to(device)
# print(flop_count_table(FlopCountAnalysis(model, torch.rand(1, 3, 32, 32).to(device))))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


# Training
def train(epoch):
    print("\nTraining Epoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # We do not need to specify AMP autocast in forward pass here since
        # that is taken care of already in the forward of individual modules.
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('batch:', batch_idx+1, '/', len(trainloader), "loss:", loss.item())

    print(f"Training Accuracy:{100.*correct/total: 0.2f}")
    print(f"Training Loss:{train_loss/(batch_idx+1): 0.3f}")
    wandb.log({
        "train/acc": 100.*correct/total, 
        "train/loss": train_loss/(batch_idx+1)})


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("\nTesting Epoch: %d" % epoch)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"Test Accuracy:{100.*correct/total: 0.2f}")
        print(f"Test Loss:{test_loss/(batch_idx+1): 0.3f}")

        wandb.log({
        "test/acc": 100.*correct/total, 
        "test/loss": test_loss/(batch_idx+1)})


for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step(epoch - 1)
