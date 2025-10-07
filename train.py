import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import time

from model import resnet18


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-100')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int,
                        default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Create model
    model = resnet18(num_classes=100).to(device)
    start_epoch = 0
    best_acc = 0
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {
              checkpoint['epoch']}, best acc: {best_acc:.2f}%")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 25], gamma=0.1)

    def train(epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {
                      batch_idx}/{len(train_loader)} | Loss: {loss.item():.3f}')
        return running_loss / len(train_loader), 100. * correct / total

    def test():
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return running_loss / len(test_loader), 100. * correct / total
    print("Starting training...")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch: {epoch + 1}/{args.epochs}')
        train_loss, train_acc = train(epoch + 1)
        test_loss, test_acc = test()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {
              test_acc:.2f}% | LR: {current_lr:.6f}')
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'resnet18_cifar100_best.pth')
            print(f'New best model saved with accuracy: {best_acc:.2f}%')

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch + 1}.pth')

    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'resnet18_cifar100_final.pth')

    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time/60:.2f} minutes')
    print(f'Best test accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
