import os
import torch
import torchvision
import torchvision.transforms as transforms


class ToDeviceWrapper:
    def __init__(self, data, device='cuda'):
        self.data = data
        self.device = device

    def __iter__(self):
        for inputs, labels in self.data:
            yield inputs.to(self.device), labels.to(self.device)

    def __len__(self):
        return len(self.data)


def load_cifar10(batch_size, device):
    print('Loading data...')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root_dir = os.path.expanduser('~/datasets/cifar10')
    print(root_dir)
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return ToDeviceWrapper(trainloader, device), ToDeviceWrapper(testloader, device)
