import os
import json
import torch
import models
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch import nn, optim
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--n_epochs', default=1, type=int)
parser.add_argument('--model_dir', default='/tmp', type=str)
parser.add_argument('--network', default='resnet50', type=str)


def run():
    def train():
        net.train()
        print('\nTrain for epoch: %d' % epoch)
        for idx, (inputs, targets) in tqdm(enumerate(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = net(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
        print('Loss %.3f after %d epochs' % (loss, epoch + 1))
        save_path = os.path.join(FLAGS.model_dir, 'model.pt')
        torch.save(net.state_dict(), save_path)

    def test():
        nonlocal best_acc
        net.eval()
        print('\nTest for epoch: %d' % epoch)
        total = 0
        correct = 0
        loss = 0.0
        with torch.no_grad():
            for idx, (inputs, targets) in tqdm(enumerate(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = net(inputs)
                loss += loss_fn(logits, targets)
                _, predictions = logits.max(1)

                total += targets.size(0)
                correct += predictions.eq(targets).sum().item()
        acc = correct / total
        if best_acc < acc:
            best_acc = acc
        print('Loss %.3f, accuracy %.3f after %d epoch' % (loss / (idx + 1), correct * 1.0 / total, epoch))

    n_epochs = FLAGS.n_epochs
    lr = FLAGS.lr
    wd = FLAGS.wd
    best_acc = -1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader = load_data()

    net = models.load_net(FLAGS.network, 10)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.to(device)
    if FLAGS.resume:
        model_path = os.path.join(FLAGS.model_dir, 'model.pt')
        if not os.path.exists(model_path):
            print('Model file `%s` does not exist!' % model_path)
        else:
            print('==>Restoring checkpoint...')
            net.load_state_dict(torch.load(model_path))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    # Print the configuration
    print('===============================================')
    print(json.dumps(FLAGS.__dict__, indent=2))
    print('===============================================')
    for epoch in range(n_epochs):
        train()
        test()
    print('Best accuracy %.3f' % best_acc)


def load_data():
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

    root_dir = os.path.expanduser('~/datasets/cifar10/')
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    return trainloader, testloader


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    run()
