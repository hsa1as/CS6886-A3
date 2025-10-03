import torchvision
import torchvision.transforms as transforms
import torch

class CIFAR10Data:
    def __init__(self, batch_size=256, resize_to=224):
        t_train = transforms.Compose([
            transforms.Resize((resize_to, resize_to)),        # upsample first
            transforms.RandomCrop(resize_to, padding=resize_to//8),  # crop with ~12.5% padding
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),
                                 (0.2023,0.1994,0.2010)),
        ])

        t_test = transforms.Compose([
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),
                                 (0.2023,0.1994,0.2010)),
        ])

        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=t_train
        )
        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=t_test
        )

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size,
            shuffle=True, num_workers=8, pin_memory=True
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=batch_size,
            shuffle=False, num_workers=8, pin_memory=True
        )
