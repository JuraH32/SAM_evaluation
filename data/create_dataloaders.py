from torchvision import datasets, transforms
import torch

class DatasetEnum:
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IMAGENET = "imagenet"

def get_dataset_statistics(dataset_name):
    """
    Get the mean and standard deviation for the given dataset.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        tuple: Mean and standard deviation of the dataset.
    """
    if dataset_name == DatasetEnum.CIFAR10:
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    elif dataset_name == DatasetEnum.CIFAR100:
        dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transforms.ToTensor())
    elif dataset_name == DatasetEnum.IMAGENET:
        dataset = datasets.ImageNet(root="./data", split="train", download=True, transform=transforms.ToTensor())
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    data = torch.cat([d[0] for d in torch.utils.data.DataLoader(dataset)])
    return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

def create_dataloaders(dataset_name="cifar10", batch_size=128, num_workers=2, pin_memory=True, shuffle=True):
    """
    Create dataloaders for the given dataset.

    Args:
        dataset_name (str): Name of the dataset to load.
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of workers for the dataloaders.
        pin_memory (bool): Whether to pin memory for the dataloaders.
        shuffle (bool): Whether to shuffle the data in the dataloaders.

    Returns:
        DataLoader: Dataloader for the training set.
        DataLoader: Dataloader for the validation set.
    """

    mean, std = get_dataset_statistics(dataset_name)

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    if dataset_name == DatasetEnum.CIFAR10:
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == DatasetEnum.CIFAR100:
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == DatasetEnum.IMAGENET:
        train_dataset = datasets.ImageNet(root='./data', split='train', download=True, transform=transform)
        test_dataset = datasets.ImageNet(root='./data', split='val', download=True, transform=transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are: {DatasetEnum.CIFAR10}, {DatasetEnum.CIFAR100}, {DatasetEnum.IMAGENET}")

    VALIDATION_SPLIT = 0.1

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * (1 - VALIDATION_SPLIT)), int(len(train_dataset) * VALIDATION_SPLIT)])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader