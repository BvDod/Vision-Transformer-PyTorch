""" Contains all functions regarding file/dataset handling """

import torchvision
import torchvision.transforms as transforms
from functions.customDatasets import xrayDataset


""" Retrieves dataset with name dataset name from disk and returns it"""
def get_dataset(dataset_name: str, print_stats = False):

    
    img_transforms = transforms.Compose([ 
        transforms.RandAugment(),                                    
        transforms.ToTensor(),]
        )

    if dataset_name == "MNIST":
        train = torchvision.datasets.MNIST(root="./datasets/", train=True, transform=img_transforms,  download=True)
        test = torchvision.datasets.MNIST(root="./datasets/", train=False, transform=img_transforms, download=True)
        input_shape = (28, 28)
        channels = 1
    
    if dataset_name == "CIFAR":
        train = torchvision.datasets.CIFAR10(root="./datasets/", train=True, transform=img_transforms,  download=True)
        test = torchvision.datasets.CIFAR10(root="./datasets/", train=False, transform=img_transforms, download=True)
        input_shape = (32, 32)
        channels = 3

    if print_stats:
        print(f"Training dataset shape: {train[0][0].shape}")
        print(f"Test dataset shape: {test[0][0].shape}")
        print("\n")

    return train, test, input_shape, channels

