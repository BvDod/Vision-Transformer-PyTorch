import glob

from torch.utils.data import Dataset
from PIL import Image

class xrayDataset(Dataset):
    """Custom class to load the x-ray data as a torchvision dataset"""

    def __init__(self, root, transform=None, train=True):
        self.transform = transform
        self.data_location = root + "Data/train/*" if train else root + "Data/test/*"
        self.data = []
        # Load all images in the target file folder
        for f in glob.iglob(self.data_location):
            self.data.append(Image.open(f))
        # Labels dont matter for now, all 1
        self.targets = [1 for i in range(len(self.data))]


    def __len__(self):
        return len(self.data)

    # Used by the dataloader to retrieve batches
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    