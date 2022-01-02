from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import os

from PIL import Image

class Dataset(Dataset):
    def __init__(self, root, image_size, transform=True):
        self.transform = transform
        self.image_size = image_size
        print(os.path.join(root, "trainA"))
        self.A = sorted(glob.glob(os.path.join(root, "trainA")))
        self.B = sorted(glob.glob(os.path.join(root, "trainB") + "/*.*"))
        print(self.A)
        print(self.B)

    def __getitem__(self, index):
        if self.transform:
            fileA = transform_img(Image.open(self.A[index % len(self.A)]), self.image_size)
            fileB = transform_img(Image.open(self.B[index % len(self.B)]), self.image_size)

        return {"A": fileA, "B": fileB}

    def __len__(self):
        return max(len(self.A), len(self.B))



def transforms(img, image_size):

    #Define transforms
    img_transforms = transforms.compose([
        transforms.Resize(int(image_size), Image.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    return img_transforms(img)

