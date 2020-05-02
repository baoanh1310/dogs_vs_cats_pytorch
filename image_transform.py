from lib import *
from config import *

class ImageTransform():
    def __init__(self, resize, mean):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomRotation(40)
                transforms.RandomAffine(translate=(0.2, 0.2), shear=0.2),
                transforms.RandomHorizontalFlip(1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean)
                ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean)
                ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)