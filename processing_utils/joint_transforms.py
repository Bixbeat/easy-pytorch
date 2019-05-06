from numpy import random
from torchvision.transforms import functional as F


def joint_horizontal_flip(img, lbl):
        """Horizontallyd flips both the target and input label
        Adaptation of the PyTorch flip class to support segmentation case
        """
        if random.random() < 0.5:
            img = F.hflip(img)
            lbl = F.hflip(lbl)
        return img, lbl
    
def joint_vertical_flip(img, lbl):
        """Vertically flips both the target and input label
        Adaptation of the PyTorch flip class to support segmentation case
        """
        if random.random() < 0.5:
            img = F.vflip(img)
            lbl = F.vflip(lbl)
        return img, lbl
    