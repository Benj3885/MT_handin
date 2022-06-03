from numpy import Inf
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import random
from cmath import inf
import re
import os

def get_imgs_per_vial(defect_paths, max_defect_vials):
    reduced_defect_paths = []
    defect_vials_count = 0
    vial_id = -1
    for defect_path in defect_paths:
        path_vial_id = int(re.findall(r'\d+', os.path.basename(defect_path))[0])
        if vial_id != path_vial_id:
            vial_id = path_vial_id
            defect_vials_count + 1
            if defect_vials_count > max_defect_vials:
                break
        reduced_defect_paths.append(defect_path)
    
    return reduced_defect_paths

class ClassificationDataset(Dataset):
    def __init__(self, data_paths, transform, good_per_defect, max_defect_imgs=inf, max_defect_vials=inf):
        self.imgs = []

        good_paths = []
        for directory in data_paths[0]:
            for img_path in glob(f'{directory}/**/*.jpg', recursive=True):
                good_paths.append(img_path)

        defect_paths = []
        for directory in data_paths[1]:
            for img_path in glob(f'{directory}/**/*.jpg', recursive=True):
                defect_paths.append(img_path)

        defect_paths.sort()

        if max_defect_imgs != inf:
            defect_paths = get_imgs_per_vial(defect_paths, max_defect_vials)

        # Uses the number of defects and how many good vials per defect to 
        # add the correct number of vials 
        for sample in good_paths:
            self.imgs.append((sample, 0))

        #random.shuffle(random_indexes)
        for i, random_i in enumerate(list(range(min(len(defect_paths), max_defect_imgs)))):
            if i >= max_defect_imgs:
                break
            self.imgs.append((defect_paths[random_i], 1))

        self.length = len(self.imgs)
        self.transform = transform

        print(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        img_in = Image.open(img_path)
        tensor = self.transform(img_in)
        return tensor, label, img_path


class DefectRemovedDataset(Dataset):
    def __init__(self, data_paths, path_random, transform, max_defect_imgs=inf, max_defect_vials=inf, p_random_vial=0):
        self.imgs = []
        self.good_path = data_paths[0][0]
        self.defect_path = data_paths[1][0]

        defect_paths = []
        for directory in data_paths[1]:
            for img_path in glob(f'{directory}/**/*.jpg', recursive=True):
                defect_paths.append(img_path)

        defect_paths.sort()

        if max_defect_imgs != inf:
            defect_paths = get_imgs_per_vial(defect_paths, max_defect_vials)

        num_defect_images = len(defect_paths)
        for i, random_i in enumerate(range(num_defect_images)):
            if i >= max_defect_imgs:
                break
            self.imgs.append(os.path.basename(defect_paths[random_i]))

        self.length = len(self.imgs)
        self.transform = transform

        print(self.length)

        self.random_vials = []
        for img_path in glob(f'{path_random}/**/*.jpg', recursive=True):
                self.random_vials.append(img_path)
        self.p_random_vial = p_random_vial

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fn = self.imgs[idx]
        img_good_path = (self.good_path + fn if random.random() > self.p_random_vial else random.choice(self.random_vials))
        img_good = Image.open(img_good_path)
        img_defect = Image.open(self.defect_path + fn)

        tensor_good = self.transform(img_good)
        tensor_defect = self.transform(img_defect)



        return tensor_good, tensor_defect, 0, 1


class RandomGoodDataset(Dataset):
    def __init__(self, data_paths, path_random, transform, max_defect_imgs=inf, max_defect_vials=inf, p_random_vial=0):
        self.imgs = []
        self.good_path = data_paths[0][0]
        self.defect_path = data_paths[1][0]

        defect_paths = []
        for img_path in glob(f'{self.defect_path}/**/*.jpg', recursive=True):
            defect_paths.append(img_path)

        defect_paths.sort()

        if max_defect_imgs != inf:
            defect_paths = get_imgs_per_vial(defect_paths, max_defect_vials)

        num_defect_images = len(defect_paths)
        for i, j in enumerate(range(num_defect_images)):
            if i >= max_defect_imgs:
                break
            self.imgs.append((defect_paths[j], 1))

        defect_paths = []
        for img_path in glob(f'{self.good_path}/**/*.jpg', recursive=True):
            self.imgs.append((img_path, 0))


        self.length = len(self.imgs)
        self.transform = transform

        print(self.length)

        path_random = "/Data/Real/CAM3/Good/current_version/train/A"
        self.random_vials = []
        for img_path in glob(f'{path_random}/**/*.jpg', recursive=True):
                self.random_vials.append(img_path)
        self.p_random_vial = p_random_vial

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fn, label = self.imgs[idx]
        #if label == 0:
            #fn = random.choice(self.random_vials)
        #img_good_path = (self.good_path + fn if random.random() > self.p_random_vial else random.choice(self.random_vials))
        img = Image.open(fn)

        tensor = self.transform(img)



        return tensor, label

class FixedBatchDataset(Dataset):
    def __init__(self, data_paths, path_random, transform, max_defect_imgs=inf, max_defect_vials=inf, p_random_vial=0):
        self.imgs = []
        self.good_path = data_paths[0][0]
        self.defect_path = data_paths[1][0]

        defect_paths = []
        for img_path in glob(f'{self.defect_path}/**/*.jpg', recursive=True):
            defect_paths.append(img_path)

        defect_paths.sort()

        if max_defect_imgs != inf:
            defect_paths = get_imgs_per_vial(defect_paths, max_defect_vials)

        num_defect_images = len(defect_paths)
        good_paths = glob(f'{self.good_path}/**/*.jpg', recursive=True)
        for i in range(num_defect_images):
            if i >= max_defect_imgs:
                break
            self.imgs.append((good_paths[i], defect_paths[i]))



        self.length = len(self.imgs)
        self.transform = transform

        print(self.length)

        path_random = "/Data/Real/CAM3/Good/current_version/train"
        self.random_vials = []
        for img_path in glob(f'{path_random}/**/*.jpg', recursive=True):
                self.random_vials.append(img_path)
        self.p_random_vial = p_random_vial

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fn1, fn2 = self.imgs[idx]
        #img_good_path = (self.good_path + fn if random.random() > self.p_random_vial else random.choice(self.random_vials))
        img_good = Image.open(fn1)
        img_defect = Image.open(fn2)

        tensor_good = self.transform(img_good)
        tensor_defect = self.transform(img_defect)

        return tensor_good, tensor_defect, 0, 1