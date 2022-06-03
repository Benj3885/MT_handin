import shutil
import argparse
from glob import glob
import random
import os
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default='/home/bjeh/BJEH/datasets/CAM3/Visible/All_chips/')
    parser.add_argument("--out_path", type=str, default='/home/bjeh/BJEH/datasets/CAM3/Visible/randomized_dataset/')
    parser.add_argument("--num_train", type=int, default=10)
    parser.add_argument("--num_val", type=int, default=10)
    args = parser.parse_args()

    # Get all paths
    all_paths = glob(f'{args.in_path}/**/*.jpg', recursive=True)

    # Get all vial ids
    vial_ids = []
    for path in all_paths:
        vial_id = int(re.findall(r'\d+', os.path.basename(path))[0])
        if vial_id not in vial_ids:
            vial_ids.append(vial_id)

    print(f"Total vials: {len(vial_ids)}")

    # Randomize vial ids and split them 
    random.shuffle(vial_ids)

    train_vial_ids = vial_ids[:args.num_train]
    val_vial_ids = vial_ids[args.num_train:args.num_train+args.num_val]
    test_vial_ids = vial_ids[args.num_train+args.num_val:]

    sets = ['train/', 'val/', 'test/']
    ids = [train_vial_ids, val_vial_ids, test_vial_ids]

    for i in range(3):
        out_dir = args.out_path + sets[i]
        os.makedirs(out_dir)
        for path in all_paths:
            vial_id = int(re.findall(r'\d+', os.path.basename(path))[0])
            if vial_id in ids[i]:
                shutil.copyfile(path, out_dir + os.path.basename(path))

    f = open(args.out_path + "info.txt", "a")
    f.write(f"In_path: {args.in_path}\n")
    f.write(f"Train vials: {args.num_train}\n")
    f.write(f"Validation vials: {args.num_val}\n")
    f.write(f"Test vials: {len(test_vial_ids)}")