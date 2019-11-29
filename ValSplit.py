import os
import numpy as np

def main():
    data_folder = '../data/'
    folds = np.loadtxt(data_folder + 'stl10_binary/fold_indices.txt')
    labels = np.load(data_folder + 'train_labels_0.npy')
    use_fold = 0
    fold = folds[use_fold]
    fold = fold.astype(int)

    # Move images from train to val
    for i in fold:
        train_path = "../data/train/{}.png".format(i)
        val_path = "../data/val/{}.png".format(i)
        if os.path.exists(train_path):
            os.rename(train_path, val_path)

    # Move val images into label folders
    for idx, i in enumerate(fold):
        val_path = "../data/val/{}.png".format(i)
        if os.path.exists(val_path):
            os.rename(val_path, "../data/val/{}/{}.png".format(labels[idx], i))

    # Move train images into label folders
    for s in range(10):
        labels = np.load('../data/train_labels_{}.npy'.format(s))
        fold = folds[s]
        fold = fold.astype(int)
        for idx, i in enumerate(fold):
            train_path = "../data/train/{}.png".format(i)
            if os.path.exists(train_path):
                os.rename(train_path, "../data/train/{}/{}.png".format(labels[idx], i))

if __name__ == '__main__':
    main()
