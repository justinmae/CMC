import os
import numpy as np

def main():
    num_folds = 10
    data_folder = '../data/'
    folds = np.loadtxt(data_folder + 'stl10_binary/fold_indices.txt')
    labels = np.load(data_folder + 'train_labels_0.npy')
    use_fold = 0
    fold = folds[use_fold]
    fold = fold.astype(int)

    # Move images from train to val
    val_path = "../data/val/"
    train_path = "../data/train/"
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    for i in fold:
        train_img = "../data/train/{}.png".format(i)
        val_img = "../data/val/{}.png".format(i)
        if os.path.exists(train_img):
            os.rename(train_img, val_img)

    # Move val images into label folders
    for i in range(num_folds):
        if not os.path.exists(val_path+str(i)):
            os.mkdir(val_path+str(i))
    for idx, i in enumerate(fold):
        val_img = "../data/val/{}.png".format(i)
        if os.path.exists(val_img):
            os.rename(val_img, "../data/val/{}/{}.png".format(labels[idx], i))

    # Move train images into label folders
    for i in range(num_folds):
        if not os.path.exists(train_path+str(i)):
            os.mkdir(train_path+str(i))
    for s in range(num_folds):
        labels = np.load('../data/train_labels_{}.npy'.format(s))
        fold = folds[s]
        fold = fold.astype(int)
        for idx, i in enumerate(fold):
            train_img = "../data/train/{}.png".format(i)
            if os.path.exists(train_img):
                os.rename(train_img, "../data/train/{}/{}.png".format(labels[idx], i))

    # Change folder names
    cnt = sum([len(files) for r, d, files in os.walk(data_folder + 'train')])
    if cnt > 2000:
        os.rename(val_path, '../data/tmp')
        os.rename(train_path, val_path)
        os.rename('../data/tmp', train_path)

if __name__ == '__main__':
    main()
