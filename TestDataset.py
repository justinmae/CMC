import re
import os
import numpy as np

def main():
    num_classes = 10
    data_folder = '../data/'
    images = np.load(data_folder + 'test_images.npy')
    images = [re.findall(r'\d+', f)[0] for f in images]
    labels = np.load(data_folder + 'test_labels.npy')
    test_path = "../data/test/"

    # Move val images into label folders
    for i in range(num_classes):
        if not os.path.exists(test_path+str(i)):
            os.mkdir(test_path+str(i))
    for idx, i in enumerate(images):
        test_img = "../data/test/{}.png".format(i)
        if os.path.exists(test_img):
            os.rename(test_img, "../data/test/{}/{}.png".format(labels[idx], i))

if __name__ == '__main__':
    main()
