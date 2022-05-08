import numpy as np
import imageio
import pandas as pd
import matplotlib.pyplot as plt
def img_to_input_box(img, epsilon, filename):
    n = img.size
    img = np.float64(img.reshape(-1))/np.float64(255)
    input_box_lb = np.zeros((2*n))
    input_box_ub = np.zeros((2*n))
    input_box_lb[:n] = img
    input_box_ub[:n] = img
    input_box_lb[n:] = img - epsilon
    input_box_ub[n:] = img + epsilon
    # input_box_lb[n:] = np.clip(img - epsilon,0,1)
    # input_box_ub[n:] = np.clip(img + epsilon,0,1)

    with open(filename, 'w') as f:
        for i in range(2*n):
            f.write(f"[{input_box_lb[i]},{input_box_ub[i]}]\n")


def load_png_img(filename):
    img = imageio.imread(filename)
    return img

def load_from_data():
    df = pd.read_csv('../data/mnist_test.csv', header=None)
    with open('images/labels.txt', 'w', encoding='utf-8') as labf:
        for i in range(df.shape[0]):
            print('label:', df.loc[i][0])
            labf.write(f'{int(df.loc[i][0])}\n')
            img = df.loc[i].to_numpy()[1:].reshape((28, 28)).astype(np.uint8)
            plt.imshow(img)
            imageio.imwrite(f'images/{i}.png', img)
            with open(f'constraints/mnist_{i}.txt', 'w') as f:
                f.write('11\n')
                f.write(f'y{df.loc[i][0]+1} l1net_and_max_label\n')


if __name__ == '__main__':
    load_from_data()
    # for i in range(100):
    #     img = load_png_img(f'images/{i}.png')
    #     img_to_input_box(img, 0.04, f'input_boxes/{i}.txt')
