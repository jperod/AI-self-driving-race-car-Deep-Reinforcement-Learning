import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from skimage import color, transform, io


class processimage:
    def process_image(obs):
        #uncomment to see original image
        # plt.imshow(obs)
        # plt.show()

        obs1 = obs.astype(np.uint8)
        obs_gray = color.rgb2gray(obs1)
        # obs_gray[abs(obs_gray - 0.60116) < 0.1] = 1
        obs_gray[84:95,0:12] = 0
        obs_gray[abs(obs_gray - 0.68616) < 0.0001] = 1
        obs_gray[abs(obs_gray - 0.75630) < 0.0001] = 1
        #uncomment to see pre processed image
        # plt.imshow(obs_gray, cmap='gray')
        # plt.show()

        #Set values between -1 and 1 for input normalization
        return 2 * obs_gray - 1