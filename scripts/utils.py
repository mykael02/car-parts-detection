import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_sample_images(images, titles=None, n=5):
    plt.figure(figsize=(15,3))
    for i in range(min(n, len(images))):
        plt.subplot(1, n, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()