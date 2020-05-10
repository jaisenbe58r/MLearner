"""
Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import os
import matplotlib.pyplot as plt

import tensorflow as tf


def load_image(filename):

    return tf.cast(tf.image.decode_png(
                        tf.io.read_file(filename), channels=1), tf.float32)


def plot_image(img, name="Image", save=False, logdir_report="/images"):
    """Gráfico Imagen."""
    plt.imshow(img[:, :, 0], cmap='gray', interpolation='none')
    plt.title(name)
    plt.axis('off')
    plt.show()
    if save:
        name = name + ".png"
        filename = os.path.join(logdir_report, name)
        plt.savefig(filename)
        plt.close()
    plt.show()


def plot_image2(img1, img2, title="Images", save=False, logdir_report="/images"):
    """Gráfico Imagen."""
    fig, axs = plt.subplots(1, 2)
    ax = axs.flatten()
    fig.suptitle(title, fontsize=20)

    ax[0].axis('off')
    ax[0].imshow(img1[:, :, 0], cmap='gray', interpolation='none')

    ax[1].axis('off')
    ax[1].imshow(img2[:, :, 0], cmap='gray', interpolation='none')

    plt.show()
    if save:
        name = title + ".png"
        filename = os.path.join(logdir_report, name)
        fig.savefig(filename)
        fig.close()
    plt.show()
