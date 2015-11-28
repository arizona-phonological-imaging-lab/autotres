from PIL import Image
from glob import glob
from .utils import get_path, image_pattern
import seaborn as sns
import matplotlib.image as mpimg
import numpy as np
import itertools
import os

class Visualizer(object):

    @classmethod
    def show_average_image(cls, d):
        d = get_path(d)
        images = list(itertools.chain(*[glob(os.path.join(d, p)) for p in image_pattern]))
        shape = np.asarray(Image.open(images[0])).shape
        ave = np.zeros(shape)
        N = len(images)
        for image in images:
            im = Image.open(image)#.convert('LA')
            np.array(im, dtype=np.float)
            ave += np.asarray(im)
        # return averaged Image
        return Image.fromarray(np.uint8(ave/N))


    @classmethod
    def visualize_training(cls, tr, savefig=None, show=False):
        """
        Visualize training and validation loss
        """
        sns.plt.plot(tr.data.Epoch.tolist(), tr.data["Training Loss"].tolist(), label="Training Loss")
        sns.plt.plot(tr.data.Epoch.tolist(), tr.data["Validation Loss"].tolist(), label="Validation Loss")
        sns.plt.xlabel("Epochs")
        sns.plt.ylabel("Loss")
        sns.plt.legend(loc="best")
        if savefig:
            sns.plt.savefig(get_path(savefig))
        if show:
            cls.display()

    @classmethod
    def plot_trace(cls, image_dir, trace, label=None):
        """
        Plots a Trace instance
        """
        # don't display a grid
        sns.set_style("whitegrid", {'axes.grid' : False})
        # plot the frame
        image_file = os.path.join(get_path(image_dir), trace.image)
        img = mpimg.imread(image_file)
        #im = plt.imshow(Z, interpolation='bilinear', cmap=cm.gray,origin='lower', extent=[-3,3,-3,3])
        sns.plt.imshow(img, aspect='auto')
        # plot the trace (non-empty coords only)
        x,y = zip(*trace.nonempty)
        # set label
        label = label if label else trace.image
        sns.plt.scatter(x,y, label=label)
        sns.plt.legend(frameon=True)

    @classmethod
    def display(cls):
        sns.plt.show()
