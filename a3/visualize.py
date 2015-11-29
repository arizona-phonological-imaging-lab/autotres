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
        sns.set_style("whitegrid", {'axes.grid' : False,
                                    'figure.frameon': False,
                                    'legend.frameon': True,
                                    'legend.isaxes': True,
                                    'image.aspect': 'equal'})
        # remove axes
        sns.plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        # plot the frame
        image_file = os.path.join(get_path(image_dir), trace.image)
        img = mpimg.imread(image_file)
        #im = plt.imshow(Z, interpolation='bilinear', cmap=cm.gray,origin='lower', extent=[-3,3,-3,3])
        sns.plt.imshow(img, interpolation='nearest', aspect='equal', zorder=1)
        # plot the trace (non-empty coords only)
        x,y = zip(*trace.nonempty)
        # set label
        label = label if label else trace.image
        #sns.plt.plot(x,y, '-r', lw=1.5, zorder=2)
        #"r-o"
        #'.r-'
        sns.plt.scatter(x,y, c='chartreuse', label=label, zorder=3)
        sns.plt.legend()
        sns.plt.tight_layout()

    @classmethod
    def display(cls):
        sns.plt.show()
