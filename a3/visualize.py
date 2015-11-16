import seaborn as sns
from .utils import get_path


class Visualizer(object):
    """
    Visualize training and validation loss
    """
    @classmethod
    def visualize_training(cls, tr, savefig=None, show=False):
        sns.plt.plot(tr.data.Epoch.tolist(), tr.data["Training Loss"].tolist(), label="Training Loss")
        sns.plt.plot(tr.data.Epoch.tolist(), tr.data["Validation Loss"].tolist(), label="Validation Loss")
        sns.plt.xlabel("Epochs")
        sns.plt.ylabel("Loss")
        sns.plt.legend(loc="best")
        if savefig:
            sns.plt.savefig(get_path(savefig))
        if show:
            sns.plt.show()
