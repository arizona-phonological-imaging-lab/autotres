from .utils import get_path
from .translate import TraceFactory
from PIL import Image, ImageFilter
from glob import glob
import shutil
import itertools
import os

# im = Image.open("20110518JF_0209.jpg").convert('LA')
# gaus = im.filter(ImageFilter.GaussianBlur)
# gaus.show()
# ee = im.filter(ImageFilter.EDGE_ENHANCE)
# ee.show()

class ImagePreprocessor(object):
    """
    Preprocess images for training/testing
    """
    image_pattern = ["*.jpg", ".*png"]

    filter_effects = {"gaussian":ImageFilter.GaussianBlur}
    @classmethod
    def copy_dir(cls, indir, outdir):
        indir = get_path(indir)
        outdir = get_path(outdir)
        shutil.copytree(indir, outdir)
        return outdir

    @classmethod
    def add_effect(cls, indir, outdir, effect):
        """
        Copies a directory and applies a Gaussian blur to each image in the copied directory
        """
        effect = effect.lower()
        # copy directory contents and return outdir path
        outdir = cls.copy_dir(indir, outdir)
        images = itertools.chain(*[glob(os.path.join(outdir, p)) for p in ImagePreprocessor.image_pattern])
        eff = ImagePreprocessor.filter_effects[effect]
        # process each image
        for image in images:
            im = Image.open(image)#.convert('LA')
            altered = im.filter(eff)
            altered.save(image)

    @classmethod
    def gaussian_blur(cls, indir, outdir):
        cls.add_effect(indir, outdir, "gaussian")
