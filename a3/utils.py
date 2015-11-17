import os
import re
import yaml

# The standard directory structure of older APIL projects
apil_old = {
    'trace': {
        'regex': r"""(?x)
            (?P<study>\d+\w+)              # in the example dataset, a 'study' is encoded in the image name as the substring preceding an '_'
            _(?P<frame>\d+)\.(?:jpg|png)   # the frame number
            \.(?P<tracer>\w+)              # the tracer id
            \.traced\.txt$""",
        'conflict': 'list'
        },
    'image': {
        'regex': r"""(?x)
            (?P<study>\d+\w+)
            _(?P<frame>\d+)
            \.(?P<ext>jpg|png)$""",
        'conflict': 'hash'
        },
    'name': {
        'regex': r"""(?x)
            (?P<fname>(?P<study>\d+\w+)
                _(?P<frame>\d+)
                \.(?P<ext>jpg|png)
            )$""",
        }
    }

def get_path(p):
    """
    expand a user-specified path.  Supports "~" shortcut.
    """
    return os.path.normpath(os.path.expanduser(p))
    
class Config(dict):
    """
    A storage class for the yaml-defined network configuration
    """
    def __init__(self, f):
        # Access the keys of the yaml dict as class attributes (i.e. .attributeName)
        super(Config, self).__init__(**self.load_config(f))
        self.__dict__ = self

    def load_config(self, f):
        """
        Expand the path to the yaml-defined conf.
        """
        return yaml.load(open(get_path(f),"r"))
