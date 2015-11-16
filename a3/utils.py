import os
import re


def get_path(p):
    """
    expand a user-specified path.  Supports "~" shortcut.
    """
    return os.path.normpath(os.path.expanduser(p))


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
