"""Moved to mycomap_vision.replications.fungitastic.retrain (see replications/fungitastic/README.md).

This old path is the same module object, not a copy: `import mycomap_vision.picek`,
`from mycomap_vision.picek import ...`, monkeypatching and pickled references all reach
retrain.py.
"""

import sys

from .replications.fungitastic import retrain as _module

sys.modules[__name__] = _module
