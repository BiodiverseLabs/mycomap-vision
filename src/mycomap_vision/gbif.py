"""Moved to mycomap_vision.replications.fungitastic.crosswalk (see replications/fungitastic/README.md).

This old path is the same module object, not a copy: `import mycomap_vision.gbif`,
`from mycomap_vision.gbif import ...`, monkeypatching and pickled references all reach
crosswalk.py.
"""

import sys

from .replications.fungitastic import crosswalk as _module

sys.modules[__name__] = _module
