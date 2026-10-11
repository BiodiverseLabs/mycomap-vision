"""Moved to mycomap_vision.replications.fungitastic.published (see replications/fungitastic/README.md).

This old path is the same module object, not a copy: `import mycomap_vision.external`,
`from mycomap_vision.external import ...`, monkeypatching and pickled references all reach
published.py.
"""

import sys

from .replications.fungitastic import published as _module

sys.modules[__name__] = _module
