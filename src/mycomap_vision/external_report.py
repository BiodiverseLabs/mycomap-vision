"""Moved to mycomap_vision.replications.fungitastic.published_report (see replications/fungitastic/README.md).

This old path is the same module object, not a copy: `import mycomap_vision.external_report`,
`from mycomap_vision.external_report import ...`, monkeypatching and pickled references all reach
published_report.py.
"""

import sys

from .replications.fungitastic import published_report as _module

sys.modules[__name__] = _module
