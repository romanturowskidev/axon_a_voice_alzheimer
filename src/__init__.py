

# Marks `src/` as a Python package



"""
Initialization file for the src package.
This package contains modules for:
- Application (`app/`): API and UI for interaction.
- Configuration (`config/`): Configuration and logging settings.
- Evaluation (`evaluation/`): Model evaluation, explainability, and validation.
- Models (`models/`): Machine learning models (CNN, RNN, training scripts).
- Utilities (`utils/`): Feature extraction, preprocessing, and model utilities.
"""

import os
import sys

# Add subdirectories to the system path for easier imports
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PACKAGE_ROOT, "app"))
sys.path.append(os.path.join(PACKAGE_ROOT, "config"))
sys.path.append(os.path.join(PACKAGE_ROOT, "evaluation"))
sys.path.append(os.path.join(PACKAGE_ROOT, "models"))
sys.path.append(os.path.join(PACKAGE_ROOT, "utils"))

__version__ = "1.0.0"
__author__ = "Roman Turowski"
__license__ = "MIT"


# This file:

# Provides documentation for the package structure.
# Adds subdirectories (app, config, evaluation, models, utils) to the system path for easier imports.
# Defines metadata such as version, author, and license.