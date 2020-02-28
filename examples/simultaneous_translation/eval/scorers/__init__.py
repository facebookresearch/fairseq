import sys
from utils import registry
(
    build_scorer, 
    register_scorer, 
    simul_scorer
) = registry.setup_registry('--scorer-type')

import importlib
import os
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('scorers.' + module)