DEFAULT_EOS = '</s>'
GET = 0
SEND = 1

from . import registry
(
    build_agent, 
    register_agent, 
    MONOTONIC_AGENT
) = registry.setup_registry('--agent-type')

import importlib
import os
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('eval.agents.' + module)