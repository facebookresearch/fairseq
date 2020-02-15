import importlib
import os

from fairseq import registry
(
    build_monotonic_attention, 
    register_monotonic_attention, 
    MONOTONIC_ATTENTION_REGISTRY
) = registry.setup_registry('--simul-type')

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        model_name = file[:file.find('.py')]
        importlib.import_module('examples.simultaneous_translation.modules.' + model_name)