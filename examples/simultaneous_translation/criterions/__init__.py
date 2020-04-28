import importlib
import os

for file in os.listdir(os.path.dirname(__file__)):
    criterion_name = file[: file.find(".py")]
    importlib.import_module(
        "examples.simultaneous_translation.criterions." + criterion_name
    )
