import importlib
import os


# ASG loss requires flashlight bindings
files_to_skip = set()
try:
    import flashlight.lib.sequence.criterion
except ImportError:
    files_to_skip.add("ASG_loss.py")

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_") and file not in files_to_skip:
        criterion_name = file[: file.find(".py")]
        importlib.import_module(
            "examples.speech_recognition.criterions." + criterion_name
        )
