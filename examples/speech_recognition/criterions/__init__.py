import importlib
import os


# ASG loss requires wav2letter
blacklist = set()
try:
    import wav2letter
except ImportError:
    blacklist.add("ASG_loss.py")

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_") and file not in blacklist:
        criterion_name = file[: file.find(".py")]
        importlib.import_module(
            "examples.speech_recognition.criterions." + criterion_name
        )
