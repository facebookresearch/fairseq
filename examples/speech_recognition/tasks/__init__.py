import importlib
import os


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        task_name = file[: file.find(".py")]
        importlib.import_module("examples.speech_recognition.tasks." + task_name)
