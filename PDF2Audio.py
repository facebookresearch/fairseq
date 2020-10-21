from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pdftotext
from gtts import gTTS

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filelocation = askopenfilename() # open the dialog GUI

with open(filelocation, "rb") as f:  # open the file in reading (rb) mode and call it f
    pdf = pdftotext.PDF(f)  # store a text version of the pdf file f in pdf variable

string_of_text = ''
for text in pdf:
    string_of_text += text

final_file = gTTS(text=string_of_text, lang='en')  # store file in variable
final_file.save("Generated Speech.mp3")  # save file to computer
