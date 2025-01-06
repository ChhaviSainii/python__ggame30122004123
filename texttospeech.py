from tkinter import *
from tkinter.messagebox import showinfo
from gtts import gTTS
import speech_recognition as sr
import os
mainwindow= Tk()
mainwindow.title('WELCOME TO AADYA TECHNOLOGY')
mainwindow.geometry('400x400')
mainwindow.resizable(0, 0)
mainwindow.configure(bg='#FBEEE6')
def say(text1):
     language = 'en'
     speech = gTTS(text = text1, lang = language, slow = False)
     speech.save("text.mp3")
     os.system("start text.mp3")
 
def TextToSpeech():
    texttospeechwindow = Toplevel(mainwindow)
    texttospeechwindow.title('Text-to-Speech Converter ')
    texttospeechwindow.geometry("400x400")
    texttospeechwindow.configure(bg='#3498DB ')
 
    Label(texttospeechwindow, text='Text-to-Speech Converter ', font=("Times New Roman", 15), bg='#D7BDE2').place(x=40)
 
    text = Text(texttospeechwindow, height=5, width=30, font=12)
    text.place(x=25, y=60)
   
    speakbutton = Button(texttospeechwindow, text='listen', bg='coral', command=lambda: say(str(text.get(1.0, END))))
    speakbutton.place(x=140, y=200)
 
Label(mainwindow, text='  WELCOME TO AADYA TECHNOLOGY CONVERTER',
     font=('Times New Roman', 16), bg='red', wrap=True, wraplength=450).place(x=15, y=0)
 
texttospeechbutton = Button(mainwindow, text='Text-To-Speech Conversion', font=('Times New Roman', 16), bg='#F1948A ', command=TextToSpeech)
texttospeechbutton.place(x=100, y=150)
 
mainwindow.update()
mainwindow.mainloop()        
