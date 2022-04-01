import tkinter as tk
from PIL import Image, ImageTk

window= tk.Tk()
window.geometry("1600x1200")
window.title("Portrait Robot") # En anglais
#mainframe=tk.LabelFrame(window, width=1600, height=1200) # À changer
#canvas.grid(columnspan=3) # lorsqu'on ajoute ceci la fenetre change de dimension
#mainframe.conf
#mainframe.pack()

TextFrame=tk.LabelFrame(window,width=1600,height=200)
TextFrame.configure(bg="grey") # A changer
TextFrame.pack()

Title=tk.Label(TextFrame, text="Portrait Robot",height= 1, width=65, fg="black")
#fontTitle=tkFont.Font(size=16)
Title.configure(font=("Courier",20,"underline"))
Title.place(relx=0.2,rely=0.1)

ToDo=tk.Label(TextFrame, text="Choose the person that looks most like the person you re looking for.",height= 1, width=80, bg="grey" ,fg="black")
#fontTitle=tkFont.Font(size=16)
ToDo.configure(font=("Courier",14))
ToDo.place(relx=0.3,rely=0.25)

ImageFrame1=tk.LabelFrame(window,width=320,height=1000)
ImageFrame1.configure(bg="grey")
ImageFrame1.place(x=0,y=200)

ImageFrame2=tk.LabelFrame(window,width=320,height=1000)
ImageFrame2.configure(bg="grey")
ImageFrame2.place(relx=0.2,y=200)

ImageFrame3=tk.LabelFrame(window,width=320,height=1000)
ImageFrame3.configure(bg="grey")
ImageFrame3.place(relx=0.4,y=200)

ImageFrame4=tk.LabelFrame(window,width=320,height=1000)
ImageFrame4.configure(bg="grey")
ImageFrame4.place(relx=0.6,y=200)

ImageFrame5=tk.LabelFrame(window,width=320,height=1000)
ImageFrame5.configure(bg="grey")
ImageFrame5.place(relx=0.8,y=200)

newsize=(200,200)

#5 images à charger # faudrait faire une fonction
Im1=Image.open("CVForum-95.jpg")
Im1=Im1.resize(newsize)
Im1=ImageTk.PhotoImage(Im1) # Conversion en image Tk

Im2=Image.open("CVForum-96.jpg")
Im2=Im2.resize(newsize)
Im2=ImageTk.PhotoImage(Im2) # Conversion en image Tk

Im3=Image.open("CVForum-97.jpg")
Im3=Im3.resize(newsize)
Im3=ImageTk.PhotoImage(Im3) # Conversion en image Tk

Im4=Image.open("CVForum-98.jpg")
Im4=Im4.resize(newsize)
Im4=ImageTk.PhotoImage(Im4) # Conversion en image Tk

Im5=Image.open("Maya.jpeg")
Im5=Im5.resize(newsize)
Im5=ImageTk.PhotoImage(Im5) # Conversion en image Tk


#Image en widget
Im1_widget=tk.Label(ImageFrame1, image=Im1)
Im1_widget.image=Im1 #nécessaire

Im2_widget=tk.Label(ImageFrame2, image=Im2)
Im2_widget.image=Im2 #nécessaire

Im3_widget=tk.Label(ImageFrame3, image=Im3)
Im3_widget.image=Im3 #nécessaire

Im4_widget=tk.Label(ImageFrame4, image=Im4)
Im4_widget.image=Im4 #nécessaire

Im5_widget=tk.Label(ImageFrame5, image=Im5)
Im5_widget.image=Im5 #nécessaire

#Button

Button1=tk.Button(ImageFrame1,command=lambda:returnperson() ,text="Person 1",width=15)
Button1.place(relx=0.17,rely=0.32)
#Button1.pack(side="top")

Button2=tk.Button(ImageFrame2,text="Person 2",width=15)
Button2.place(relx=0.17,rely=0.32)

Button3=tk.Button(ImageFrame3,text="Person 3",width=15)
Button3.place(relx=0.17,rely=0.32)

Button4=tk.Button(ImageFrame4,text="Person 4",width=15)
Button4.place(relx=0.17,rely=0.32)

Button5=tk.Button(ImageFrame5,text="Person 5",width=15)
Button5.place(relx=0.17,rely=0.32)

#Si on clique sur le bouton alors ?

def returnperson():
    print("The choosen one is")

#Affichage
Im1_widget.place(relx=0.12,rely=0.1)
Im2_widget.place(relx=0.12,rely=0.1)
Im3_widget.place(relx=0.12,rely=0.1)
Im4_widget.place(relx=0.12,rely=0.1)
Im5_widget.place(relx=0.12,rely=0.1)


window.mainloop() #ok ca s'affiche
