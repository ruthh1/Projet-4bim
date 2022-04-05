import tkinter as tk
from tkinter.filedialog import asksaveasfile
from PIL import Image, ImageTk
import os

window= tk.Tk()

window.title("Robot Portrait") # En anglais
#mainframe=tk.LabelFrame(window, width=1600, height=1200) # À changer
#canvas.grid(columnspan=3) # lorsqu'on ajoute ceci la fenetre change de dimension
#mainframe.conf
#mainframe.pack()

window.resizable(False, False)

# get the screen dimension
window_width = window.winfo_screenwidth()
window_height = window.winfo_screenheight()

print("window_width=",window_width)
print("window_height =",window_height)

window.geometry(f'{window_width}x{window_height}')


##GRAPHICAL INTERFACE ##

TextFrame=tk.LabelFrame(window,width=window_width,height=0.2*window_height)
#TextFrame.grid(row=0,column=0,columnspan=5,sticky=tk.NW)
TextFrame.configure(bg="grey") # A changer
TextFrame.pack()

Title=tk.Label(TextFrame, text="Robot Portrait",height= 1, width=65, fg="black")
#fontTitle=tkFont.Font(size=16)
Title.configure(font=("Courier",20,"underline"))
Title.place(relx=0.2,rely=0.1)

ToDo=tk.Label(TextFrame, text="Choose the person that looks most like the person you re looking for.",height= 1, width=80, bg="grey" ,fg="black")
#fontTitle=tkFont.Font(size=16)
ToDo.configure(font=("Courier",14))
ToDo.place(relx=0.3,rely=0.25)

nbr_columns=5

ImageFrame1=tk.LabelFrame(window,width=0.2*window_width,height=0.8*window_height)
ImageFrame1.configure(bg="grey")
ImageFrame1.place(x=0,rely=0.2)

ImageFrame2=tk.LabelFrame(window,width=0.2*window_width,height=0.8*window_height)
ImageFrame2.configure(bg="grey")
ImageFrame2.place(relx=0.2,rely=0.2)

ImageFrame3=tk.LabelFrame(window,width=0.2*window_width,height=0.8*window_height)
ImageFrame3.configure(bg="grey")
ImageFrame3.place(relx=0.4,rely=0.2)

ImageFrame4=tk.LabelFrame(window,width=0.2*window_width,height=0.8*window_height)
ImageFrame4.configure(bg="grey")
ImageFrame4.place(relx=0.6,rely=0.2)

ImageFrame5=tk.LabelFrame(window,width=0.2*window_width,height=0.8*window_height)
ImageFrame5.configure(bg="grey")
ImageFrame5.place(relx=0.8,rely=0.2)


#Buttons

Button1=tk.Button(ImageFrame1,command=lambda:[returnperson(0),ChangeColor(Button1)] ,text="Person 1",width=15)
Button1.place(relx=0.17,rely=0.38)
#Button1.pack(side="top")

Button2=tk.Button(ImageFrame2,command=lambda:[returnperson(1),ChangeColor(Button2)] , text="Person 2",width=15)
Button2.place(relx=0.17,rely=0.38)

Button3=tk.Button(ImageFrame3,command=lambda:[returnperson(2), ChangeColor(Button3)] ,text="Person 3",width=15)
Button3.place(relx=0.17,rely=0.38)

Button4=tk.Button(ImageFrame4,command=lambda:[returnperson(3),ChangeColor(Button4)], text="Person 4",width=15)
Button4.place(relx=0.17,rely=0.38)

Button5=tk.Button(ImageFrame5,command=lambda:[returnperson(4),ChangeColor(Button5)],text="Person 5",width=15)
Button5.place(relx=0.17,rely=0.38)

Button6=tk.Button(ImageFrame1,command=lambda:[returnperson(5),ChangeColor(Button6)] ,text="Person 6",width=15)
Button6.place(relx=0.17,rely=0.495)

Button7=tk.Button(ImageFrame2,command=lambda:[returnperson(6),ChangeColor(Button7)] ,text="Person 7",width=15)
Button7.place(relx=0.17,rely=0.495)

Button8=tk.Button(ImageFrame3,command=lambda:[returnperson(7),ChangeColor(Button8)] ,text="Person 8",width=15)
Button8.place(relx=0.17,rely=0.495)

Button9=tk.Button(ImageFrame4,command=lambda:[returnperson(8),ChangeColor(Button9)] ,text="Person 9",width=15)
Button9.place(relx=0.17,rely=0.495)

Button10=tk.Button(ImageFrame5,command=lambda:[returnperson(9),ChangeColor(Button10)] ,text="Person 10",width=15)
Button10.place(relx=0.17,rely=0.495)



#Functions

def ChangeColor(button):
    #if button["fg"]=="black":
    button.configure(fg="blue")

def Images(list_images):

    #new size for the images
    newsize=(200,200)
    #list_images =os.listdir("/Users/yasminemayakamili/Desktop/4BIM/4BIMS2/Projet4BIM/InterfaceGraphique/generated_images")
    for i in range(10):
        path= "/Users/yasminemayakamili/Desktop/4BIM/4BIMS2/Projet4BIM/InterfaceGraphique/generated_images" +'/'+list_images[i]
        Im=Image.open(path)
        Im=Im.resize(newsize)
        Im=ImageTk.PhotoImage(Im)
        # i va de 0 à 9
        j=i+1
        if (j%5==1) :
            Im_widget=tk.Label(ImageFrame1, image=Im)
            Im_widget.image=Im #nécessaire
            Im_widget.place(relx=0.12,rely=j*0.09)
        if (j%5==2) :
            Im_widget=tk.Label(ImageFrame2, image=Im)
            Im_widget.image=Im #nécessaire
            Im_widget.place(relx=0.12,rely=(j-1)*0.09)
        if (j%5==3) :
            Im_widget=tk.Label(ImageFrame3, image=Im)
            Im_widget.image=Im #nécessaire
            Im_widget.place(relx=0.12,rely=(j-2)*0.09)
        if (j%5==4) :
            Im_widget=tk.Label(ImageFrame4, image=Im)
            Im_widget.image=Im #nécessaire
            Im_widget.place(relx=0.12,rely=(j-3)*0.09)
        if (j%5==0) :
            Im_widget=tk.Label(ImageFrame5, image=Im)
            Im_widget.image=Im #nécessaire
            Im_widget.place(relx=0.12,rely=(j-4)*0.09)


def returnperson(x):
    suspects=[]
    print("The choosen one is the number "+str(x+1))
    path= "/Users/yasminemayakamili/Desktop/4BIM/4BIMS2/Projet4BIM/InterfaceGraphique/generated_images" +'/'+list_images[x]
    img=Image.open(path)
    #file = asksaveasfile(mode='w', defaultextension = '.png')
    #if file :
    #    img.save(file)
    img.save("./suspects"+"/"+list_images[x]) # il faudra le empty à chaque fois qu'on recalcule les nouvelles images --> os.remove.
    #return img

#def remove_one_suspect_image(x,button): #not possible car changement de couleur trop rapide entre les deux fonctions dans callback du bouton
    #if (button["fg"]=="blue"):
        #if (os.path.exists("./suspects"+"/"+list_images[x])) :
            #button.configure(fg="black")
            #print(x+1,"was in the file.")
            #os.remove("./suspects"+"/"+list_images[x])


def remove_all_suspects_images():
    ##print("xxxxxxx")
    for x in range(10):
        #print(list_images[x])
        #print(os.path.exists("./suspects"+"/"+list_images[x]))
        if os.path.exists("./suspects"+"/"+list_images[x]):
            print(x+1,"was in the file.")
            os.remove("./suspects"+"/"+list_images[x])




##Main ##
## On commence par display les images qui nous sont donné par algorithme précédent
# Est ce qu'on fait a path or a list ? or first a path and then a list ?
#Il faudrait faire un while --> tant que le nombre d'images de la liste > 0
list_images =os.listdir("/Users/yasminemayakamili/Desktop/4BIM/4BIMS2/Projet4BIM/InterfaceGraphique/generated_images") #Path ou se trouverait les images générée par l'algo
Images(list_images) # function that displays the images en fonction de la liste d'images que nous avons
#il faudrait vider la liste des suspects à chaque fois --> another path et en fonction des nouveaux suspects on génère les nouvelles images pr que la personne choisisse.
#remove_all_suspects_images() # une fois finie et les dossiers envoyé dans algo !!
print()

#il faudrait ajouter un bouton ou une fonction cancel ou on enleverait que ceux de cette sélection. bouton cancel et on regarde les boutons qui sont bleus pr les cancels

button_next = tk.Button(ImageFrame5,text=">>NEXT",fg="black",bg="white",height=2,width=3)#,command=lambda:remove_images()) il faudrait faire une commande qui envoie fichier dans algo pour en sortir de nouveaux
button_next.place(relx=0.6,rely=0.85)
#Raffichage de la fenêtre avec nouvelle liste d'images

remove_all_suspects_images() # une fois finie et les dossiers envoyé dans algo !! # est ce qu'on le met au début aussi ou pas ?? 
print()


#os.remove("./suspects"+"/")





window.mainloop() #ok ca s'affiche
