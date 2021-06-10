from tkinter import *

# create root
root = Tk()

# set title to root and block the resizable
root.title("Naive Bayes")
root.resizable(0, 0)

# create frame
width = 750
height = 650
frame = Frame(root, width=str(width), height=str(height))
frame.pack()

# create label
label = Label(frame, text="Search the data base", font=("Montserrat", 20)).place(x=10, y=10)

# browser
Tk().withdraw()
# filename = askopenfilename()

root.mainloop()