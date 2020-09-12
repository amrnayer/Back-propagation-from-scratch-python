from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Combobox
from BackPropagation_Task_CS import neuralnetwork
import numpy as np
p=""
def run():
    app = Tk()
    app.geometry('500x200')
    #ComboBoxes
    featurs=["X1","X2","X3","X4"]
    classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    act=['Sigmoind','Hyperbolic_tangent']
    n_hidden=Entry(app)
    n_hidden.place(x=60,y=10)
    n_neurons=Entry(app)
    n_neurons.place(x=70,y=50)
    act_function=Combobox(app,value=act,width=20)
    act_function.place(x=60,y=90)
    learningrate = Entry(app)
    learningrate.place(x=80, y=130, width=140)
    epochs = Entry(app)
    epochs.place(x=100, y=170, width=140)
    var1 = IntVar()
    bias=Checkbutton(app,  variable=var1)
    bias.place(x=240,y=10)
    X1 = Entry(app)
    X1.place(x=350, y=10, width=140)
    X2 = Entry(app)
    X2.place(x=350, y=50, width=140)
    X3 = Entry(app)
    X3.place(x=350, y=90, width=140)
    X4 = Entry(app)
    X4.place(x=350, y=130, width=140)
    #lables
    lable1=Label(app,text="n_hidden")
    lable1.place(x=5,y=10)
    lable2=Label(app,text="n_neurons")
    lable2.place(x=5,y=50)
    lable3=Label(app,text="function")
    lable3.place(x=5,y=90)
    lable4=Label(app,text="LearningRate")
    lable4.place(x=5,y=130)
    lable5=Label(app,text="Num_of epochs")
    lable5.place(x=5,y=170)
    lable7=Label(app,text="Bais")
    lable7.place(x=220,y=10)
    lable8 = Label(app, text="X1")
    lable8.place(x=300, y=10)
    lable9 = Label(app, text="X2")
    lable9.place(x=300, y=50)
    lable10 = Label(app, text="X3")
    lable10.place(x=300, y=90)
    lable11 = Label(app, text="X4")
    lable11.place(x=300, y=130)
    def Get_info():
        try:
            n_neu=list
            n_hi=int(n_hidden.get())
            n_neu=n_neurons.get().split(',')
            n_neu=np.insert(n_neu,0,len(featurs),axis=0)
            n_neu=np.append(n_neu,[len(classes)])
            n=[int(i) for i in n_neu]
            fu=act_function.get()
            LearningRate = float(learningrate.get())
            Epochs =int(epochs.get())
            b = var1.get()
            global p
            p=neuralnetwork("IrisData.txt",featurs,classes,b,n_hi,Epochs,LearningRate,n,fu)
            p.compile()
        except:
            messagebox.showinfo("Error", "Values are missing")
    def Get_pred():
        try:
            X_1=float(X1.get())
            X_2 = float(X2.get())
            X_3 = float(X3.get())
            X_4 = float(X4.get())
            global p
            result=p.insert_sample([X_1,X_2,X_3,X_4])
            messagebox.showinfo("Result", result)
        except:
            messagebox.showinfo("Error", "Generate Model First")
    backprop= Button(app, text="Generate Model", width=15, command=Get_info)
    backprop.place(x=250, y=150)
    get_pred = Button(app, text="Predicate", width=15, command=Get_pred)
    get_pred.place(x=370, y=150)
    app.mainloop()
run()