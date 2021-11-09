import pickle
import numpy as np
from tkinter import *
import json
global __locations
global __data_columns
global __model

#function for prediction

def get_estimated_price(location, sqft, bhk, bath):
    print(location, sqft, bhk, bath)
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    price=round(__model.predict([x])[0], 2)
    label4 = Label(window, text=str(price)+" Lakhs", relief="solid",
                   width=24, font=("arial", 14, "bold"))
    label4.place(x=140, y=340)
    return price
def get_location_names():
    return __locations

def load_saved_artifacts():
    global __locations
    global __data_columns
    with open("./columns.json", 'r')as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    global __model
    with open("./banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)

load_saved_artifacts()
#tkinter
window = Tk()
window.geometry("610x400")
window.title("Price Prediction")

fn = DoubleVar(window)
var = StringVar(window)
var1 = IntVar(window)
var2 = IntVar(window)

def exit1():
    exit()

label_h = Label(window, text="HOUSE PRICE PREDICTION", relief="solid", width=28, font=("arial", 19, "bold"))
label_h.place(x=90, y=20)

label_l = Label(window, text="Location", width=30, font=("arial", 10, "bold"))
label_l.place(x=100, y=100)
droplist = OptionMenu(window, var, *__locations)
var.set("Select Location")
droplist.config(width=15)
droplist.place(x=300, y=100)

label_a = Label(window, text="Area(Total Square Feet)", width=30, font=("arial", 10, "bold"))
label_a.place(x=100, y=140)
entry_a = Entry(window, textvar=fn)
entry_a.place(x=305, y=140)

label_s = Label(window, text="Size", width=30, font=("arial", 10, "bold"))
label_s.place(x=100, y=180)
list2 = ['1', '2', '3', '4', '5', '6']
droplist = OptionMenu(window, var1, *list2)
var1.set("Select BHK")
droplist.config(width=15)
droplist.place(x=300, y=180)

label_b = Label(window, text="Bathroom", width=30, font=("arial", 10, "bold"))
label_b.place(x=100, y=220)
list3 = ['1', '2', '3', '4', '5', '6']
droplist = OptionMenu(window, var2, *list3)
var2.set("Select Bathroom")
droplist.config(width=15)
droplist.place(x=300, y=220)

# def second_win():
#     window2 = Tk()
#     window2.geometry('350x350')
#     window2.title("Price")
#     label = Label(window2, text="Predicted Price", font=("arial", '12', 'bold'))
#     label.place(x=100, y=30)
#     price = get_estimated_price(var.get(), float(fn.get()), var1.get(), var2.get())
#
#     label_ = Label(window2, text=str(price)+" Lakhs", font=("arial", '12', 'bold'))
#     label_.place(x=100, y=150)
#     b_2 = Button(window2, text="Exit", width=12, bg='brown', fg='white', command=exit1)
#     b_2.place(x=125, y=300)
#
b_p = Button(window, text="PREDICT", width=50, bg='brown', fg='white', command=lambda:get_estimated_price(var.get(), float(fn.get()), var1.get(), var2.get()))
b_p.place(x=100, y=280)





window.mainloop()