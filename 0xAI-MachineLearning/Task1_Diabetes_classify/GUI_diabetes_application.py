import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import joblib
from pathlib import Path




def predict():
    arr = np.array([])
    for name, widget in widgets.items():
        if name == "smoking_history":
            input = smoking_map[widget.get()]
        else:
            input = widget.get()

        if not input:
            warning_label.config(text="Please Enter all fields!")
            return
        else:
            try:
                warning_label.config(text="")
                arr = np.append(arr, [float(input)])
            except ValueError:
                warning_label.config(text="Please Enter a numeric value!")
                return
    
    input_arr = pd.DataFrame([arr], columns=features)
    
    input_arr = scaler.transform(input_arr)
    prediction = model.predict(input_arr)

    if prediction[0] == 1:
        result.config(text="Diabatic", foreground="red")
    elif prediction[0] == 0:
        result.config(text="Not Diabatic", foreground="green")

    
# makes sure the file paths are loadded correctly even if the file is run from a diffrnt location
script_dir = Path(__file__).parent # gets the path of the parent directory
model_path = script_dir / 'diabetes_model.joblib'
scaler_path = script_dir / 'scaler.joblib'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


features = ["age","hypertension","heart_disease","smoking_history","bmi","HbA1c_level","blood_glucose_level"]
smoking_map = {'never': '0', 'No Info': '0', 'current': '2', 'former': '1', 'ever': '2', 'not current': '1'}
widgets = {}

wn = tk.Tk()
wn.title("Diabetes Prediction")

for i, feature in enumerate(features):
    wn.grid_columnconfigure(i, weight=1) # centers the grid along the column
    label = ttk.Label(wn, text=feature)
    label.grid(row=0, column=i,padx=5, pady=5)

    if feature == "smoking_history":
        dropdown = ttk.Combobox(wn, values=['never', 'No Info', 'current', 'former', 'ever', 'not current'], state='readonly')
        dropdown.grid(row=1, column=i,padx=5, pady=15)
        widgets[feature] = dropdown
    else:
        entry = ttk.Entry(wn)
        entry.grid(row=1, column=i,padx=5, pady=15)
        widgets[feature] = entry

    


btn = ttk.Button(wn, text="Predict", width=30,  command=predict).grid(row=2, column=len(features)//2)


result = ttk.Label(wn, text="Result will be shown here")
result.grid(row=3, column=len(features)//2, pady=15)

warning_label = ttk.Label(wn, text="", foreground="red")
warning_label.grid(row=4,column=len(features)//2, pady=15)
    



wn.mainloop()