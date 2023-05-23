import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import csv

#data collection
df = pd.read_csv("E:\Python\Dataset\CPI_2021.csv")
arr=df.to_numpy()
existing = arr

#CPI historic graph
def display_historic_graph():
    df = pd.read_csv("E:\Python\Dataset\CPI_2021.csv")
    df = df.set_index('year') 
    df.plot(figsize=(15, 5),
            title='CPI')
    plt.show()

#historic inflation bar graph
def display_historic_bar_graph():
    df = pd.read_csv('E:\Python\Dataset\CPI_2021.csv')
    arr = df.to_numpy()
    rows, cols = arr.shape
    N_arr = np.zeros((rows, cols+1))  
    N_arr[:, :-1] = arr  

    for i in range(rows):
        CPIT = N_arr[i, 1]
        CPIP = N_arr[i-1, 1] if i > 0 else 0  
        if CPIT == 0 or CPIP == 0:
            Inf = 0
        else:
            Inf = ((CPIT-CPIP)/CPIP)*100
        N_arr[i, -1] = Inf
    
    x = N_arr[:, 0]  
    y = N_arr[:, 2]  

    plt.bar(x, y) 
    plt.xlabel('Year')
    plt.ylabel('Inflation Rate (%)')
    plt.title('Inflation Rates Over Time')
    plt.show()

#testing and training graph
train = df.loc[df.index < 2010] 
test = df.loc[df.index >= 2010]

years = existing[:, 0]
historic_values = existing[:, 1]
model = LinearRegression()
model.fit(years.reshape(-1, 1), historic_values)

future_years = np.arange(years[-1]+1, years[-1]+20+1)
future_values = model.predict(future_years.reshape(-1, 1))
future= np.column_stack((future_years, future_values))
combined = np.concatenate((existing, future))
rows=len(combined)
rows, cols = combined.shape
arr = np.zeros((rows, cols+1))  # create a new array with an additional column for the calculated values
arr[:, :-1] = combined  # copy the contents of 'combined' to the new array, except for the last column
   
for i in range(rows):
    CPIT = combined[i][1]
    CPIP = combined[i-1][1] if i > 0 else 0  # handle the case when i=0 (i.e., first row)
    Inf = ((CPIT-CPIP)/CPIP)*100
    arr[i, -1] = Inf  # assign the calculated value to the last column of the ith row of 'arr'

r_squared = model.score(years.reshape(-1, 1), historic_values) 
arr[58,2]=5
print("R-squared value:", r_squared)

def predict_inflation(year):
    rows=len(combined)
    columns=len(combined[0])
    for i in range(rows):
        if combined[i][0]==year:
            CPI_This=combined[i][1]
            CPI_Prev=combined[i-1][1]
            break

    Inflation = ((CPI_This-CPI_Prev)/CPI_Prev)*100
    return CPI_This, CPI_Prev, Inflation

#CPI graph up to the entered year
def display_cpi_graph(year):
    filtered_data = combined[combined[:, 0] <= year]
    years = filtered_data[:, 0]
    cpi = filtered_data[:, 1]
    plt.plot(years, cpi)
    plt.xlabel("Year")
    plt.ylabel("CPI")
    plt.title("Consumer Price Index")
    plt.show()

#Inflaion Bar graph up to the entered year
def display_inflation_graph(year):
    # filter the array to get the inflation rates from 20 years before the user-entered year up to the user-entered year
    start_year = year - 20
    infl_values = arr[(arr[:, 0] >= start_year) & (arr[:, 0] <= year), -1]

    # create a bar chart of the inflation rates
    plt.bar(range(len(infl_values)), infl_values)
    plt.xticks(range(len(infl_values)), [f'{int(year-20+i)}' for i in range(len(infl_values))])
    plt.xlabel('Year')
    plt.ylabel('Inflation Rate (%)')
    plt.title(f'Inflation Rates from {year-20} to {year}')
    plt.show()


# GUI
sg.theme('DefaultNoMoreNagging')
layout = [[sg.Text("Enter Year")],
          [sg.Input(size=(35,15), key='_YEAR_')],   
          [sg.Button("Calculate", bind_return_key=True)],
          [sg.Output(size=(33,5), key='_OUTPUT_')],  
        #   [sg.Text('', key='_ANSWER_', size=(60,5))],
          [sg.Text("Accuracy", key='_R_SQUARED_')],
          [sg.Button("CPI Historic Graph")],
          [sg.Button("CPI Graph Upto Predicted Year")],
          [sg.Button("Inflation Historic Bar Graph")],
          [sg.Button("Inflation Bar Graph Upto Predicted Year")],
          [sg.Button("Exit")]]

window = sg.Window('Inflation Prediction System', layout, size=(400,400))

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:   #exit ma or close ma click garyo bhane banda garne
        break
    elif event == "CPI Historic Graph":
        display_historic_graph()
    elif event == "CPI Graph Upto Predicted Year":
        try:
            year = int(values['_YEAR_'])
            display_cpi_graph(year)
        except:
            print("Please enter a valid year.")
    elif event == "Inflation Historic Bar Graph":
        display_historic_bar_graph()
    elif event == "Inflation Bar Graph Upto Predicted Year":
        try:
            year = int(values['_YEAR_'])
            display_inflation_graph(year)
        except:
            print("Please enter a valid year.")
    elif event == "Calculate":
        try:
            year = int(values['_YEAR_'])     #user le year enter garepachi yo portion gets executed
            CPI_This, CPI_Prev, inflation = predict_inflation(year)
            print(f"CPI of the year {year} = {CPI_This:.2f}")
            print(f"CPI of the year {year-1} = {CPI_Prev:.2f}")
            print(f"Inflation Rate of the year {year} = {inflation:.2f}%")
            window['_R_SQUARED_'].update(f"Accuracy: {r_squared: .2f}")

        except:
            print("Please enter a valid year.")    #range bahira ko year enter garda this message is shown  
window.close()