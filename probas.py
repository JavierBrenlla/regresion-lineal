import numpy as np #Librería numérica
import matplotlib.pyplot as plt # Para crear gráficos con matplotlib
from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn
import datetime #Para trabajar con fechas

# fechas = [5102022, 12042021, 20092021, 4042022, 3102022, 10042023]
fechas = [datetime.datetime(2021,10,5).timestamp(), datetime.datetime(2021,4,12).timestamp(), datetime.datetime(2021,9,20).timestamp(), datetime.datetime(2022,4,4).timestamp(), datetime.datetime(2022,3,10).timestamp(), datetime.datetime(2023,4,10).timestamp(), datetime.datetime(2023,10,9).timestamp()]
datos = [42,18,34,17,26,37,6]

# prediccion = [9102023]
prediccion = [datetime.datetime(2024,4,10).timestamp()]

regresion_lineal = LinearRegression()

regresion_lineal.fit(np.array(fechas).reshape(-1,1), datos)

prediccion = regresion_lineal.predict(np.array(prediccion).reshape(-1,1))

print(prediccion)