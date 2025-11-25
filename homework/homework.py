import numpy as np

def log1p_transform(x):
    return np.log1p(x)

# Laboratorio 5 - Predicción de precios usando regresión lineal

# Introducción:

#  En este dataset se desea pronosticar el precio de vhiculos usados. El dataset original contiene las siguientes columnas:

# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba en la carpeta "files/input/".

# Análisis de los datos:


import pandas as pd
import os

Data_Folder = "files/input"

Input_Data_Train_Zip = os.path.join(Data_Folder, "train_data.csv.zip")
Input_Data_Test_Zip = os.path.join(Data_Folder, "test_data.csv.zip") 


Data_Train = pd.read_csv(Input_Data_Train_Zip)
Data_Test = pd.read_csv(Input_Data_Test_Zip)

print(Data_Train.head())
print(Data_Test.head())



# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.


Data_Train_Original= Data_Train.copy()
Data_Test_Original= Data_Test.copy()

Current_Year = 2021

Data_Train['Age'] = Current_Year - Data_Train['Year']
Data_Train.drop(columns=['Year', 'Car_Name'], inplace=True)
print(Data_Train.describe(include='all'))


Data_Test['Age'] = Current_Year - Data_Test['Year']
Data_Test.drop(columns=['Year', 'Car_Name'], inplace=True)
print(Data_Test.describe())



# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.


target = 'Present_Price'


X_Train = Data_Train.drop(columns=[target])
y_Train = Data_Train[target]

X_Test = Data_Test.drop(columns=[target])
y_Test = Data_Test[target]


# Análisis previo de los datos

print(X_Train.describe(include='all'))
print(X_Test.describe(include='all'))



# Analisis Correlación

from sklearn.feature_selection import mutual_info_regression


numeric_features = ['Selling_Price', 'Driven_kms', 'Owner', 'Age']
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']

# Detección de correlación X respecto de Y: Calcular MI (Mutual Information) = Posible Data Leakage

# MI para numericas

X_train_num = X_Train[numeric_features]

print("\nColumnas numéricas utilizadas para MI:")
print(X_train_num.columns.tolist())

mi = mutual_info_regression(X_train_num, y_Train, random_state=42)

mi_series = pd.Series(mi, index=numeric_features)
mi_series = mi_series.sort_values(ascending=False)

print("\nMutual Information entre variables numéricas y la Y:")
print(mi_series)

print("\nVariables con MI sospechosamente alta:")
print(mi_series[mi_series > mi_series.quantile(0.90)])

# R/: Se identifica que Selling_Price está altamente relacionada con el Target (Present_Price)
# Se podria tratar de Data LeakAge. Pero!! Si se elimina la variable, no pasa el test.

# Detección de correlación entre las X: Calcular Correlación Pearson

# Pearson para numericas

X_train_num = X_Train[numeric_features]

corr_matrix = X_train_num.corr()

corr_abs = corr_matrix.abs()

high_corr_pairs = [
    (corr_abs.index[i], corr_abs.columns[j], corr_abs.iloc[i, j])
    for i in range(len(corr_abs))
    for j in range(i + 1, len(corr_abs))
    if corr_abs.iloc[i, j] > 0.80
]

print("\n===== PARES CON CORRELACIÓN ALTA (> 0.80) =====")

if high_corr_pairs:
    for var1, var2, corr in high_corr_pairs:
        print(f"{var1}  —  {var2}:  {corr:.3f}")
else:
    print("No se encontraron pares altamente correlacionados.")

# R/: No hay correlación entre las X numercicas.


# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.  

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd


numeric_features = ['Selling_Price', 'Driven_kms', 'Owner', 'Age']
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']


# Detección de outliers usando el método IQR

df = X_Train.copy()

outlier_info = {}

for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    outliers_col = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
    
    outlier_info[col] = {
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "n_outliers": outliers_col.shape[0],
        "outliers": outliers_col[col].values
    }

outlier_info

# Ahora que sé que hay aoutliers, quiero ver cuáles son:

outlier_rows = {}

for col, info in outlier_info.items():
    lower = info["lower_limit"]
    upper = info["upper_limit"]

    
    mask = (df[col] < lower) | (df[col] > upper)
    
    
    outlier_rows[col] = df[mask]


for col, rows in outlier_rows.items():
    print(f"\n=== OUTLIERS EN {col} ===")
    print(f"Total filas: {len(rows)}")
    print(rows)


# R/: Se encuentran outliers en Driven_kms que podrian ser trtados con transformación logarítmica.
# Se pasa el test sin esta modificación.


preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Pipeline final
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(f_regression)),
    ('model', LinearRegression())
])


# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

param_grid = {
    #'selector__k': [1 ,2 ,3 , 4, 5, 6, 7, 8, 9, 10, 11,'all'],
    'selector__k': [11] 
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

grid_search.fit(X_Train, y_Train)

best_model = grid_search.best_estimator_

print("K Variables para usar son:", grid_search.best_params_)



# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

import gzip
import pickle


os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid_search, f)

print("Modelo guardado en files/models/model.pkl.gz")

# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}

import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


y_train_pred = best_model.predict(X_Train)
y_test_pred = best_model.predict(X_Test)

metrics_train = {
    'type': 'metrics','dataset': 'train',
    'r2': r2_score(y_Train, y_train_pred),
    'mse': mean_squared_error(y_Train, y_train_pred),
    'rmse': np.sqrt(mean_squared_error(y_Train, y_train_pred)),
    'mad': -mean_absolute_error(y_Train, y_train_pred)
}

print(metrics_train)

metrics_test = {
    'type': 'metrics',
    'dataset': 'test',
    'r2': r2_score(y_Test, y_test_pred),
    'mse': mean_squared_error(y_Test, y_test_pred),
    'rmse': np.sqrt(mean_squared_error(y_Test, y_test_pred)),
    'mad': -mean_absolute_error(y_Test, y_test_pred)
}

print(metrics_test)


os.makedirs("files/output", exist_ok=True)

output_path = os.path.join("files/output", "metrics.json")
with open(output_path, "w") as f:
    f.write(json.dumps(metrics_train) + "\n")
    f.write(json.dumps(metrics_test) + "\n")

print(f"Métricas guardadas en {output_path}")


