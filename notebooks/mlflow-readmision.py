# Import MLflow, keras and tensorflow
import mlflow
import mlflow.keras
import keras
import tensorflow as tf
import tensorflow.keras as tk
from keras import models
from keras import layers
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Entrenamiento de una red feed-forward para el problema de clasificación con datos de readmisión de pacientes diabéticos')
parser.add_argument('--batch_size', '-b', type=int, default=128)
parser.add_argument('--epochs', '-e', type=int, default=20)
parser.add_argument('--learning_rate', '-l', type=float, default=0.05)
parser.add_argument('--num_hidden_units', '-n', type=int, default=64)
parser.add_argument('--num_hidden_layers', '-N', type=int, default=1)
parser.add_argument('--dropout', '-d', type=float, default=0.25)
parser.add_argument('--momentum', '-m', type=float, default=0.85)


args = parser.parse_args([])

# Usaremos esta función para definir Descenso de Gradiente Estocástico como optimizador
def get_optimizer():
    """
    :return: Keras optimizer
    """
    optimizer = keras.optimizers.SGD(learning_rate=args.learning_rate,momentum=args.momentum, nesterov=True)
    return optimizer


# Obtenemos el dataset y aislamos la variable objetivo
df = pd.read_csv('diabetic_data_processed.csv').drop('Unnamed: 0', axis=1)
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Reescalar columnas numéricas (se usa RobustScaler debido a la alta presencia de outliers mostrada en el análisis exploratorio)
numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = RobustScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Se obtienen las variables dummy a partir de las categoricas (eliminando la primera categoría en cada caso para evitar colinealidad)
X = pd.get_dummies(X, columns=[e for e in X.columns if X[e].dtype == object], drop_first=True)

X = np.asarray(X).astype('float32')

# Dividir en set de entrenamiento y de prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# Esta función define una corrida del modelo, con entrenamiento y 
# registro en MLflow
def run_mlflow(run_name="MLflow CE Readmision"):
    # Iniciamos una corrida de MLflow
    mlflow.start_run(run_name=run_name)
    run = mlflow.active_run()
    # MLflow asigna un ID al experimento y a la corrida
    experimentID = run.info.experiment_id
    runID = run.info.run_uuid
    # reistro automáticos de las métricas de keras
    mlflow.keras.autolog()
    model = models.Sequential()
    model.add(layers.Dense(x_train.shape[1], activation=tf.nn.relu, input_shape=(x_train.shape[1],)))
    # Agregamos capas ocultas a la red
    # en los argumentos: --num_hidden_layers o -N 
    for n in range(0, args.num_hidden_layers):
        # agregamos una capa densa (completamente conectada) con función de activación relu
        model.add(layers.Dense(args.num_hidden_units, activation=tf.nn.relu))
        # agregamos dropout como método de regularización para aleatoriamente descartar una capa
        # si los gradientes son muy pequeños
        model.add(layers.Dropout(args.dropout))
        # capa final con 10 nodos de salida y activación softmax 
        model.add(layers.Dense(3, activation=tf.nn.softmax))
        # Use Scholastic Gradient Descent (SGD) or Adadelta
        # https://keras.io/optimizers/
        optimizer = get_optimizer()

    # compilamos el modelo y definimos la función de pérdida  
    # otras funciones de pérdida comunes para problemas de clasificación
    # 1. sparse_categorical_crossentropy
    # 2. binary_crossentropy
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['categorical_accuracy'])

    # entrenamos el modelo
    print("-" * 100)
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    # evaluamos el modelo
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    mlflow.end_run(status='FINISHED')
    return (experimentID, runID)


# corrida con parámetros diferentes a los por defecto
args = parser.parse_args(["--batch_size", '128', '--epochs', '20'])
(experimentID, runID) = run_mlflow()
print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
print(tf.__version__)
print("-" * 100)