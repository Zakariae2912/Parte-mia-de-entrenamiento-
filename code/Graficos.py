import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import plotly.express as px


spark = SparkSession.builder \
    .appName("Graficos_Sepsis_Challenge2019") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

#
ruta_parquet_local = "CAMBIAR RUTA"
ruta_parquet_spark = "file://" + ruta_parquet_local

ruta_graficos_local = "/home/adminp/TFA/graficos_eda"

os.makedirs(
    ruta_graficos_local,
    exist_ok=True
)


print("=" * 70)
print("GENERACIÓN DE GRÁFICOS DEL DATASET CHALLENGE 2019")
print("=" * 70)

# Carga de parquet del preprocesado

dataset = spark.read.parquet(ruta_parquet_spark)

print("\nDataset cargado correctamente desde Parquet.")

print("\nEsquema del dataset:")
dataset.printSchema()

print("\nPrimeras filas del dataset:")
dataset.show(5, truncate=False)

n_registros = dataset.count()
n_variables = len(dataset.columns)

print("\nNúmero total de registros horarios:", n_registros)
print("Número total de variables:", n_variables)

# Creacion de version resumida del Dataset

dataset_paciente = dataset.groupBy(
    "hospital",
    "patient_id"
).agg(
    F.max("Age").alias("Age"),
    F.max("Gender").alias("Gender"),
    F.max("Unit1").alias("Unit1"),
    F.max("Unit2").alias("Unit2"),
    F.min("ICULOS").alias("ICULOS_min"),
    F.max("ICULOS").alias("ICULOS_max"),
    F.count("ICULOS").alias("n_registros_paciente"),
    F.max("SepsisLabel").alias("paciente_con_sepsis"),
    F.min(
        F.when(
            F.col("SepsisLabel") == 1,
            F.col("ICULOS")
        )
    ).alias("primera_hora_sepsis")
)

n_pacientes = dataset_paciente.count()

print("\nDataset a nivel paciente creado correctamente.")
print("Número total de pacientes:", n_pacientes)

print("\nPrimeras filas del dataset a nivel paciente:")
dataset_paciente.show(5, truncate=False)

# ==========================================================
# FUNCIÓN AUXILIAR PARA GUARDAR GRÁFICOS
# ==========================================================

def guardar_grafico_plotly(figura, nombre_archivo):
  

    ruta_salida = os.path.join(
        ruta_graficos_local,
        nombre_archivo
    )

    figura.write_html(
        ruta_salida
    )

    print("Gráfico guardado:", ruta_salida)


def spark_a_pandas(tabla_spark):
  
    return tabla_spark.toPandas()
