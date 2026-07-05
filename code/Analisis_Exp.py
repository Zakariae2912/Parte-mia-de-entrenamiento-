from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("Analisis_Exploratorio_Sepsis_Challenge2019") \
    .getOrCreate()

#SOLO avisa si hay error
spark.sparkContext.setLogLevel("ERROR")

# NO leer el dataset original.
ruta_parquet_local = "/home/adminp/TFA/parquet/dataset_preprocesado"
ruta_parquet_spark = "file://" + ruta_parquet_local


print("=" * 70)
print("ANÁLISIS EXPLORATORIO DEL DATASET CHALLENGE 2019")
print("=" * 70)

# CARGA DEL PARQUET PREPROCESADO

dataset = spark.read.parquet(ruta_parquet_spark)

print("\nDataset cargado correctamente desde Parquet.")

# Ver dimensiones del Dataset
#Cuantos filas, columnas, pacientes y variables.

print("\n" + "=" * 70)
print("1. DIMENSIONES GENERALES DEL DATASET")
print("=" * 70)

n_registros = dataset.count()

# Esto evita mezclar pacientes si existiera el mismo patient_id en ambos hospitales.
n_pacientes = dataset.select(
    "hospital",
    "patient_id"
).distinct().count()

n_variables = len(dataset.columns)

print(f"Número total de registros horarios: {n_registros}")
print(f"Número total de pacientes únicos: {n_pacientes}")
print(f"Número total de variables: {n_variables}")

print("\nVariables disponibles en el dataset:")
for variable in dataset.columns:
    print(variable)

#Exploracion de variables clave y ver si falta alguna en el parquet

print("\n" + "=" * 70)
print("2. Exploracion de variables clave")
print("=" * 70)

columnas_clave = [
    "hospital",
    "patient_id",
    "ICULOS",
    "SepsisLabel",
    "Age",
    "Gender",
    "Unit1",
    "Unit2"
]

columnas_ausentes = [
    columna for columna in columnas_clave
    if columna not in dataset.columns
]

if len(columnas_ausentes) == 0:
    print("Todas las columnas clave están presentes.")
else:
    print("Columnas clave ausentes:")
    print(columnas_ausentes)

# Distribucion por hospital

print("\n" + "=" * 70)
print("3. Distribución por hospital")
print("=" * 70)

print("\nRegistros horarios por hospital:")

dataset.groupBy("hospital") \
    .count() \
    .orderBy("hospital") \
    .show()

print("\nPacientes únicos por hospital:")

dataset.select(
    "hospital",
    "patient_id"
).distinct() \
 .groupBy("hospital") \
 .count() \
 .orderBy("hospital") \
 .show()


#CREACIÓN DE DATASET A NIVEL PACIENTE

print("\n" + "=" * 70)
print("4. CREACIÓN DE DATASET A NIVEL PACIENTE")
print("=" * 70)

# Se han juntado todas las horas de cada paciente en una sola fila 

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

print("\nPrimeras filas del dataset a nivel paciente:")
dataset_paciente.show(10, truncate=False)

print("\nNúmero de pacientes en dataset_paciente:")
print(dataset_paciente.count())
