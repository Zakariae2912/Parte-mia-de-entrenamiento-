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


# Guardar graficos

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


# Pacientes y registros por hospital

print("\n" + "=" * 70)
print("Pacientes y registros por hospital")
print("=" * 70)



tabla_registros_hospital = dataset.groupBy(
    "hospital"
).agg(
    F.count("*").alias("n_registros")
).orderBy(
    "hospital"
)

fig_registros_hospital = px.bar(
    spark_a_pandas(tabla_registros_hospital),
    x="hospital",
    y="n_registros",
    title="Registros horarios por hospital",
    labels={
        "hospital": "Hospital",
        "n_registros": "Número de registros horarios"
    }
)

guardar_grafico_plotly(
    fig_registros_hospital,
    "01_registros_por_hospital.html"
)


tabla_pacientes_hospital = dataset_paciente.groupBy(
    "hospital"
).agg(
    F.count("*").alias("n_pacientes")
).orderBy(
    "hospital"
)

fig_pacientes_hospital = px.bar(
    spark_a_pandas(tabla_pacientes_hospital),
    x="hospital",
    y="n_pacientes",
    title="Pacientes únicos por hospital",
    labels={
        "hospital": "Hospital",
        "n_pacientes": "Número de pacientes"
    }
)

guardar_grafico_plotly(
    fig_pacientes_hospital,
    "02_pacientes_por_hospital.html"
)

# ==========================================================
# 6. SEPSIS GLOBAL Y POR HOSPITAL
# ==========================================================

print("\n" + "=" * 70)
print("6. SEPSIS GLOBAL Y POR HOSPITAL")
print("=" * 70)


# Septicos vs No Septicos.

tabla_sepsis_global = dataset_paciente.groupBy(
    "paciente_con_sepsis"
).agg(
    F.count("*").alias("n_pacientes")
).withColumn(
    "grupo",
    F.when(
        F.col("paciente_con_sepsis") == 1,
        F.lit("Con sepsis")
    ).otherwise(
        F.lit("Sin sepsis")
    )
).withColumn(
    "porcentaje",
    F.round(
        100 * F.col("n_pacientes") / F.lit(n_pacientes),
        2
    )
).orderBy(
    "paciente_con_sepsis"
)

fig_sepsis_global = px.bar(
    spark_a_pandas(tabla_sepsis_global),
    x="grupo",
    y="n_pacientes",
    title="Pacientes con y sin sepsis",
    labels={
        "grupo": "Grupo",
        "n_pacientes": "Número de pacientes"
    },
    hover_data=["porcentaje"]
)

guardar_grafico_plotly(
    fig_sepsis_global,
    "03_pacientes_con_sin_sepsis.html"
)


tabla_sepsis_hospital = dataset_paciente.groupBy(
    "hospital"
).agg(
    F.count("*").alias("n_pacientes"),
    F.sum("paciente_con_sepsis").alias("n_pacientes_sepsis")
).withColumn(
    "porcentaje_sepsis",
    F.round(
        100 * F.col("n_pacientes_sepsis") / F.col("n_pacientes"),
        2
    )
).orderBy(
    "hospital"
)

fig_sepsis_hospital = px.bar(
    spark_a_pandas(tabla_sepsis_hospital),
    x="hospital",
    y="porcentaje_sepsis",
    title="Porcentaje de pacientes con sepsis por hospital",
    labels={
        "hospital": "Hospital",
        "porcentaje_sepsis": "Pacientes con sepsis (%)"
    },
    hover_data=[
        "n_pacientes",
        "n_pacientes_sepsis"
    ]
)

guardar_grafico_plotly(
    fig_sepsis_hospital,
    "04_porcentaje_sepsis_por_hospital.html"
)


# Genero, Edad Y Duracion UCI

print("\n" + "=" * 70)
print("7. Genero, Edad Y Duracion en UCI")
print("=" * 70)


# ----------------------------------------------------------
# Distribución por sexo
# ----------------------------------------------------------

tabla_genero = dataset_paciente.groupBy(
    "Gender"
).agg(
    F.count("*").alias("n_pacientes")
).withColumn(
    "porcentaje",
    F.round(
        100 * F.col("n_pacientes") / F.lit(n_pacientes),
        2
    )
).orderBy(
    "Gender"
)

fig_genero = px.bar(
    spark_a_pandas(tabla_genero),
    x="Gender",
    y="n_pacientes",
    title="Distribución de pacientes por genero",
    labels={
        "Gender": "Gender",
        "n_pacientes": "Número de pacientes"
    },
    hover_data=["porcentaje"]
)

guardar_grafico_plotly(
    fig_genero,
    "05_distribucion_genero.html"
)


edad_pandas = dataset_paciente.select(
    "Age"
).filter(
    F.col("Age").isNotNull()
).toPandas()

fig_edad = px.histogram(
    edad_pandas,
    x="Age",
    nbins=40,
    title="Distribución de edad",
    labels={
        "Age": "Edad"
    }
)

guardar_grafico_plotly(
    fig_edad,
    "06_distribucion_edad.html"
)

iculos_pandas = dataset_paciente.select(
    "ICULOS_max"
).filter(
    F.col("ICULOS_max").isNotNull()
).toPandas()

fig_iculos = px.histogram(
    iculos_pandas,
    x="ICULOS_max",
    nbins=50,
    title="Distribución de la duración máxima del seguimiento en UCI",
    labels={
        "ICULOS_max": "ICULOS máximo"
    }
)

guardar_grafico_plotly(
    fig_iculos,
    "07_distribucion_iculos_max.html"
)
