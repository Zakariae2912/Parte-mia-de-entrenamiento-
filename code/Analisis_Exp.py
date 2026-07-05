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
print("Ver dimensiones del Dataset")
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

#Exploracion de variables clave

print("\n" + "=" * 70)
print("Exploracion de variables clave")
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
print("Distribución por hospital")
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
print("CREACIÓN DE DATASET A NIVEL PACIENTE")
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

# Descripcion global de la cohorte

print("\n" + "=" * 70)
print("Descripcion global de la cohorte")
print("=" * 70)

print("\nEdad y duración del seguimiento en UCI:")

dataset_paciente.agg(
    F.count("*").alias("n_pacientes"),
    F.mean("Age").alias("edad_media"),
    F.stddev("Age").alias("edad_desv_std"),
    F.min("Age").alias("edad_min"),
    F.max("Age").alias("edad_max"),
    F.mean("ICULOS_max").alias("iculos_max_media"),
    F.stddev("ICULOS_max").alias("iculos_max_desv_std"),
    F.min("ICULOS_max").alias("iculos_max_min"),
    F.max("ICULOS_max").alias("iculos_max_max")
).show(truncate=False)

print("\nDistribución por sexo:")

dataset_paciente.groupBy("Gender") \
    .count() \
    .orderBy("Gender") \
    .show()

print("\nDistribución por Unit1:")

dataset_paciente.groupBy("Unit1") \
    .count() \
    .orderBy("Unit1") \
    .show()

print("\nDistribución por Unit2:")

dataset_paciente.groupBy("Unit2") \
    .count() \
    .orderBy("Unit2") \
    .show()

#Sépticos y no sépticos: 

print("\n" + "=" * 70)
print("Sépticos y no sépticos: ")
print("=" * 70)

print("\nDistribución global de pacientes con y sin sepsis:")

dataset_paciente.groupBy("paciente_con_sepsis") \
    .count() \
    .orderBy("paciente_con_sepsis") \
    .show()

print("\nDistribución de pacientes con y sin sepsis por hospital:")

dataset_paciente.groupBy(
    "hospital",
    "paciente_con_sepsis"
).count() \
 .orderBy("hospital", "paciente_con_sepsis") \
 .show()

print("\nResumen de la primera hora de aparición de SepsisLabel = 1:")

dataset_paciente.filter(
    F.col("paciente_con_sepsis") == 1
).agg(
    F.count("*").alias("n_pacientes_sepsis"),
    F.mean("primera_hora_sepsis").alias("media_horas"),
    F.stddev("primera_hora_sepsis").alias("desv_std"),
    F.min("primera_hora_sepsis").alias("min_horas"),
    F.max("primera_hora_sepsis").alias("max_horas")
).show(truncate=False)

# 7. Comparacion intercentro

print("\n" + "=" * 70)
print("Comparacion intercentro")
print("=" * 70)

resumen_hospital = dataset_paciente.groupBy("hospital").agg(
    F.count("*").alias("n_pacientes"),
    F.sum("paciente_con_sepsis").alias("n_pacientes_sepsis"),
    F.mean("Age").alias("edad_media"),
    F.stddev("Age").alias("edad_desv_std"),
    F.mean("ICULOS_max").alias("iculos_max_media"),
    F.stddev("ICULOS_max").alias("iculos_max_desv_std")
).withColumn(
    "porcentaje_sepsis",
    F.round(
        100 * F.col("n_pacientes_sepsis") / F.col("n_pacientes"),
        2
    )
)

resumen_hospital.show(truncate=False)

# Sepsis vs No Sepsis

print("\n" + "=" * 70)
print("Sepsis vs No Sepsis")
print("=" * 70)

dataset_paciente.groupBy("paciente_con_sepsis").agg(
    F.count("*").alias("n_pacientes"),
    F.mean("Age").alias("edad_media"),
    F.stddev("Age").alias("edad_desv_std"),
    F.mean("ICULOS_max").alias("iculos_max_media"),
    F.stddev("ICULOS_max").alias("iculos_max_desv_std")
).orderBy("paciente_con_sepsis") \
 .show(truncate=False)

# 
# Valores perdidos globales.
# Excluye hospital y patient_id

print("\n" + "=" * 70)
print("Valores perdidos globales")
print("=" * 70)

variables_para_missing = [
    columna for columna in dataset.columns
    if columna not in ["hospital", "patient_id"]
]

filas_missing = []

for variable in variables_para_missing:

    n_nulos = dataset.filter(
        F.col(variable).isNull()
    ).count()

    porcentaje = 100 * n_nulos / n_registros

    filas_missing.append(
        (
            variable,
            n_nulos,
            round(porcentaje, 2)
        )
    )

tabla_missing = spark.createDataFrame(
    filas_missing,
    ["variable", "n_nulos", "porcentaje_nulos"]
)

print("\nVariables con mayor porcentaje de valores perdidos:")

tabla_missing.orderBy(
    F.desc("porcentaje_nulos")
).show(
    50,
    truncate=False
)

print("\nVariables con más del 95 % de valores perdidos:")

tabla_missing.filter(
    F.col("porcentaje_nulos") > 95
).orderBy(
    F.desc("porcentaje_nulos")
).show(
    truncate=False
)


# Valores perdidos por hospital

print("\n" + "=" * 70)
print("valores perdidos por hospital")
print("=" * 70)

variables_missing_hospital = [
    "HR",
    "O2Sat",
    "SaO2",
    "O2Sat_combined",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "FiO2",
    "Lactate",
    "Calcium",
    "Creatinine",
    "WBC",
    "Platelets",
    "Hgb"
]

variables_missing_hospital = [
    variable for variable in variables_missing_hospital
    if variable in dataset.columns
]

for variable in variables_missing_hospital:

    print(f"\nValores perdidos por hospital para {variable}:")

    dataset.groupBy("hospital").agg(
        F.count("*").alias("n_registros"),
        F.sum(
            F.when(
                F.col(variable).isNull(),
                1
            ).otherwise(0)
        ).alias("n_nulos")
    ).withColumn(
        "porcentaje_nulos",
        F.round(
            100 * F.col("n_nulos") / F.col("n_registros"),
            2
        )
    ).orderBy("hospital") \
     .show(truncate=False)

# Variables depuradas

print("\n" + "=" * 70)
print("variables depuradas")
print("=" * 70)

variables_depuradas = [
    "HR",
    "Temp",
    "SBP",
    "Resp",
    "FiO2",
    "BaseExcess",
    "HCO3",
    "Chloride",
    "Potassium",
    "Hgb"
]

filas_depuradas = []

for variable in variables_depuradas:

    columna_limpia = variable + "_limpia"
    columna_flag = variable + "_fuera_rango"

    if columna_limpia in dataset.columns and columna_flag in dataset.columns:

        n_original = dataset.filter(
            F.col(variable).isNotNull()
        ).count()

        n_limpia = dataset.filter(
            F.col(columna_limpia).isNotNull()
        ).count()

        n_fuera_rango = dataset.filter(
            F.col(columna_flag) == 1
        ).count()

        filas_depuradas.append(
            (
                variable,
                n_original,
                n_limpia,
                n_fuera_rango,
                round(100 * n_fuera_rango / n_registros, 4)
            )
        )

if len(filas_depuradas) > 0:

    tabla_depuradas = spark.createDataFrame(
        filas_depuradas,
        [
            "variable",
            "n_original_no_nulo",
            "n_limpia_no_nulo",
            "n_fuera_rango",
            "porcentaje_fuera_rango"
        ]
    )

    tabla_depuradas.orderBy(
        F.desc("porcentaje_fuera_rango")
    ).show(
        truncate=False
    )

else:
    print("No se encontraron variables depuradas en el dataset.")
#Ya se generaron variables limpias y flagos en el preprocesado fuera de rango pero bueno bueno esto ultimo lo dejamos
#Como medida de seguridad extra

