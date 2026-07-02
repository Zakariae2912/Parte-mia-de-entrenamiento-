from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, input_file_name, regexp_extract, lit, isnan, when, avg, stddev, min, max, count, lag, first, countDistinct, min as  spark_min, max as spark_max, avg, coalesce, round
from pyspark.storagelevel import StorageLevel
import os


# CREACIÓN DE LA SESIÓN SPARK
spark = SparkSession.builder \
    .appName("Preprocesado_Sepsis_Challenge2019") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Muestra únicamente los mensajes de error de Spark
spark.sparkContext.setLogLevel("ERROR")


# ==========================================================
# DEFINICIÓN DE LAS RUTAS
# ==========================================================

# Defininicion de la carpeta principal donde se encuentra el dataset
# Esta ruta deberá adaptarse si cambia entre máquinas
ruta_base = "/home/adminp/physionet.org/files/challenge-2019/1.0.0/training"

# Define la ruta de todos los archivos PSV del conjunto A
ruta_setA = ruta_base + "/training_setA/*.psv"

# Define la ruta de todos los archivos PSV del conjunto B
ruta_setB = ruta_base + "/training_setB/*.psv"

# INICIO DEL PREPROCESADO
# Muestra un encabezado para identificar el inicio del proceso
print("=" * 70)
print("PREPROCESADO DEL DATASET CHALLENGE 2019")
print("=" * 70)

# LECTURA DEL CONJUNTO A y B 
# Lee todos los archivos PSV del conjunto A
dfA = spark.read \
    .option("header", True) \
    .option("delimiter", "|") \
    .option("nullValue", "NaN") \
    .option("nanValue", "NaN") \
    .csv("file://" + ruta_setA)

# Lee todos los archivos PSV del conjunto B
dfB = spark.read \
    .option("header", True) \
    .option("delimiter", "|") \
    .option("nullValue", "NaN") \
    .option("nanValue", "NaN") \
    .csv("file://" + ruta_setB)

# IDENTIFICACIÓN DEL PACIENTE Y DEL HOSPITAL

# CAMBIO: patient_id se obtiene antes de unir los datasets para
# conservar correctamente la identificación de cada paciente.
dfA = dfA.withColumn(
    "patient_id",
    regexp_extract(input_file_name(), r"(p[0-9]+)\.psv", 1)
)

# CAMBIO: se añade el hospital de origen para poder realizar
# posteriormente la comparación y validación intercentro.
dfA = dfA.withColumn(
    "hospital",
    lit("A")
)

# Extrae el identificador del paciente en el conjunto B.
dfB = dfB.withColumn(
    "patient_id",
    regexp_extract(input_file_name(), r"(p[0-9]+)\.psv", 1)
)

# Añade el hospital de procedencia al conjunto B.
dfB = dfB.withColumn(
    "hospital",
    lit("B")
)

# UNIÓN DE LOS CONJUNTOS A Y B
# CAMBIO: la unión se realiza después de añadir hospital y patient_id,
# evitando perder la procedencia de los registros.
dataset = dfA.unionByName(dfB)


# ==========================================================
# CONVERSIÓN DE VARIABLES NUMÉRICAS
# ==========================================================

# Variables que deben convertirse a formato numérico.
variables_numericas = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
    "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
    "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS", "SepsisLabel"
]

# CAMBIO: se añaden Gender, Unit1 y Unit2 porque también contienen
# valores numéricos y deben tener un tipo homogéneo.
for variable in variables_numericas:
    dataset = dataset.withColumn(
        variable,
        col(variable).cast("double")
    )

print("\nDataset leído, unido y convertido correctamente.")


# ==========================================================
# VALIDACIÓN INICIAL DEL DATASET
# ==========================================================

# Muestra el número de pacientes de cada hospital.
print("\nNúmero de pacientes por hospital:")

dataset.select(
    "hospital",
    "patient_id"
).distinct() \
 .groupBy("hospital") \
 .count() \
 .orderBy("hospital") \
 .show()

# CAMBIO: se utiliza hospital + patient_id como identificador,
# evitando mezclar pacientes pertenecientes a centros diferentes.
numero_pacientes = dataset.select(
    "hospital",
    "patient_id"
).distinct().count()

print("Número total de pacientes:", numero_pacientes)

# Se calcula una sola vez para evitar repetir la misma operación Spark.
total_registros = dataset.count()

print("\nNúmero total de registros horarios:")
print(total_registros)

# CAMBIO: se aclara que esta distribución corresponde a registros
# horarios y no al número de pacientes con sepsis.
print("\nDistribución horaria de SepsisLabel:")

dataset.groupBy(
    "SepsisLabel"
).count() \
 .orderBy("SepsisLabel") \
 .show()

# Se incluye hospital para comprobar visualmente la procedencia.
print("\nPrimeras filas:")

dataset.select(
    "hospital",
    "patient_id",
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "Resp",
    "Age",
    "HospAdmTime",
    "ICULOS",
    "SepsisLabel"
).show(
    10,
    truncate=False
)

print("\nEsquema del dataset:")
dataset.printSchema()


# ==========================================================
# ANÁLISIS DE VALORES PERDIDOS
# ==========================================================

print("\n" + "=" * 70)
print("PORCENTAJE DE VALORES PERDIDOS")
print("=" * 70)

# CAMBIO: se mantiene este análisis porque permite evaluar
# la calidad de las variables antes de aplicar el preprocesado.
for variable in variables_numericas:

    # Cuenta tanto los valores nulos como los valores NaN.
    nulos = dataset.filter(
        col(variable).isNull()
        | isnan(col(variable))
    ).count()

    # Calcula el porcentaje de valores ausentes.
    porcentaje = (nulos / total_registros) * 100

    print(f"{variable:20s}: {porcentaje:6.2f}%")


# CAMBIO: LA ESTADISTICA DESC
# 04_analisis_exploratorio.py, porque no transforma los datos.

# ==========================================================
# COMPROBACIÓN ÚNICA DE RANGOS FISIOLÓGICAMENTE PLAUSIBLES
# ==========================================================

# CAMBIO: se elimina la validación repetida porque el número de
# pacientes, registros y el esquema ya se comprobaron anteriormente.

print("\n" + "=" * 70)
print("COMPROBACIÓN DE RANGOS FISIOLÓGICAMENTE PLAUSIBLES")
print("=" * 70)

# CAMBIO: se establece un único conjunto de rangos para evitar
# utilizar límites diferentes en varias partes del preprocesado.
rangos_clinicos = {
    "HR": (20, 250),
    "O2Sat": (0, 100),
    "Temp": (25, 45),
    "SBP": (40, 260),
    "MAP": (30, 200),
    "Resp": (4, 70),
    "FiO2": (0.20, 1.00),
    "Lactate": (0.1, 30)
}

# Recorre las variables incluidas en el control de plausibilidad.
for variable, (limite_inferior, limite_superior) in rangos_clinicos.items():

    # Identifica los valores observados que quedan fuera del rango.
    # Los valores nulos y NaN no se clasifican como fuera de rango.
    condicion_fuera_rango = (
        col(variable).isNotNull()
        & (~isnan(col(variable)))
        & (
            (col(variable) < limite_inferior)
            | (col(variable) > limite_superior)
        )
    )

    # CAMBIO: crea un indicador de calidad sin borrar la variable original.
    dataset = dataset.withColumn(
        variable + "_fuera_rango",
        when(condicion_fuera_rango, 1).otherwise(0)
    )

    # CAMBIO: crea una versión depurada; los valores no plausibles
    # se convierten en ausentes, pero se conserva la columna original.
    dataset = dataset.withColumn(
        variable + "_limpia",
        when(
            (
                col(variable).isNotNull()
                & (~isnan(col(variable)))
                & (col(variable) >= limite_inferior)
                & (col(variable) <= limite_superior)
            ),
            col(variable)
        ).otherwise(None)
    )

    # Cuenta los registros marcados para revisar la magnitud del problema.
    fuera_rango = dataset.filter(
        col(variable + "_fuera_rango") == 1
    ).count()

    porcentaje = (
        fuera_rango / total_registros
    ) * 100

    print(
        f"{variable:10s}: "
        f"{fuera_rango:8d} registros fuera de rango "
        f"({porcentaje:6.4f}%)"
    )

print("\nLas variables originales se conservan para garantizar la trazabilidad.")
print("Las versiones terminadas en '_limpia' se utilizarán en fases posteriores.")

#Variables temporales creo que hay que mirarlo despues 
# COMPROBACIÓN DE VALORES NO PLAUSIBLES tambien creo que hay que eliminarlo porque creo que estaba dupblicado.
# ==========================================================
# CONTROLES FINALES DE CALIDAD
# ==========================================================

print("\n" + "=" * 70)
print("CONTROLES FINALES DE CALIDAD DEL DATASET")
print("=" * 70)

# CAMBIO: se eliminan el análisis de sepsis, la combinación O2Sat/SaO2
# y las tendencias de 24 horas porque pertenecen al análisis o modelado.


# ==========================================================
# VARIABLES CON ELEVADA PROPORCIÓN DE VALORES PERDIDOS
# ==========================================================

# Identifica variables completamente vacías o con más del 95 % de ausentes.
variables_95 = []
variables_100 = []

for variable in variables_numericas:

    # Cuenta conjuntamente los valores nulos y NaN.
    valores_perdidos = dataset.filter(
        col(variable).isNull()
        | isnan(col(variable))
    ).count()

    porcentaje_perdidos = (
        valores_perdidos / total_registros
    ) * 100

    if porcentaje_perdidos == 100:
        variables_100.append(variable)

    if porcentaje_perdidos > 95:
        variables_95.append(variable)

print("\nVariables completamente vacías:")
print(variables_100)

print("\nVariables con más del 95 % de valores perdidos:")
print(variables_95)


# ==========================================================
# COMPROBACIÓN DE DUPLICADOS
# ==========================================================

# CAMBIO: se utiliza hospital + patient_id + ICULOS para evitar
# considerar iguales registros pertenecientes a centros diferentes.
duplicados = dataset.groupBy(
    "hospital",
    "patient_id",
    "ICULOS"
).count().filter(
    col("count") > 1
)

print(
    "\nNúmero de registros duplicados por "
    "hospital + patient_id + ICULOS:"
)
print(duplicados.count())


# ==========================================================
# RESUMEN DE LA COBERTURA TEMPORAL
# ==========================================================

# CAMBIO: este bloque resume el seguimiento disponible por paciente;
# no se denomina comprobación del orden porque no verifica cada hora.
resumen_temporal = dataset.groupBy(
    "hospital",
    "patient_id"
).agg(
    min("ICULOS").alias("ICULOS_min"),
    max("ICULOS").alias("ICULOS_max"),
    count("ICULOS").alias("n_registros")
)

print("\nResumen temporal básico por paciente:")
resumen_temporal.show(10, truncate=False)


# CAMBIO: no se utiliza repartition(8), ya que fijar el número de
# particiones sin estudiar el entorno podría empeorar el rendimiento.

