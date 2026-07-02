# Importa la herramienta necesaria para iniciar una sesión de Spark
from pyspark.sql import SparkSession
# Importa las funciones necesarias en este primer bloque
from pyspark.sql.functions import (
    col,                # Permite seleccionar y transformar columnas
    input_file_name,    # Obtiene el nombre del archivo de origen
    regexp_extract,     # Extrae el identificador del paciente del nombre del archivo
    lit,                # Crea una columna con un valor constante
    isnan               # Detecta valores NaN en columnas numéricas
)


# ==========================================================
# INICIO DE LA SESIÓN SPARK
# ==========================================================

# Crea la sesión de Spark para el preprocesado
spark = SparkSession.builder \
    .appName("Preprocesado_Sepsis_Challenge2019") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Reduce los mensajes informativos mostrados por Spark
spark.sparkContext.setLogLevel("ERROR")


# ==========================================================
# RUTAS DE LOS ARCHIVOS
# ==========================================================

# Define la ruta local del dataset
# Esta ruta deberá adaptarse en cada máquina si fuera diferente
ruta_base = (
    "/home/adminp/physionet.org/files/"
    "challenge-2019/1.0.0/training"
)

# Define la ruta de los archivos del hospital A
ruta_setA = ruta_base + "/training_setA/*.psv"

# Define la ruta de los archivos del hospital B
ruta_setB = ruta_base + "/training_setB/*.psv"


# ==========================================================
# MENSAJE DE INICIO
# ==========================================================

# Muestra el encabezado del proceso
print("=" * 70)
print("PREPROCESADO DEL DATASET CHALLENGE 2019")
print("=" * 70)


# ==========================================================
# LECTURA DEL HOSPITAL A
# ==========================================================

# Lee todos los archivos PSV del hospital A
dfA = spark.read \
    .option("header", True) \
    .option("delimiter", "|") \
    .option("nullValue", "NaN") \
    .option("nanValue", "NaN") \
    .csv("file://" + ruta_setA)


# ==========================================================
# LECTURA DEL HOSPITAL B
# ==========================================================

# Lee todos los archivos PSV del hospital B
dfB = spark.read \
    .option("header", True) \
    .option("delimiter", "|") \
    .option("nullValue", "NaN") \
    .option("nanValue", "NaN") \
    .csv("file://" + ruta_setB)


# ==========================================================
# IDENTIFICACIÓN DEL PACIENTE Y DEL HOSPITAL
# ==========================================================

# Extrae el identificador del paciente del nombre del archivo del hospital A
dfA = dfA.withColumn(
    "patient_id",
    regexp_extract(
        input_file_name(),
        r"(p[0-9]+)\.psv",
        1
    )
)

# Añade la procedencia hospitalaria a los registros del conjunto A
dfA = dfA.withColumn(
    "hospital",
    lit("A")
)

# Extrae el identificador del paciente del nombre del archivo del hospital B
dfB = dfB.withColumn(
    "patient_id",
    regexp_extract(
        input_file_name(),
        r"(p[0-9]+)\.psv",
        1
    )
)

# Añade la procedencia hospitalaria a los registros del conjunto B
dfB = dfB.withColumn(
    "hospital",
    lit("B")
)


# ==========================================================
# UNIÓN DE LOS DOS HOSPITALES
# ==========================================================

# Une los conjuntos A y B conservando el hospital de procedencia
dataset = dfA.unionByName(dfB)


# ==========================================================
# CONVERSIÓN DE VARIABLES NUMÉRICAS
# ==========================================================

# Define las columnas que deben convertirse a formato numérico
variables_numericas = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
    "SepsisLabel"
]

# Convierte cada variable de la lista al tipo double
for variable in variables_numericas:
    dataset = dataset.withColumn(
        variable,
        col(variable).cast("double")
    )

# Confirma que la lectura y la conversión se han definido
print(
    "\nDataset leído, unido y convertido "
    "a formato numérico correctamente."
)


# ==========================================================
# VALIDACIÓN INICIAL
# ==========================================================

# Cuenta los pacientes diferentes en cada hospital
print("\nNúmero de pacientes por hospital:")

dataset.select(
    "hospital",
    "patient_id"
).distinct() \
 .groupBy("hospital") \
 .count() \
 .orderBy("hospital") \
 .show()

# Cuenta los pacientes utilizando hospital y patient_id
numero_pacientes = dataset.select(
    "hospital",
    "patient_id"
).distinct().count()

# Muestra el número total de pacientes
print("Número total de pacientes:", numero_pacientes)

# Calcula una sola vez el número total de registros horarios
total_registros = dataset.count()

# Muestra el número total de registros horarios
print("\nNúmero total de registros horarios:")
print(total_registros)

# Muestra la distribución horaria de la variable resultado
print("\nDistribución horaria de SepsisLabel:")

dataset.groupBy(
    "SepsisLabel"
).count() \
 .orderBy("SepsisLabel") \
 .show()

# Muestra algunas columnas de las primeras filas
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

# Muestra los tipos de datos del dataset
print("\nEsquema del dataset:")
dataset.printSchema()


# ==========================================================
# ANÁLISIS DE VALORES PERDIDOS
# ==========================================================

# Muestra el encabezado del análisis de valores ausentes
print("\n" + "=" * 70)
print("PORCENTAJE DE VALORES PERDIDOS")
print("=" * 70)

# Recorre todas las variables numéricas
for variable in variables_numericas:

    # Cuenta conjuntamente los valores nulos y los valores NaN
    nulos = dataset.filter(
        col(variable).isNull()
        | isnan(col(variable))
    ).count()

    # Calcula el porcentaje de valores perdidos
    porcentaje = (
        nulos / total_registros
    ) * 100

    # Muestra el porcentaje de valores perdidos de cada variable
    print(
        f"{variable:20s}: "
        f"{porcentaje:6.2f}%"
    )
