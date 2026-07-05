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
print("Septicos y no septicos")
print("=" * 70)

print("\nDistribución global de pacientes con y sin sepsis:")

tabla_sepsis_global = dataset_paciente.groupBy(
    "paciente_con_sepsis"
).agg(
    F.count("*").alias("n_pacientes")
).withColumn(
    "porcentaje",
    F.round(
        100 * F.col("n_pacientes") / F.lit(total_pacientes),
        2
    )
).orderBy("paciente_con_sepsis")

tabla_sepsis_global.show()


print("\nDistribución de pacientes con y sin sepsis por hospital:")

totales_hospital = dataset_paciente.groupBy("hospital").agg(
    F.count("*").alias("total_pacientes_hospital")
)

tabla_sepsis_hospital = dataset_paciente.groupBy(
    "hospital",
    "paciente_con_sepsis"
).agg(
    F.count("*").alias("n_pacientes")
).join(
    totales_hospital,
    on="hospital",
    how="left"
).withColumn(
    "porcentaje_hospital",
    F.round(
        100 * F.col("n_pacientes") / F.col("total_pacientes_hospital"),
        2
    )
).orderBy(
    "hospital",
    "paciente_con_sepsis"
)

tabla_sepsis_hospital.show()


print("\nResumen de la primera hora de aparición de SepsisLabel = 1:")

dataset_paciente_sepsis = dataset_paciente.filter(
    F.col("paciente_con_sepsis") == 1
)

resumen_primera_sepsis = dataset_paciente_sepsis.agg(
    F.count("*").alias("n_pacientes_sepsis"),
    F.mean("primera_hora_sepsis").alias("media_horas"),
    F.stddev("primera_hora_sepsis").alias("desv_std"),
    F.min("primera_hora_sepsis").alias("min_horas"),
    F.max("primera_hora_sepsis").alias("max_horas")
)

resumen_primera_sepsis.show(truncate=False)


# Mediana y rango intercuartílico de la primera hora de sepsis.
# Es importante porque el tiempo hasta sepsis suele tener distribución asimétrica.

if dataset_paciente_sepsis.count() > 0:

    cuantiles_primera_sepsis = dataset_paciente_sepsis.approxQuantile(
        "primera_hora_sepsis",
        [0.25, 0.50, 0.75],
        0.01
    )

    print("\nPrimera hora de sepsis - p25, mediana, p75:")
    print(cuantiles_primera_sepsis)

else:
    print("\nNo hay pacientes con sepsis para calcular cuantiles.")

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

# Sepsis vs No Sepsis a nivel paciente
# Ademas de la desviacion y media para el tipo de dataset es interesante añadir Percentil(25/75) y Mediana

print("\n" + "=" * 70)
print("Sepsis vs No Sepsis a nivel paciente")
print("=" * 70)


comparacion_sepsis = dataset_paciente.groupBy(
    "paciente_con_sepsis"
).agg(
    F.count("*").alias("n_pacientes"),

    F.mean("Age").alias("edad_media"),
    F.stddev("Age").alias("edad_desv_std"),
    F.expr("percentile_approx(Age, 0.25, 10000)").alias("edad_p25"),
    F.expr("percentile_approx(Age, 0.50, 10000)").alias("edad_mediana"),
    F.expr("percentile_approx(Age, 0.75, 10000)").alias("edad_p75"),

    F.mean("ICULOS_max").alias("iculos_max_media"),
    F.stddev("ICULOS_max").alias("iculos_max_desv_std"),
    F.expr("percentile_approx(ICULOS_max, 0.25, 10000)").alias("iculos_max_p25"),
    F.expr("percentile_approx(ICULOS_max, 0.50, 10000)").alias("iculos_max_mediana"),
    F.expr("percentile_approx(ICULOS_max, 0.75, 10000)").alias("iculos_max_p75")
).orderBy(
    "paciente_con_sepsis"
)

comparacion_sepsis.show(truncate=False)

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

# Resumen de Flags existentes

print("\n" + "=" * 70)
print("Resumen de Flags existentes")
print("=" * 70)

flags_calidad = [
    columna for columna in dataset.columns
    if columna.endswith("_fuera_rango")
]

# Añadimos indicadores de incoherencia entre presiones.
flags_presion = [
    "DBP_mayor_MAP",
    "MAP_mayor_SBP",
    "DBP_mayor_SBP"
]

flags_calidad = flags_calidad + [
    flag for flag in flags_presion
    if flag in dataset.columns
]

filas_flags = []

for flag in flags_calidad:

    n_flag = dataset.filter(
        F.col(flag) == 1
    ).count()

    filas_flags.append(
        (
            flag,
            n_flag,
            round(100 * n_flag / n_registros, 4)
        )
    )

if len(filas_flags) > 0:

    tabla_flags = spark.createDataFrame(
        filas_flags,
        [
            "flag",
            "n_registros",
            "porcentaje_registros"
        ]
    )

    tabla_flags.orderBy(
        F.desc("porcentaje_registros")
    ).show(
        truncate=False
    )

else:
    print("No se encontraron columnas de flags de calidad.")


# Revision de 02Sat, SaO2 y O2Sat_Combined

print("\n" + "=" * 70)
print("Revision de 02Sat, SaO2 y O2Sat_Combined")
print("=" * 70)

columnas_oxigenacion = [
    "O2Sat",
    "SaO2",
    "O2Sat_combined"
]

columnas_oxigenacion = [
    columna for columna in columnas_oxigenacion
    if columna in dataset.columns
]

if len(columnas_oxigenacion) > 0:

    print("\nNúmero de valores válidos en variables de oxigenación:")

    dataset.select(
        [
            F.count(F.col(columna)).alias(columna + "_n_validos")
            for columna in columnas_oxigenacion
        ]
    ).show(truncate=False)

    print("\nEstadísticos descriptivos de variables de oxigenación:")

    dataset.select(
        columnas_oxigenacion
    ).describe().show(
        truncate=False
    )

else:
    print("No se encontraron variables de oxigenación en el dataset.")


if "O2Sat" in dataset.columns and "SaO2" in dataset.columns:

    registros_ambas = dataset.filter(
        F.col("O2Sat").isNotNull()
        & F.col("SaO2").isNotNull()
    ).count()

    print("\nRegistros con O2Sat y SaO2 disponibles simultáneamente:")
    print(registros_ambas)

    if registros_ambas > 0:

        print("\nDiferencia entre O2Sat y SaO2 cuando ambas están disponibles:")

        dataset.filter(
            F.col("O2Sat").isNotNull()
            & F.col("SaO2").isNotNull()
        ).select(
            F.mean(F.col("O2Sat") - F.col("SaO2")).alias("diferencia_media"),
            F.stddev(F.col("O2Sat") - F.col("SaO2")).alias("diferencia_desv_std"),
            F.min(F.col("O2Sat") - F.col("SaO2")).alias("diferencia_min"),
            F.max(F.col("O2Sat") - F.col("SaO2")).alias("diferencia_max")
        ).show(truncate=False)


# Revision la variable calcium

print("\n" + "=" * 70)
print("Revision de la variable calcium")
print("=" * 70)

if "Calcium" in dataset.columns:

    n_calcium = dataset.filter(
        F.col("Calcium").isNotNull()
    ).count()

    print("\nNúmero de registros con Calcium disponible:")
    print(n_calcium)

    print("\nEstadísticos descriptivos de Calcium:")

    dataset.select(
        "Calcium"
    ).describe().show(
        truncate=False
    )

    print("\nCalcium por hospital:")

    dataset.groupBy("hospital").agg(
        F.count("Calcium").alias("n_validos"),
        F.mean("Calcium").alias("media"),
        F.stddev("Calcium").alias("desv_std"),
        F.min("Calcium").alias("minimo"),
        F.max("Calcium").alias("maximo")
    ).orderBy("hospital") \
     .show(truncate=False)

    if n_calcium > 0:

        cuantiles_calcium = dataset.approxQuantile(
            "Calcium",
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
            0.01
        )

        print("\nCuantiles aproximados de Calcium:")
        print("p01, p05, p25, p50, p75, p95, p99")
        print(cuantiles_calcium)

else:
    print("La variable Calcium no está presente en el dataset.")

# Varibales por SepsisLabel

print("\n" + "=" * 70)
print("Variables por SepsisLabel")
print("=" * 70)

variables_clinicas_principales = [
    "HR_limpia",
    "Temp_limpia",
    "SBP_limpia",
    "Resp_limpia",
    "FiO2_limpia",
    "MAP",
    "Lactate",
    "O2Sat_combined",
    "BaseExcess_limpia",
    "HCO3_limpia",
    "Potassium_limpia",
    "Hgb_limpia"
]

variables_clinicas_principales = [
    variable for variable in variables_clinicas_principales
    if variable in dataset.columns
]

for variable in variables_clinicas_principales:

    print(f"\nVariable: {variable}")

    dataset.groupBy("SepsisLabel").agg(
        F.count(variable).alias("n_validos"),
        F.mean(variable).alias("media"),
        F.stddev(variable).alias("desv_std"),
        F.min(variable).alias("minimo"),
        F.max(variable).alias("maximo")
    ).orderBy("SepsisLabel") \
     .show(truncate=False)

# Tendencia las primeras 24 horas

print("\n" + "=" * 70)
print("Tendencia la primeras 24 horas")
print("=" * 70)

#Echale un vistazo a esto en PORFA
variables_tendencias_24h = [
    "lactato_min_24h",
    "lactato_max_24h",
    "fio2_min_24h",
    "fio2_max_24h",
    "o2sat_min_24h",
    "o2sat_max_24h",
    "map_min_24h",
    "map_max_24h",
    "delta_lactato_24h",
    "delta_fio2_24h",
    "delta_o2sat_24h",
    "delta_map_24h"
]

variables_tendencias_24h = [
    variable for variable in variables_tendencias_24h
    if variable in dataset.columns
]

if len(variables_tendencias_24h) > 0:

    dataset.select(
        variables_tendencias_24h
    ).describe().show(
        truncate=False
    )

else:
    print("No se encontraron variables de tendencias de primeras 24 horas.")

# Resumen final

print("\n" + "=" * 70)
print("Resumen final")
print("=" * 70)

print("\nEste script ha realizado:")
print("- Validación del Parquet preprocesado.")
print("- Recuento de registros, pacientes y variables.")
print("- Comprobación de variables clave.")
print("- Distribución de registros y pacientes por hospital.")
print("- Construcción de un dataset a nivel paciente.")
print("- Descripción global de la cohorte.")
print("- Análisis de pacientes con y sin sepsis.")
print("- Comparación intercentro a nivel paciente.")
print("- Comparación entre pacientes con y sin sepsis.")
print("- Revisión de valores perdidos globales y por hospital.")
print("- Revisión de variables depuradas y flags de calidad.")
print("- Revisión de O2Sat, SaO2 y O2Sat_combined.")
print("- Revisión específica de Calcium.")
print("- Comparación de variables clínicas principales por SepsisLabel.")
print("- Revisión descriptiva de variables temporales.")
print("- Revisión de tendencias durante las primeras 24 horas.")

print("\nNo se han aplicado nuevas limpiezas ni transformaciones definitivas.")
print("El objetivo ha sido explorar, validar y resumir el dataset preprocesado.")


# ==========================================================
# CIERRE DE SPARK
# ==========================================================

spark.stop()
