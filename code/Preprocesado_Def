from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel

# ==========================================================
# SESIÓN SPARK
# ==========================================================

spark = SparkSession.builder \
    .appName("Preprocesado_Sepsis_Challenge2019") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 70)
print("PREPROCESADO DEL DATASET CHALLENGE 2019")
print("=" * 70)

# ==========================================================
# RUTAS
# ==========================================================

ruta_base = "/home/adminp/physionet.org/files/challenge-2019/1.0.0/training"

ruta_setA = ruta_base + "/training_setA/*.psv"
ruta_setB = ruta_base + "/training_setB/*.psv"

# ==========================================================
# LECTURA DE LOS CONJUNTOS A Y B
# ==========================================================

dfA = spark.read \
    .option("header", True) \
    .option("delimiter", "|") \
    .option("nullValue", "NaN") \
    .option("nanValue", "NaN") \
    .csv("file://" + ruta_setA)

dfB = spark.read \
    .option("header", True) \
    .option("delimiter", "|") \
    .option("nullValue", "NaN") \
    .option("nanValue", "NaN") \
    .csv("file://" + ruta_setB)

# ==========================================================
# IDENTIFICACIÓN DE PACIENTE Y HOSPITAL ANTES DE LA UNIÓN
# ==========================================================

dfA = dfA.withColumn(
    "patient_id",
    F.regexp_extract(F.input_file_name(), r"(p[0-9]+)\.psv", 1)
).withColumn(
    "hospital",
    F.lit("A")
)

dfB = dfB.withColumn(
    "patient_id",
    F.regexp_extract(F.input_file_name(), r"(p[0-9]+)\.psv", 1)
).withColumn(
    "hospital",
    F.lit("B")
)

dataset = dfA.unionByName(dfB)


# ==========================================================
# CONVERSIÓN DE TIPOS DE DATOS
# ==========================================================

variables_double = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
    "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
    "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets", "Age", "HospAdmTime", "ICULOS"
]

variables_integer = [
    "Gender",
    "Unit1",
    "Unit2",
    "SepsisLabel"
]

variables_string = [
    "hospital",
    "patient_id"
]

for variable in variables_double:
    dataset = dataset.withColumn(variable, F.col(variable).cast("double"))

for variable in variables_integer:
    dataset = dataset.withColumn(variable, F.col(variable).cast("int"))

for variable in variables_string:
    dataset = dataset.withColumn(variable, F.col(variable).cast("string"))

print("\nConversión de tipos completada.")
print("Variables continuas convertidas a DoubleType.")
print("Variables binarias convertidas a IntegerType.")
print("Identificadores convertidos a StringType.")



# ==========================================================
# NORMALIZACIÓN DE NaN COMO NULL
# ==========================================================

for variable in variables_double:
    dataset = dataset.withColumn(
        variable,
        F.when(F.isnan(F.col(variable)), None).otherwise(F.col(variable))
    )

print("\nValores NaN normalizados como NULL.")

# ==========================================================
# ELIMINACIÓN DE DUPLICADOS EXACTOS
# ==========================================================

registros_antes = dataset.count()
dataset = dataset.dropDuplicates()
registros_despues = dataset.count()

print("\nDuplicados exactos eliminados:")
print(registros_antes - registros_despues)

# ==========================================================
# PERSISTENCIA
# ==========================================================

dataset = dataset.persist(StorageLevel.MEMORY_AND_DISK)
total_registros = dataset.count()

# ==========================================================
# VALIDACIÓN INICIAL
# ==========================================================

print("\nPacientes por hospital:")
dataset.select("hospital", "patient_id").distinct() \
    .groupBy("hospital").count().orderBy("hospital").show()

numero_pacientes = dataset.select("hospital", "patient_id").distinct().count()

print("Número total de pacientes:", numero_pacientes)
print("Número total de registros horarios:", total_registros)

print("\nDistribución horaria de SepsisLabel:")
dataset.groupBy("SepsisLabel").count().orderBy("SepsisLabel").show()

print("\nPrimeras filas: identificación del paciente y evolución temporal:")
dataset.select(
    "hospital", "patient_id", "ICULOS", "Age", "HospAdmTime", "SepsisLabel"
).show(10, truncate=False)

print("\nPrimeras filas: constantes vitales principales:")
dataset.select(
    "HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"
).show(10, truncate=False)

print("\nEsquema del dataset:")
dataset.printSchema()

# ==========================================================
# VALIDACIÓN DE VARIABLES CLAVE
# ==========================================================

identificadores_invalidos = dataset.filter(
    F.col("patient_id").isNull() | (F.col("patient_id") == "")
).count()

hospitales_invalidos = dataset.filter(
    F.col("hospital").isNull() | ~F.col("hospital").isin("A", "B")
).count()

sepsis_invalidas = dataset.filter(
    F.col("SepsisLabel").isNotNull() & ~F.col("SepsisLabel").isin(0, 1)
).count()

iculos_invalidos = dataset.filter(
    F.col("ICULOS").isNull() | (F.col("ICULOS") < 1)
).count()

print("\nValidación de variables clave:")
print("Identificadores inválidos:", identificadores_invalidos)
print("Hospitales inválidos:", hospitales_invalidos)
print("Etiquetas de sepsis inválidas:", sepsis_invalidas)
print("Valores ICULOS inválidos:", iculos_invalidos)

# ==========================================================
# VALORES PERDIDOS
# ==========================================================

print("\n" + "=" * 70)
print("PORCENTAJE DE VALORES PERDIDOS")
print("=" * 70)

porcentajes_perdidos = {}

for variable in variables_double + variables_integer:
    nulos = dataset.filter(F.col(variable).isNull()).count()
    porcentaje = 100 * nulos / total_registros
    porcentajes_perdidos[variable] = porcentaje
    print(f"{variable:20s}: {porcentaje:6.2f}%")

variables_100 = [v for v, p in porcentajes_perdidos.items() if p == 100]
variables_95 = [v for v, p in porcentajes_perdidos.items() if p > 95]

print("\nVariables completamente vacías:")
print(variables_100)

print("\nVariables con más del 95 % de valores perdidos:")
print(variables_95)

if len(variables_100) > 0:
    dataset = dataset.drop(*variables_100)
    print("\nVariables completamente vacías eliminadas:")
    print(variables_100)
else:
    print("\nNo existen variables completamente vacías.")

# ==========================================================
# VARIABLE COMBINADA DE OXIGENACIÓN
# ==========================================================

dataset = dataset.withColumn(
    "O2Sat_combined",
    F.coalesce(F.col("O2Sat"), F.col("SaO2"))
)

print("\nVariable O2Sat_combined creada usando O2Sat y, si falta, SaO2.")

# ==========================================================
# ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES PRINCIPALES
# ==========================================================

print("\n" + "=" * 70)
print("ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES PRINCIPALES")
print("=" * 70)

variables_principales = [
    "HR", "O2Sat", "O2Sat_combined", "Temp", "SBP", "MAP", "Resp",
    "FiO2", "Lactate", "Age", "HospAdmTime", "ICULOS", "SepsisLabel"
]

for variable in variables_principales:
    fila = dataset.select(
        F.count(F.col(variable)).alias("n_validos"),
        F.avg(F.col(variable)).alias("media"),
        F.stddev(F.col(variable)).alias("desv_std"),
        F.min(F.col(variable)).alias("minimo"),
        F.max(F.col(variable)).alias("maximo")
    ).collect()[0]

    print(f"\nVariable: {variable}")
    print(f"  N válidos : {fila['n_validos']}")
    print(f"  Media     : {fila['media']:.2f}" if fila["media"] is not None else "  Media     : NA")
    print(f"  Desv.Std. : {fila['desv_std']:.2f}" if fila["desv_std"] is not None else "  Desv.Std. : NA")
    print(f"  Mínimo    : {fila['minimo']}")
    print(f"  Máximo    : {fila['maximo']}")

# ==========================================================
# DEPURACIÓN CONSERVADORA DE CONSTANTES VITALES
# ==========================================================

print("\n" + "=" * 70)
print("DEPURACIÓN CONSERVADORA DE CONSTANTES VITALES")
print("=" * 70)

rangos_limpieza = {
    "HR": (20, 250),
    "Temp": (33, 42),
    "SBP": (60, 240),
    "Resp": (4, 45),
    "FiO2": (0.20, 1.00)
}

for variable, (lim_inf, lim_sup) in rangos_limpieza.items():

    condicion_fuera = (
        F.col(variable).isNotNull() &
        ((F.col(variable) < lim_inf) | (F.col(variable) > lim_sup))
    )

    dataset = dataset.withColumn(
        variable + "_fuera_rango",
        F.when(condicion_fuera, 1).otherwise(0)
    )

    dataset = dataset.withColumn(
        variable + "_limpia",
        F.when(
            F.col(variable).isNotNull() &
            (F.col(variable) >= lim_inf) &
            (F.col(variable) <= lim_sup),
            F.col(variable)
        ).otherwise(None)
    )

    fuera = dataset.filter(F.col(variable + "_fuera_rango") == 1).count()
    porcentaje = 100 * fuera / total_registros

    print(f"{variable:10s}: {fuera:8d} registros fuera de rango ({porcentaje:6.4f}%)")

print("\nSe conservan las variables originales y se crean versiones limpias.")

# ==========================================================
# COHERENCIA ENTRE PRESIONES
# ==========================================================

print("\n" + "=" * 70)
print("COHERENCIA ENTRE SBP, MAP Y DBP")
print("=" * 70)

dataset = dataset.withColumn(
    "DBP_mayor_MAP",
    F.when(
        F.col("DBP").isNotNull() & F.col("MAP").isNotNull() &
        (F.col("DBP") > F.col("MAP")),
        1
    ).otherwise(0)
)

dataset = dataset.withColumn(
    "MAP_mayor_SBP",
    F.when(
        F.col("MAP").isNotNull() & F.col("SBP").isNotNull() &
        (F.col("MAP") > F.col("SBP")),
        1
    ).otherwise(0)
)

dataset = dataset.withColumn(
    "DBP_mayor_SBP",
    F.when(
        F.col("DBP").isNotNull() & F.col("SBP").isNotNull() &
        (F.col("DBP") > F.col("SBP")),
        1
    ).otherwise(0)
)

print("DBP > MAP:", dataset.filter(F.col("DBP_mayor_MAP") == 1).count())
print("MAP > SBP:", dataset.filter(F.col("MAP_mayor_SBP") == 1).count())
print("DBP > SBP:", dataset.filter(F.col("DBP_mayor_SBP") == 1).count())

# ==========================================================
# DEPURACIÓN CONSERVADORA ANALÍTICA
# ==========================================================

print("\n" + "=" * 70)
print("DEPURACIÓN CONSERVADORA DE VARIABLES ANALÍTICAS")
print("=" * 70)

rangos_analiticos = {
    "BaseExcess": (-40, 40),
    "HCO3": (0.1, 55),
    "Chloride": (50, 160),
    "Potassium": (1.5, 10),
    "Hgb": (2, 25)
}

for variable, (lim_inf, lim_sup) in rangos_analiticos.items():

    condicion_fuera = (
        F.col(variable).isNotNull() &
        ((F.col(variable) < lim_inf) | (F.col(variable) > lim_sup))
    )

    dataset = dataset.withColumn(
        variable + "_fuera_rango",
        F.when(condicion_fuera, 1).otherwise(0)
    )

    dataset = dataset.withColumn(
        variable + "_limpia",
        F.when(
            F.col(variable).isNotNull() &
            (F.col(variable) >= lim_inf) &
            (F.col(variable) <= lim_sup),
            F.col(variable)
        ).otherwise(None)
    )

    fuera = dataset.filter(F.col(variable + "_fuera_rango") == 1).count()
    porcentaje = 100 * fuera / total_registros

    print(f"{variable:15s}: {fuera:8d} registros fuera de rango ({porcentaje:6.4f}%)")

print("\nCalcium se deja pendiente para revisión exploratoria.")

# ==========================================================
# VARIABLES PENDIENTES Y SIN DEPURACIÓN AUTOMÁTICA
# ==========================================================

variables_dudosas = ["Calcium"]

variables_sin_depuracion_automatica = [
    "O2Sat", "EtCO2", "MAP", "DBP",
    "pH", "PaCO2", "SaO2",
    "BUN", "Alkalinephos", "Bilirubin_direct", "Bilirubin_total",
    "Magnesium", "Phosphate", "Creatinine", "Glucose", "Lactate", "AST",
    "Hct", "WBC", "Platelets", "PTT", "Fibrinogen", "TroponinI"
]

print("\nVariables pendientes de revisión específica:")
print(variables_dudosas)

print("\nVariables conservadas sin depuración automática:")
print(variables_sin_depuracion_automatica)

# ==========================================================
# RESUMEN DE VARIABLES DEPURADAS
# ==========================================================

print("\n" + "=" * 70)
print("RESUMEN DE VARIABLES DEPURADAS")
print("=" * 70)

variables_depuradas = [
    "HR", "Temp", "SBP", "Resp", "FiO2",
    "BaseExcess", "HCO3", "Chloride", "Potassium", "Hgb"
]

for variable in variables_depuradas:

    columna_limpia = variable + "_limpia"
    columna_flag = variable + "_fuera_rango"

    valores_validos = dataset.filter(F.col(columna_limpia).isNotNull()).count()
    valores_depurados = dataset.filter(F.col(columna_flag) == 1).count()

    print(
        f"{variable:15s}: "
        f"{valores_validos:8d} valores válidos | "
        f"{valores_depurados:6d} valores depurados"
    )

# ==========================================================
# PACIENTES CON Y SIN SEPSIS
# ==========================================================

print("\n" + "=" * 70)
print("PACIENTES CON Y SIN SEPSIS")
print("=" * 70)

pacientes_sepsis = dataset.groupBy("hospital", "patient_id").agg(
    F.max("SepsisLabel").alias("paciente_con_sepsis")
)

pacientes_sepsis.groupBy("paciente_con_sepsis").count() \
    .orderBy("paciente_con_sepsis").show()

tiempo_primera_sepsis = dataset.filter(F.col("SepsisLabel") == 1) \
    .groupBy("hospital", "patient_id") \
    .agg(F.min("ICULOS").alias("primera_hora_sepsis"))

print("\nTiempo hasta primera aparición de SepsisLabel=1:")
tiempo_primera_sepsis.agg(
    F.count("*").alias("n_pacientes_sepsis"),
    F.mean("primera_hora_sepsis").alias("media_horas"),
    F.stddev("primera_hora_sepsis").alias("desv_std"),
    F.min("primera_hora_sepsis").alias("min_horas"),
    F.max("primera_hora_sepsis").alias("max_horas")
).show(truncate=False)

# ==========================================================
# DUPLICADOS HORARIOS CONFLICTIVOS
# ==========================================================

duplicados_temporales = dataset.groupBy(
    "hospital", "patient_id", "ICULOS"
).count().filter(F.col("count") > 1)

numero_duplicados_temporales = duplicados_temporales.count()

print("\nCombinaciones repetidas de hospital + patient_id + ICULOS:")
print(numero_duplicados_temporales)

if numero_duplicados_temporales > 0:
    print("Existen registros horarios conflictivos que deberán revisarse.")
else:
    print("No existen registros horarios conflictivos.")

# ==========================================================
# RESUMEN TEMPORAL BÁSICO
# ==========================================================

resumen_temporal = dataset.groupBy("hospital", "patient_id").agg(
    F.min("ICULOS").alias("ICULOS_min"),
    F.max("ICULOS").alias("ICULOS_max"),
    F.count("ICULOS").alias("n_registros")
)

print("\nResumen temporal básico por paciente:")
resumen_temporal.show(10, truncate=False)

# ==========================================================
# VARIABLES TEMPORALES
# ==========================================================

print("\n" + "=" * 70)
print("CREACIÓN DE VARIABLES TEMPORALES")
print("=" * 70)

ventana_paciente = Window.partitionBy("hospital", "patient_id").orderBy("ICULOS")
ventana_baseline = Window.partitionBy("hospital", "patient_id").orderBy("ICULOS") \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

for variable in ["HR", "MAP", "FiO2", "Lactate"]:

    dataset = dataset.withColumn(
        variable + "_prev",
        F.lag(F.col(variable)).over(ventana_paciente)
    )

    dataset = dataset.withColumn(
        variable + "_baseline",
        F.first(F.col(variable), ignorenulls=True).over(ventana_baseline)
    )

    dataset = dataset.withColumn(
        variable + "_delta_prev",
        F.when(
            F.col(variable).isNotNull() & F.col(variable + "_prev").isNotNull(),
            F.round(F.col(variable) - F.col(variable + "_prev"), 2)
        )
    )

    dataset = dataset.withColumn(
        variable + "_delta_baseline",
        F.when(
            F.col(variable).isNotNull() & F.col(variable + "_baseline").isNotNull(),
            F.round(F.col(variable) - F.col(variable + "_baseline"), 2)
        )
    )

print("Variables temporales creadas correctamente.")

print("\nEJEMPLO DE EVOLUCIÓN TEMPORAL (HR y MAP)\n")

dataset.select(
    "hospital",
    "patient_id",
    "ICULOS",
    "HR",
    "HR_delta_prev",
    "HR_delta_baseline",
    "MAP",
    "MAP_delta_prev",
    "MAP_delta_baseline",
    "SepsisLabel"
).show(
    10,
    truncate=False,
    vertical=True
)
# ==========================================================
# TENDENCIAS DURANTE LAS PRIMERAS 24 HORAS
# ==========================================================

# ==========================================================
# TENDENCIAS DURANTE LAS PRIMERAS 24 HORAS
# ==========================================================

primeras_24h = dataset.filter(F.col("ICULOS") <= 24)

tendencias_24h = primeras_24h.groupBy("hospital", "patient_id").agg(
    F.min("Lactate").alias("lactato_min_24h"),
    F.max("Lactate").alias("lactato_max_24h"),
    F.min("FiO2").alias("fio2_min_24h"),
    F.max("FiO2").alias("fio2_max_24h"),
    F.min("O2Sat_combined").alias("o2sat_min_24h"),
    F.max("O2Sat_combined").alias("o2sat_max_24h"),
    F.min("MAP").alias("map_min_24h"),
    F.max("MAP").alias("map_max_24h")
)

tendencias_24h = tendencias_24h.withColumn(
    "delta_lactato_24h", F.round(F.col("lactato_max_24h") - F.col("lactato_min_24h"), 2)
).withColumn(
    "delta_fio2_24h", F.round(F.col("fio2_max_24h") - F.col("fio2_min_24h"), 2)
).withColumn(
    "delta_o2sat_24h", F.round(F.col("o2sat_max_24h") - F.col("o2sat_min_24h"), 2)
).withColumn(
    "delta_map_24h", F.round(F.col("map_max_24h") - F.col("map_min_24h"), 2)
)

print("\n" + "=" * 70)
print("RESUMEN DE TENDENCIAS TEMPORALES DURANTE LAS PRIMERAS 24 HORAS")
print("=" * 70)

tendencias_24h.select(
    "delta_lactato_24h",
    "delta_fio2_24h",
    "delta_o2sat_24h",
    "delta_map_24h"
).describe().show(truncate=False)

dataset = dataset.join(
    tendencias_24h,
    on=["hospital", "patient_id"],
    how="left"
)


#=============================================
#RESUMEN DE VALORES FISIOLOGICAMENTE POCO PLAUSIBLES
#=================================================

print("\n" + "=" * 70)
print("RESUMEN DE VALORES FISIOLÓGICAMENTE POCO PLAUSIBLES")
print("=" * 70)

flags_fuera_rango = [
    "HR_fuera_rango",
    "Temp_fuera_rango",
    "SBP_fuera_rango",
    "Resp_fuera_rango",
    "FiO2_fuera_rango",
    "BaseExcess_fuera_rango",
    "HCO3_fuera_rango",
    "Chloride_fuera_rango",
    "Potassium_fuera_rango",
    "Hgb_fuera_rango"
]

for flag in flags_fuera_rango:
    if flag in dataset.columns:
        n = dataset.filter(F.col(flag) == 1).count()
        porcentaje = 100 * n / total_registros
        print(f"{flag:25s}: {n:8d} registros ({porcentaje:6.4f}%)")

# ==========================================================
# REPARTITION
# ==========================================================

print("\nNúmero de particiones antes de repartition:")
print(dataset.rdd.getNumPartitions())

dataset = dataset.repartition(8)

print("\nNúmero de particiones después de repartition:")
print(dataset.rdd.getNumPartitions())

# ==========================================================
# VALIDACIÓN FINAL
# ==========================================================

print("\n" + "=" * 70)
print("VALIDACIÓN FINAL DEL DATASET PREPROCESADO")
print("=" * 70)

print("Número final de registros:", dataset.count())
print("Número final de pacientes:", dataset.select("hospital", "patient_id").distinct().count())
print("Número final de columnas:", len(dataset.columns))

print("\nPacientes finales por hospital:")
dataset.select("hospital", "patient_id").distinct() \
    .groupBy("hospital").count().orderBy("hospital").show()

# ==========================================================
# GUARDADO EN PARQUET
# ==========================================================

ruta_salida_local = "/home/adminp/TFA/parquet/dataset_preprocesado"
ruta_salida_spark = "file://" + ruta_salida_local

dataset.write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(ruta_salida_spark)

print("\nDataset preprocesado guardado correctamente en formato Parquet:")
print(ruta_salida_local)

# ==========================================================
# COMPROBACIÓN DEL PARQUET
# ==========================================================

dataset_comprobacion = spark.read.parquet(ruta_salida_spark)

print("\nRegistros recuperados del Parquet:")
print(dataset_comprobacion.count())

print("\nPacientes recuperados del Parquet:")
print(dataset_comprobacion.select("hospital", "patient_id").distinct().count())

print("\nColumnas recuperadas del Parquet:")
print(len(dataset_comprobacion.columns))

dataset.unpersist()

spark.stop()

