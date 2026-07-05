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

# Sepsis golbal y por hospital

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



# Distribución por Genero

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

# Missing por hospital

print("\n" + "=" * 70)
print("Missing por hospital")
print("=" * 70)

variables_missing_clinicas = [
    "Lactate",
    "MAP",
    "FiO2",
    "O2Sat",
    "SaO2",
    "O2Sat_combined",
    "Resp",
    "Temp",
    "HR",
    "Creatinine",
    "WBC",
    "Platelets",
    "Hgb",
    "Calcium"
]

variables_missing_clinicas = [
    variable for variable in variables_missing_clinicas
    if variable in dataset.columns
]

tablas_missing = []

for variable in variables_missing_clinicas:

    tabla_variable = dataset.groupBy(
        "hospital"
    ).agg(
        F.count("*").alias("n_registros"),
        F.sum(
            F.when(
                F.col(variable).isNull(),
                1
            ).otherwise(0)
        ).alias("n_nulos")
    ).withColumn(
        "variable",
        F.lit(variable)
    ).withColumn(
        "porcentaje_nulos",
        F.round(
            100 * F.col("n_nulos") / F.col("n_registros"),
            2
        )
    ).select(
        "variable",
        "hospital",
        "porcentaje_nulos"
    )

    tablas_missing.append(tabla_variable)


if len(tablas_missing) > 0:

    tabla_missing_hospital = tablas_missing[0]

    for tabla in tablas_missing[1:]:
        tabla_missing_hospital = tabla_missing_hospital.unionByName(tabla)

    fig_missing_hospital = px.bar(
        spark_a_pandas(tabla_missing_hospital),
        x="variable",
        y="porcentaje_nulos",
        color="hospital",
        barmode="group",
        title="Valores perdidos por hospital en variables clínicas principales",
        labels={
            "variable": "Variable",
            "porcentaje_nulos": "Valores perdidos (%)",
            "hospital": "Hospital"
        }
    )

    guardar_grafico_plotly(
        fig_missing_hospital,
        "08_missing_por_hospital_variables_clinicas.html"
    )

else:
    print("No se encontraron variables clínicas para calcular missing.")


# LACTATO


print("\n" + "=" * 70)
print("9. LACTATO")
print("=" * 70)


if "Lactate" in dataset.columns:

    lactato_pandas = dataset.select(
        "SepsisLabel",
        "Lactate"
    ).filter(
        F.col("Lactate").isNotNull()
    ).withColumn(
        "grupo_sepsis",
        F.when(
            F.col("SepsisLabel") == 1,
            F.lit("SepsisLabel = 1")
        ).otherwise(
            F.lit("SepsisLabel = 0")
        )
    ).toPandas()

    fig_lactato = px.box(
        lactato_pandas,
        x="grupo_sepsis",
        y="Lactate",
        title="Lactato por SepsisLabel",
        labels={
            "grupo_sepsis": "Grupo",
            "Lactate": "Lactato"
        }
    )

    guardar_grafico_plotly(
        fig_lactato,
        "09_lactato_por_sepsislabel.html"
    )

else:
    print("La variable Lactate no está disponible.")


# Delta de lactato en primeras 24 horas


if "delta_lactato_24h" in dataset.columns:

    delta_lactato_pandas = dataset.select(
        "hospital",
        "patient_id",
        "delta_lactato_24h"
    ).dropDuplicates(
        ["hospital", "patient_id"]
    ).filter(
        F.col("delta_lactato_24h").isNotNull()
    ).toPandas()

    fig_delta_lactato = px.histogram(
        delta_lactato_pandas,
        x="delta_lactato_24h",
        nbins=40,
        title="Distribución del delta de lactato durante las primeras 24 horas",
        labels={
            "delta_lactato_24h": "Delta lactato 24h"
        }
    )

    guardar_grafico_plotly(
        fig_delta_lactato,
        "10_delta_lactato_24h.html"
    )

else:
    print("La variable delta_lactato_24h no está disponible.")


#Presion arterial y perfusion

print("\n" + "=" * 70)
print("Presion arteria y perfusion")
print("=" * 70)


if "MAP" in dataset.columns:

    map_pandas = dataset.select(
        "SepsisLabel",
        "MAP"
    ).filter(
        F.col("MAP").isNotNull()
    ).withColumn(
        "grupo_sepsis",
        F.when(
            F.col("SepsisLabel") == 1,
            F.lit("SepsisLabel = 1")
        ).otherwise(
            F.lit("SepsisLabel = 0")
        )
    ).toPandas()

    fig_map = px.box(
        map_pandas,
        x="grupo_sepsis",
        y="MAP",
        title="Presión arterial media por SepsisLabel",
        labels={
            "grupo_sepsis": "Grupo",
            "MAP": "MAP"
        }
    )

    guardar_grafico_plotly(
        fig_map,
        "11_map_por_sepsislabel.html"
    )

else:
    print("La variable MAP no está disponible.")


# Delta de MAP en primeras 24 horas

if "delta_map_24h" in dataset.columns:

    delta_map_pandas = dataset.select(
        "hospital",
        "patient_id",
        "delta_map_24h"
    ).dropDuplicates(
        ["hospital", "patient_id"]
    ).filter(
        F.col("delta_map_24h").isNotNull()
    ).toPandas()

    fig_delta_map = px.histogram(
        delta_map_pandas,
        x="delta_map_24h",
        nbins=40,
        title="Distribución del delta de MAP durante las primeras 24 horas",
        labels={
            "delta_map_24h": "Delta MAP 24h"
        }
    )

    guardar_grafico_plotly(
        fig_delta_map,
        "12_delta_map_24h.html"
    )

else:
    print("La variable delta_map_24h no está disponible.")


# Incoherencias entre presiones arteriales


flags_presion = [
    "DBP_mayor_MAP",
    "MAP_mayor_SBP",
    "DBP_mayor_SBP"
]

flags_presion = [
    flag for flag in flags_presion
    if flag in dataset.columns
]

filas_flags_presion = []

for flag in flags_presion:

    n_flag = dataset.filter(
        F.col(flag) == 1
    ).count()

    filas_flags_presion.append(
        (
            flag,
            n_flag,
            round(100 * n_flag / n_registros, 4)
        )
    )

if len(filas_flags_presion) > 0:

    tabla_flags_presion = spark.createDataFrame(
        filas_flags_presion,
        [
            "flag",
            "n_registros",
            "porcentaje_registros"
        ]
    )

    fig_flags_presion = px.bar(
        spark_a_pandas(tabla_flags_presion),
        x="flag",
        y="porcentaje_registros",
        title="Incoherencias entre presiones arteriales",
        labels={
            "flag": "Flag",
            "porcentaje_registros": "Registros afectados (%)"
        },
        hover_data=["n_registros"]
    )

    guardar_grafico_plotly(
        fig_flags_presion,
        "13_incoherencias_presion_arterial.html"
    )

else:
    print("No se encontraron flags de incoherencia entre presiones.")


# oxigencion y soporte respiratorio

print("\n" + "=" * 70)
print("oxigenacion y soporte respiratorio")
print("=" * 70)


# FiO2 limpia por SepsisLabel


if "FiO2_limpia" in dataset.columns:

    fio2_pandas = dataset.select(
        "SepsisLabel",
        "FiO2_limpia"
    ).filter(
        F.col("FiO2_limpia").isNotNull()
    ).withColumn(
        "grupo_sepsis",
        F.when(
            F.col("SepsisLabel") == 1,
            F.lit("SepsisLabel = 1")
        ).otherwise(
            F.lit("SepsisLabel = 0")
        )
    ).toPandas()

    fig_fio2 = px.box(
        fio2_pandas,
        x="grupo_sepsis",
        y="FiO2_limpia",
        title="FiO2 limpia por SepsisLabel",
        labels={
            "grupo_sepsis": "Grupo",
            "FiO2_limpia": "FiO2 limpia"
        }
    )

    guardar_grafico_plotly(
        fig_fio2,
        "14_fio2_limpia_por_sepsislabel.html"
    )

else:
    print("La variable FiO2_limpia no está disponible.")


# Disponibilidad de O2Sat, SaO2 y O2Sat_combined

variables_oxigenacion = [
    "O2Sat",
    "SaO2",
    "O2Sat_combined"
]

variables_oxigenacion = [
    variable for variable in variables_oxigenacion
    if variable in dataset.columns
]

filas_oxigenacion = []

for variable in variables_oxigenacion:

    n_validos = dataset.filter(
        F.col(variable).isNotNull()
    ).count()

    filas_oxigenacion.append(
        (
            variable,
            n_validos,
            round(100 * n_validos / n_registros, 2)
        )
    )

if len(filas_oxigenacion) > 0:

    tabla_oxigenacion = spark.createDataFrame(
        filas_oxigenacion,
        [
            "variable",
            "n_validos",
            "porcentaje_validos"
        ]
    )

    fig_oxigenacion = px.bar(
        spark_a_pandas(tabla_oxigenacion),
        x="variable",
        y="porcentaje_validos",
        title="Disponibilidad de variables de oxigenación",
        labels={
            "variable": "Variable",
            "porcentaje_validos": "Valores válidos (%)"
        },
        hover_data=["n_validos"]
    )

    guardar_grafico_plotly(
        fig_oxigenacion,
        "15_disponibilidad_oxigenacion.html"
    )

else:
    print("No se encontraron variables de oxigenación.")


# O2Sat_combined por SepsisLabel


if "O2Sat_combined" in dataset.columns:

    o2sat_combined_pandas = dataset.select(
        "SepsisLabel",
        "O2Sat_combined"
    ).filter(
        F.col("O2Sat_combined").isNotNull()
    ).withColumn(
        "grupo_sepsis",
        F.when(
            F.col("SepsisLabel") == 1,
            F.lit("SepsisLabel = 1")
        ).otherwise(
            F.lit("SepsisLabel = 0")
        )
    ).toPandas()

    fig_o2sat_combined = px.box(
        o2sat_combined_pandas,
        x="grupo_sepsis",
        y="O2Sat_combined",
        title="O2Sat_combined por SepsisLabel",
        labels={
            "grupo_sepsis": "Grupo",
            "O2Sat_combined": "O2Sat_combined"
        }
    )

    guardar_grafico_plotly(
        fig_o2sat_combined,
        "16_o2sat_combined_por_sepsislabel.html"
    )

else:
    print("La variable O2Sat_combined no está disponible.")


# Diferencia O2Sat - SaO2 cuando ambas están disponibles


if "O2Sat" in dataset.columns and "SaO2" in dataset.columns:

    diferencia_oxigenacion = dataset.filter(
        F.col("O2Sat").isNotNull()
        & F.col("SaO2").isNotNull()
    ).withColumn(
        "diferencia_O2Sat_SaO2",
        F.col("O2Sat") - F.col("SaO2")
    ).select(
        "diferencia_O2Sat_SaO2"
    ).toPandas()

    if len(diferencia_oxigenacion) > 0:

        fig_diferencia_oxigenacion = px.histogram(
            diferencia_oxigenacion,
            x="diferencia_O2Sat_SaO2",
            nbins=50,
            title="Diferencia entre O2Sat y SaO2 cuando ambas están disponibles",
            labels={
                "diferencia_O2Sat_SaO2": "O2Sat - SaO2"
            }
        )

        guardar_grafico_plotly(
            fig_diferencia_oxigenacion,
            "17_diferencia_o2sat_sao2.html"
        )

    else:
        print("No hay registros con O2Sat y SaO2 disponibles simultáneamente.")


# CONSTANTES VITALES DE RESPUESTA SISTÉMICA
print("\n" + "=" * 70)
print("Constantes vitales de respuesta sistemica")
print("=" * 70)




variables_vitales = [
    "HR_limpia",
    "Temp_limpia",
    "Resp_limpia"
]

variables_vitales = [
    variable for variable in variables_vitales
    if variable in dataset.columns
]

for variable in variables_vitales:

    vital_pandas = dataset.select(
        "SepsisLabel",
        variable
    ).filter(
        F.col(variable).isNotNull()
    ).withColumn(
        "grupo_sepsis",
        F.when(
            F.col("SepsisLabel") == 1,
            F.lit("SepsisLabel = 1")
        ).otherwise(
            F.lit("SepsisLabel = 0")
        )
    ).toPandas()

    fig_vital = px.box(
        vital_pandas,
        x="grupo_sepsis",
        y=variable,
        title=variable + " por SepsisLabel",
        labels={
            "grupo_sepsis": "Grupo",
            variable: variable
        }
    )

    guardar_grafico_plotly(
        fig_vital,
        variable + "_por_sepsislabel.html"
    )
