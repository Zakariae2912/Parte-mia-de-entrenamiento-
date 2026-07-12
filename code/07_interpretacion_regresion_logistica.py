# ============================================================
# 05_analisis_sepsis.py
# MODELO PREDICTIVO DE SEPSIS A NIVEL PACIENTE
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array

# ------------------------------------------------------------
# SESIÓN SPARK
# ------------------------------------------------------------

spark = SparkSession.builder \
    .appName("Modelo_Sepsis_LOSO") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ------------------------------------------------------------
# CARGA DEL DATASET PREPROCESADO
# ------------------------------------------------------------

df = spark.read.parquet(
    "file:///home/adminp/TFA3/parquet/dataset_preprocesado"
)

print("Registros:", df.count())
print("Columnas:", len(df.columns))

# ------------------------------------------------------------
# ETIQUETA A NIVEL PACIENTE
# ------------------------------------------------------------

pacientes = df.groupBy(
    "hospital",
    "patient_id"
).agg(
    F.max("SepsisLabel").alias("y")
)

print("Pacientes totales:", pacientes.count())

pacientes.groupBy(
    "hospital",
    "y"
).count().orderBy(
    "hospital",
    "y"
).show()

# ------------------------------------------------------------
# CONJUNTO COMPLETO PARA EL ANÁLISIS DEFINITIVO
# ------------------------------------------------------------
# Se utilizan todos los pacientes disponibles.
# La matriz predictiva posterior excluirá a los pacientes que
# ya presenten SepsisLabel = 1 durante las primeras 24 horas.

sub = pacientes

print("Pacientes utilizados:", sub.count())

print("\nDistribución por hospital y clase:")

sub.groupBy(
    "hospital",
    "y"
).count().orderBy(
    "hospital",
    "y"
).show()

# ------------------------------------------------------------
# BLOQUE 2: MATRIZ DE FEATURES A NIVEL PACIENTE
# ------------------------------------------------------------
# Se utilizan únicamente las primeras 24 horas del ingreso.
# Los pacientes con SepsisLabel = 1 durante ese periodo se excluyen,
# porque el objetivo es predecir la aparición posterior de sepsis.

# 1) Seleccionar los registros de los pacientes de la submuestra

ids_sub = sub.select(
    "hospital",
    "patient_id"
)

df_sub = df.join(
    ids_sub,
    on=["hospital", "patient_id"],
    how="inner"
)

# 2) Identificar a los pacientes con sepsis durante las primeras 24 horas

sepsis_24h = df_sub.filter(
    F.col("ICULOS") <= 24
).groupBy(
    "hospital",
    "patient_id"
).agg(
    F.max("SepsisLabel").alias("sepsis_durante_24h")
)

# 3) Conservar únicamente pacientes sin sepsis durante la ventana

pacientes_validos = sepsis_24h.filter(
    F.col("sepsis_durante_24h") == 0
).select(
    "hospital",
    "patient_id"
)

print(
    "Pacientes excluidos por presentar sepsis durante las primeras 24 h:",
    sub.count() - pacientes_validos.count()
)

# 4) Seleccionar las primeras 24 horas de los pacientes válidos

df_24h = df_sub.join(
    pacientes_validos,
    on=["hospital", "patient_id"],
    how="inner"
).filter(
    F.col("ICULOS") <= 24
)

# 5) Variables clínicas longitudinales

variables_clinicas = [
    "HR",
    "MAP",
    "SBP",
    "Resp",
    "Temp",
    "O2Sat_combined",
    "FiO2",
    "Lactate"
]

# 6) Crear las agregaciones por paciente

agregaciones = []

for variable in variables_clinicas:

    agregaciones.append(
        F.avg(variable).alias(variable + "_media")
    )

    agregaciones.append(
        F.min(variable).alias(variable + "_min")
    )

    agregaciones.append(
        F.max(variable).alias(variable + "_max")
    )

    agregaciones.append(
        F.stddev(variable).alias(variable + "_desv_std")
    )

# Variables estáticas

agregaciones.append(
    F.max("Age").alias("Age")
)

agregaciones.append(
    F.max("Gender").alias("Gender")
)

# 7) Crear una fila por paciente

matriz = df_24h.groupBy(
    "hospital",
    "patient_id"
).agg(
    *agregaciones
)

# 8) Incorporar la variable objetivo

matriz = matriz.join(
    sub.select(
        "hospital",
        "patient_id",
        "y"
    ),
    on=["hospital", "patient_id"],
    how="inner"
)

# 9) Comprobaciones finales

print("Filas en la matriz:", matriz.count())
print("Columnas en la matriz:", len(matriz.columns))

matriz.groupBy(
    "hospital",
    "y"
).count().orderBy(
    "hospital",
    "y"
).show()

matriz.show(5, truncate=False)

# ------------------------------------------------------------
# BLOQUE 3: PREPARACIÓN PARA EL MODELO
# ------------------------------------------------------------
# Definimos qué columnas son las FEATURES (predictores) del modelo.
# Excluimos los identificadores (hospital, patient_id) y la etiqueta (y).
# Todas las demás columnas de la matriz son features.

columnas_no_features = ["hospital", "patient_id", "y"]

features = [c for c in matriz.columns if c not in columnas_no_features]

print("Número de features:", len(features))
print("Features:", features)



# ------------------------------------------------------------
# BLOQUE 4: VALIDACIÓN INTERCENTRO (LOSO)
# REGRESIÓN LOGÍSTICA
# ------------------------------------------------------------



    
def entrenar_y_validar(hospital_train, hospital_val):

    print("\n" + "=" * 55)
    print(
        "Entrenamiento:",
        hospital_train,
        "| Validación:",
        hospital_val
    )
    print("=" * 55)

    # --------------------------------------------------------
    # 1. SEPARACIÓN POR HOSPITAL
    # --------------------------------------------------------

    train = matriz.filter(
        F.col("hospital") == hospital_train
    ).withColumn(
        "y",
        F.col("y").cast("double")
    )

    val = matriz.filter(
        F.col("hospital") == hospital_val
    ).withColumn(
        "y",
        F.col("y").cast("double")
    )

    print("Pacientes de entrenamiento:", train.count())
    print("Pacientes de validación:", val.count())


    print("\nDistribución de clases en entrenamiento:")
    train.groupBy("y").count().orderBy("y").show()

    # --------------------------------------------------------
    # 2. ELIMINAR FEATURES COMPLETAMENTE VACÍAS EN TRAIN
    # --------------------------------------------------------
    # Si una variable está completamente vacía en el hospital
    # de entrenamiento, no puede calcularse su mediana.

    conteos_no_nulos = train.agg(
        *[
            F.count(F.col(columna)).alias(columna)
            for columna in features
        ]
    ).collect()[0].asDict()

    features_validas = [
        columna
        for columna in features
        if conteos_no_nulos[columna] > 0
    ]

    features_excluidas = [
        columna
        for columna in features
        if conteos_no_nulos[columna] == 0
    ]

    print("Features utilizadas:", len(features_validas))

    if len(features_excluidas) > 0:
        print(
            "Features excluidas por estar completamente vacías:",
            features_excluidas
        )

    # Nuevos nombres para las variables después de la imputación

    features_imputadas = [
        columna + "_imp"
        for columna in features_validas
    ]

    # --------------------------------------------------------
    # 3. PESOS PARA COMPENSAR EL DESBALANCEO
    # --------------------------------------------------------

    conteos_clases = train.groupBy(
        "y"
    ).count().collect()

    n_clase_0 = 0
    n_clase_1 = 0

    for fila in conteos_clases:

        if fila["y"] == 0:
            n_clase_0 = fila["count"]

        if fila["y"] == 1:
            n_clase_1 = fila["count"]

    total_train = n_clase_0 + n_clase_1

    if n_clase_0 == 0 or n_clase_1 == 0:

        raise ValueError(
            "El conjunto de entrenamiento debe contener "
            "pacientes de ambas clases."
        )

    peso_0 = total_train / (2 * n_clase_0)
    peso_1 = total_train / (2 * n_clase_1)

    train = train.withColumn(
        "peso_clase",
        F.when(
            F.col("y") == 0,
            F.lit(peso_0)
        ).otherwise(
            F.lit(peso_1)
        )
    )

    print("Peso clase 0:", round(peso_0, 4))
    print("Peso clase 1:", round(peso_1, 4))

    # --------------------------------------------------------
    # 4. IMPUTACIÓN POR MEDIANA
    # --------------------------------------------------------
    # El imputer se ajustará únicamente con train porque forma
    # parte del Pipeline que se entrena con pipeline.fit(train).

    imputer = Imputer(
        strategy="median",
        inputCols=features_validas,
        outputCols=features_imputadas
    )

    # --------------------------------------------------------
    # 5. CREACIÓN DEL VECTOR DE FEATURES
    # --------------------------------------------------------

    assembler = VectorAssembler(
        inputCols=features_imputadas,
        outputCol="features_vec",
        handleInvalid="error"
    )

    # --------------------------------------------------------
    # 6. ESTANDARIZACIÓN
    # --------------------------------------------------------

    scaler = StandardScaler(
        inputCol="features_vec",
        outputCol="features_scaled",
        withMean=True,
        withStd=True
    )

    # --------------------------------------------------------
    # 7. REGRESIÓN LOGÍSTICA
    # --------------------------------------------------------

    lr = LogisticRegression(
        featuresCol="features_scaled",
        labelCol="y",
        weightCol="peso_clase",
        maxIter=100
    )

    # --------------------------------------------------------
    # 8. PIPELINE
    # --------------------------------------------------------

    pipeline = Pipeline(
        stages=[
            imputer,
            assembler,
            scaler,
            lr
        ]
    )

    # Todos los pasos se ajustan exclusivamente con train

    modelo = pipeline.fit(train)

    # Aplicamos el modelo al hospital externo

    pred = modelo.transform(val)

    # --------------------------------------------------------
    # 9. AUROC
    # --------------------------------------------------------

    evaluador_roc = BinaryClassificationEvaluator(
        labelCol="y",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auroc = evaluador_roc.evaluate(pred)

    # --------------------------------------------------------
    # 10. AUPRC
    # --------------------------------------------------------

    evaluador_pr = BinaryClassificationEvaluator(
        labelCol="y",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    auprc = evaluador_pr.evaluate(pred)

    print(
        "Entrena",
        hospital_train,
        "-> Valida",
        hospital_val
    )

    print("AUROC:", round(auroc, 3))
    print("AUPRC:", round(auprc, 3))

    return modelo, pred, auroc, auprc


# ------------------------------------------------------------
# EJECUCIÓN DE LOS DOS PLIEGUES LOSO
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("VALIDACIÓN INTERCENTRO - REGRESIÓN LOGÍSTICA")
print("=" * 55)

modelo_AB, pred_AB, auroc_AB, auprc_AB = entrenar_y_validar(
    "A",
    "B"
)

modelo_BA, pred_BA, auroc_BA, auprc_BA = entrenar_y_validar(
    "B",
    "A"
)

# ------------------------------------------------------------
# RESUMEN
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("RESUMEN DE LA VALIDACIÓN INTERCENTRO")
print("=" * 55)

print("\nHospital A -> Hospital B")
print("AUROC:", round(auroc_AB, 3))
print("AUPRC:", round(auprc_AB, 3))

print("\nHospital B -> Hospital A")
print("AUROC:", round(auroc_BA, 3))
print("AUPRC:", round(auprc_BA, 3))

print("\nPromedio de ambos sentidos")

print(
    "AUROC medio:",
    round((auroc_AB + auroc_BA) / 2, 3)
)

print(
    "AUPRC medio:",
    round((auprc_AB + auprc_BA) / 2, 3)
)

# ------------------------------------------------------------
# BLOQUE 5: MATRIZ DE CONFUSIÓN Y MÉTRICAS CLÍNICAS
# Umbral predeterminado de la regresión logística = 0,5
# ------------------------------------------------------------

def evaluar_clasificacion(predicciones, nombre_validacion):

    print("\n" + "=" * 55)
    print("EVALUACIÓN:", nombre_validacion)
    print("=" * 55)

    # Contar cada combinación entre la etiqueta real y la predicción

    conteos = predicciones.groupBy(
        "y",
        "prediction"
    ).count().collect()

    verdaderos_negativos = 0
    falsos_positivos = 0
    falsos_negativos = 0
    verdaderos_positivos = 0

    for fila in conteos:

        valor_real = int(fila["y"])
        valor_predicho = int(fila["prediction"])
        n = fila["count"]

        if valor_real == 0 and valor_predicho == 0:
            verdaderos_negativos = n

        elif valor_real == 0 and valor_predicho == 1:
            falsos_positivos = n

        elif valor_real == 1 and valor_predicho == 0:
            falsos_negativos = n

        elif valor_real == 1 and valor_predicho == 1:
            verdaderos_positivos = n

    total = (
        verdaderos_negativos
        + falsos_positivos
        + falsos_negativos
        + verdaderos_positivos
    )

    # Métricas

    sensibilidad = (
        verdaderos_positivos
        / (verdaderos_positivos + falsos_negativos)
        if (verdaderos_positivos + falsos_negativos) > 0
        else 0
    )

    especificidad = (
        verdaderos_negativos
        / (verdaderos_negativos + falsos_positivos)
        if (verdaderos_negativos + falsos_positivos) > 0
        else 0
    )

    precision = (
        verdaderos_positivos
        / (verdaderos_positivos + falsos_positivos)
        if (verdaderos_positivos + falsos_positivos) > 0
        else 0
    )

    f1 = (
        2 * precision * sensibilidad
        / (precision + sensibilidad)
        if (precision + sensibilidad) > 0
        else 0
    )

    accuracy = (
        (verdaderos_positivos + verdaderos_negativos) / total
        if total > 0
        else 0
    )

    balanced_accuracy = (
        sensibilidad + especificidad
    ) / 2

    prevalencia = (
        (verdaderos_positivos + falsos_negativos) / total
        if total > 0
        else 0
    )

    # Resultados

    print("\nMatriz de confusión:")
    print("Verdaderos negativos:", verdaderos_negativos)
    print("Falsos positivos:", falsos_positivos)
    print("Falsos negativos:", falsos_negativos)
    print("Verdaderos positivos:", verdaderos_positivos)

    print("\nMétricas con umbral 0,5:")
    print("Prevalencia:", round(prevalencia, 3))
    print("Sensibilidad:", round(sensibilidad, 3))
    print("Especificidad:", round(especificidad, 3))
    print("Precisión:", round(precision, 3))
    print("F1:", round(f1, 3))
    print("Accuracy:", round(accuracy, 3))
    print("Balanced accuracy:", round(balanced_accuracy, 3))

    return {
        "VN": verdaderos_negativos,
        "FP": falsos_positivos,
        "FN": falsos_negativos,
        "VP": verdaderos_positivos,
        "prevalencia": prevalencia,
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy
    }


# Evaluación de ambos sentidos del LOSO

metricas_AB = evaluar_clasificacion(
    pred_AB,
    "Hospital A -> Hospital B"
)

metricas_BA = evaluar_clasificacion(
    pred_BA,
    "Hospital B -> Hospital A"
)

# ------------------------------------------------------------
# BLOQUE 6: COMPARACIÓN DE DIFERENTES UMBRALES
# ------------------------------------------------------------

def comparar_umbrales(predicciones, nombre_validacion):

    print("\n" + "=" * 70)
    print("COMPARACIÓN DE UMBRALES:", nombre_validacion)
    print("=" * 70)

    # Extraer la probabilidad estimada de sepsis
    pred_prob = predicciones.withColumn(
        "probabilidad_sepsis",
        vector_to_array(F.col("probability"))[1]
    )

    umbrales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    resultados = []

    for umbral in umbrales:

        pred_umbral = pred_prob.withColumn(
            "prediccion_umbral",
            F.when(
                F.col("probabilidad_sepsis") >= umbral,
                1
            ).otherwise(0)
        )

        conteos = pred_umbral.agg(

            F.sum(
                F.when(
                    (F.col("y") == 0) &
                    (F.col("prediccion_umbral") == 0),
                    1
                ).otherwise(0)
            ).alias("VN"),

            F.sum(
                F.when(
                    (F.col("y") == 0) &
                    (F.col("prediccion_umbral") == 1),
                    1
                ).otherwise(0)
            ).alias("FP"),

            F.sum(
                F.when(
                    (F.col("y") == 1) &
                    (F.col("prediccion_umbral") == 0),
                    1
                ).otherwise(0)
            ).alias("FN"),

            F.sum(
                F.when(
                    (F.col("y") == 1) &
                    (F.col("prediccion_umbral") == 1),
                    1
                ).otherwise(0)
            ).alias("VP")

        ).collect()[0]

        VN = conteos["VN"]
        FP = conteos["FP"]
        FN = conteos["FN"]
        VP = conteos["VP"]

        sensibilidad = (
            VP / (VP + FN)
            if (VP + FN) > 0 else 0
        )

        especificidad = (
            VN / (VN + FP)
            if (VN + FP) > 0 else 0
        )

        precision = (
            VP / (VP + FP)
            if (VP + FP) > 0 else 0
        )

        f1 = (
            2 * precision * sensibilidad
            / (precision + sensibilidad)
            if (precision + sensibilidad) > 0 else 0
        )

        balanced_accuracy = (
            sensibilidad + especificidad
        ) / 2

        resultados.append(
            (
                float(umbral),
                int(VN),
                int(FP),
                int(FN),
                int(VP),
                float(sensibilidad),
                float(especificidad),
                float(precision),
                float(f1),
                float(balanced_accuracy)
            )
        )

    tabla_umbrales = spark.createDataFrame(
        resultados,
        [
            "umbral",
            "VN",
            "FP",
            "FN",
            "VP",
            "sensibilidad",
            "especificidad",
            "precision",
            "F1",
            "balanced_accuracy"
        ]
    )

    tabla_umbrales.orderBy("umbral").show(
        truncate=False
    )

    return tabla_umbrales


tabla_umbrales_AB = comparar_umbrales(
    pred_AB,
    "Hospital A -> Hospital B"
)

tabla_umbrales_BA = comparar_umbrales(
    pred_BA,
    "Hospital B -> Hospital A"
)
