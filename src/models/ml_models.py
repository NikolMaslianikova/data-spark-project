# ml_models.py

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, hour, when

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def load_weather_data(spark):
    """
    Читаємо РІВНО 6 файлів з папки ../data відносно цього скрипта.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")

    files = [
        "Chidambaram.csv",
        "Arantangi.csv",
        "Allur.csv",
        "Abiramam.csv",
        "Avadi.csv",
        "Bommayapalaiyam.csv",
    ]
    paths = [os.path.join(data_dir, f) for f in files]

    print("Завантажую дані з файлів:")
    for p in paths:
        print("  ", p)

    df = (
        spark.read
        .csv(paths, header=True, inferSchema=True)
    )

    if "_c0" in df.columns:
        df = df.drop("_c0")

    print(f"Кількість рядків до dropna: {df.count()}")
    df = df.dropna().cache()
    print(f"Кількість рядків після dropna: {df.count()}")

    return df


def add_time_features(df):
    """
    Додаємо місяць і годину з поля date.
    """
    df = df.withColumn("month", month(col("date")))
    df = df.withColumn("hour", hour(col("date")))
    return df


# ========= LABEL-и ДЛЯ 3 КЛАСИФІКАЦІЙНИХ ПИТАНЬ =========

def add_label_comfort_day(df):
    """
    Питання 1: 'Комфортний день' чи ні.
    Умова (можна описати на слайді):
    - 20°C <= temperature_2m <= 28°C
    - 40% <= relative_humidity_2m <= 70%
    - precipitation < 0.1
    - wind_speed_10m < 8 м/с
    label_comfort_day = 1, інакше 0
    """
    return df.withColumn(
        "label_comfort_day",
        when(
            (col("temperature_2m") >= 20.0) &
            (col("temperature_2m") <= 28.0) &
            (col("relative_humidity_2m") >= 40.0) &
            (col("relative_humidity_2m") <= 70.0) &
            (col("precipitation") < 0.1) &
            (col("wind_speed_10m") < 8.0),
            1.0
        ).otherwise(0.0)
    )


def add_label_wet_day(df):
    """
    Питання 2: 'Мокрий день' (сильні опади) чи ні.
    Приклад порогу:
    - precipitation >= 5 мм за добу → 1
    - інакше → 0
    """
    return df.withColumn(
        "label_wet_day",
        when(col("precipitation") >= 5.0, 1.0).otherwise(0.0)
    )


def add_label_warm_night(df):
    """
    Питання 3: 'Тепла ніч' чи ні.
    Вважаємо ніч: hour ∈ [0, 6]
    Умова:
    - hour між 0 та 6 включно
    - temperature_2m >= 22°C
    Інше — 0.
    """
    return df.withColumn(
        "label_warm_night",
        when(
            (col("hour") >= 0) &
            (col("hour") <= 6) &
            (col("temperature_2m") >= 22.0),
            1.0
        ).otherwise(0.0)
    )


# ========= ОЦІНКА КЛАСИФІКАЦІЇ =========

def evaluate_classification(model_name, predictions, label_col="label"):
    """
    Accuracy, Precision, Recall, F1 (як просила).
    """
    acc_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1"
    )
    prec_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="weightedPrecision"
    )
    rec_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="weightedRecall"
    )

    acc = acc_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    prec = prec_eval.evaluate(predictions)
    rec = rec_eval.evaluate(predictions)

    print(f"{model_name}: accuracy = {acc:.4f}, precision = {prec:.4f}, "
          f"recall = {rec:.4f}, f1 = {f1:.4f}")


# ========= ЗАГАЛЬНА ФУНКЦІЯ ДЛЯ ЗАПУСКУ 3 КЛАСИФІКАТОРІВ =========

def run_classification_task(
    df,
    label_col: str,
    task_name: str,
    feature_cols: list,
):
    print("\n" + "=" * 80)
    print(f"=== КЛАСИФІКАЦІЙНА ЗАДАЧА: {task_name} (label = {label_col}) ===")
    print("Використані ознаки:", feature_cols)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    data = assembler.transform(df).select("features", label_col)

    # трохи підрізаємо датасет, щоб не вбити ноут випадково
    data = data.sample(withReplacement=False, fraction=0.3, seed=42)

    data = data.withColumnRenamed(label_col, "label")

    train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)
    print(f"Train rows: {train_df.count()}, Test rows: {test_df.count()}")

    models = [
        (
            "LogisticRegression",
            LogisticRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=20,
            )
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                numTrees=30,
                maxDepth=7,
                seed=42,
            )
        ),
        (
            "GBTClassifier",
            GBTClassifier(
                featuresCol="features",
                labelCol="label",
                maxIter=20,
                maxDepth=5,
                seed=42,
            )
        ),
    ]

    for model_name, model in models:
        print(f"\n--- Модель: {model_name} ---")
        fitted = model.fit(train_df)
        preds = fitted.transform(test_df)

        preds.select("label", "probability", "prediction").show(10)

        evaluate_classification(
            model_name=f"{task_name} / {model_name}",
            predictions=preds,
            label_col="label",
        )


def main():
    spark = (
        SparkSession.builder
        .appName("Weather_Classification_TamilNadu")
        .getOrCreate()
    )

    df = load_weather_data(spark)
    df = add_time_features(df)

    # ====== КЛАСИФІКАЦІЯ: 3 БІЗНЕС-ПИТАННЯ ======

    base_features = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "snow_depth",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "wind_direction_100m",
        "wind_gusts_10m",
        "month",
        "hour",
    ]

    # 1) Комфортний день
    df1 = add_label_comfort_day(df)
    run_classification_task(
        df=df1,
        label_col="label_comfort_day",
        task_name="Комфортний день (температура/вологість/вітер/без дощу)",
        feature_cols=base_features,
    )

    # 2) Мокрий день (сильні опади)
    df2 = add_label_wet_day(df1)
    run_classification_task(
        df=df2,
        label_col="label_wet_day",
        task_name="Мокрий день (сильні опади)",
        feature_cols=base_features,
    )

    # 3) Тепла ніч (ніч і температура не падає нижче порогу)
    df3 = add_label_warm_night(df2)
    run_classification_task(
        df=df3,
        label_col="label_warm_night",
        task_name="Тепла ніч (нічна температура не нижче 22°C)",
        feature_cols=base_features,
    )

    spark.stop()


if __name__ == "__main__":
    main()
