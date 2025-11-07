from pyspark.sql import DataFrame, functions as F, Column
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def run_classification(
    df: DataFrame,
    label_expr: Column,
    feature_cols: list[str],
    name: str,
) -> float:
    """
    Запускає одну модель класифікації LogisticRegression.

    :param df: Вхідний DataFrame із погодними даними.
    :param label_expr: Вираз для створення цільової змінної (label).
    :param feature_cols: Список назв стовпців для ознак.
    :param name: Назва моделі/питання (для виводу).
    :return: Точність (accuracy) моделі.
    """
    df = df.withColumn("label", label_expr).dropna(subset=feature_cols)
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(df)
    train, test = data.randomSplit([0.8, 0.2], seed=42)

    model: LogisticRegressionModel = LogisticRegression(
        featuresCol="features", labelCol="label"
    ).fit(train)

    preds = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    acc: float = evaluator.evaluate(preds)
    print(f"{name} → Accuracy: {acc:.3f}")
    return acc


def run_all_classifications(df: DataFrame) -> None:
    """
    Запускає всі класифікаційні моделі для бізнес-питань.
    """
    print("\n=== КЛАСИФІКАЦІЙНІ МОДЕЛІ (Anastasiia) ===\n")

    run_classification(
        df,
        label_expr=(F.col("relative_humidity_2m") > 80).cast("int"),
        feature_cols=["temperature_2m", "pressure_msl", "wind_speed_10m"],
        name="High humidity days"
    )

    run_classification(
        df,
        label_expr=(F.col("pressure_msl") > 1015).cast("int"),
        feature_cols=["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        name="Pressure spikes"
    )

    run_classification(
        df,
        label_expr=(F.col("wind_speed_100m") > 20).cast("int"),
        feature_cols=["temperature_2m", "relative_humidity_2m", "pressure_msl"],
        name="High wind speed"
    )

    run_classification(
        df,
        label_expr=(F.col("relative_humidity_2m") > 80).cast("int"),
        feature_cols=["dew_point_2m", "cloud_cover", "wind_speed_10m"],
        name="Humidity series (Bhopal)"
    )

    run_classification(
        df,
        label_expr=(F.col("cloud_cover") > 70).cast("int"),
        feature_cols=["temperature_2m", "wind_speed_10m", "pressure_msl"],
        name="Cloud cover vs latitude"
    )

    run_classification(
        df,
        label_expr=(F.col("precipitation") < 1).cast("int"),
        feature_cols=["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        name="Dry days (Chennai)"
    )
