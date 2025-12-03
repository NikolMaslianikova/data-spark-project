from typing import Dict, List, Tuple

from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, functions as F


# Thresholds aligned with prior analytics in the project
DRY_DAY_PRECIPITATION_THRESHOLD = 1.0
HUMIDITY_THRESHOLD = 80.0
PRESSURE_SPIKE_THRESHOLD = 1015.0


def _train_eval_models(
    data: DataFrame,
    feature_cols: List[str],
    label_name: str,
    question_title: str,
) -> List[Tuple[str, float, float, float, float]]:
    """
    Helper: assemble features, split, train 3 classifiers (LR, RF, GBT),
    and return [(model_name, acc, prec, rec, f1)].
    Prints per-model metrics and a short per-question summary.
    """

    # Drop rows with missing values in features/label
    data = data.dropna(subset=feature_cols + [label_name])
    if not data.head(1):
        print(f"[Classification] '{question_title}': not enough data after preprocessing. Skipping.")
        return []

    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(data)

    # Check number of classes
    classes = [r[0] for r in data.select(label_name).distinct().collect()]
    if len(classes) < 2:
        print(f"[Classification] '{question_title}': only one class present — skipping.")
        return []

    train, test = data.randomSplit([0.8, 0.2], seed=42)
    if not train.head(1) or not test.head(1):
        print(f"[Classification] '{question_title}': empty train/test split — skipping.")
        return []

    algos: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(featuresCol="features", labelCol=label_name, maxIter=50),
        "RandomForest(100)": RandomForestClassifier(featuresCol="features", labelCol=label_name, numTrees=100, maxDepth=8),
        "GBT(64)": GBTClassifier(featuresCol="features", labelCol=label_name, maxIter=64, maxDepth=6, stepSize=0.1),
    }

    me = MulticlassClassificationEvaluator(labelCol=label_name, predictionCol="prediction")

    results: List[Tuple[str, float, float, float, float]] = []
    print(f"\n=== ПИТАННЯ: {question_title} ===")
    print(f"Ознаки: {', '.join(feature_cols)}")

    for name, algo in algos.items():
        model = algo.fit(train)
        preds = model.transform(test)

        acc = me.setMetricName("accuracy").evaluate(preds)
        f1 = me.setMetricName("f1").evaluate(preds)
        prec = me.setMetricName("weightedPrecision").evaluate(preds)
        rec = me.setMetricName("weightedRecall").evaluate(preds)

        results.append((name, acc, prec, rec, f1))
        print(f"[{name}] Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")

    if results:
        print("Підсумок (за F1):")
        for name, acc, prec, rec, f1 in sorted(results, key=lambda x: x[4], reverse=True):
            print(f"- {name}: F1={f1:.3f}, Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}")

    return results


def run_three_questions_three_models(df: DataFrame) -> None:
    """
    Train and evaluate THREE classifiers (LR, RF, GBT) for THREE different questions
    (total 9 models). Prints ONLY: Accuracy, Precision, Recall, F1 for each model.

    Questions:
      1) Dry vs not dry: label = (precipitation < 1.0)
      2) High humidity:  label = (relative_humidity_2m > 80.0)
      3) Pressure spike: label = (pressure_msl > 1015.0)
    """

    print("\n=== КЛАСИФІКАЦІЯ: 3 ПИТАННЯ × 3 МОДЕЛІ (усього 9) ===\n")

    # 1) Dry day question
    df_dry = df.withColumn(
        "label_dry",
        (F.col("precipitation") < F.lit(DRY_DAY_PRECIPITATION_THRESHOLD)).cast("int"),
    )
    features_dry: List[str] = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        # optionally add: "pressure_msl", "cloud_cover"
    ]
    _train_eval_models(df_dry, features_dry, "label_dry", "1) Сухо чи ні (dry vs not dry)")

    # 2) High humidity question
    df_hum = df.withColumn(
        "label_humid",
        (F.col("relative_humidity_2m") > F.lit(HUMIDITY_THRESHOLD)).cast("int"),
    )
    features_hum: List[str] = [
        "temperature_2m",
        "dew_point_2m",
        "cloud_cover",
        "wind_speed_10m",
    ]
    _train_eval_models(df_hum, features_hum, "label_humid", "2) Висока вологість (> 80%)")

    # 3) Pressure spike question
    df_prs = df.withColumn(
        "label_pressure_spike",
        (F.col("pressure_msl") > F.lit(PRESSURE_SPIKE_THRESHOLD)).cast("int"),
    )
    features_prs: List[str] = [
        "temperature_2m",
        "relative_humidity_2m",
        "cloud_cover",
        "wind_speed_10m",
    ]
    _train_eval_models(
        df_prs, features_prs, "label_pressure_spike", "3) Спайк тиску (pressure spike)"
    )
