import os
from typing import List

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

HUMIDITY_THRESHOLD = 80.0
PRESSURE_SPIKE_THRESHOLD = 1015.0
DRY_DAY_PRECIPITATION_THRESHOLD = 1.0

RESULTS_DIR = "/app/results"
MODEL_DIR = "/app/artifacts/lr_dry_days"


def run_all_classifications(df: DataFrame) -> None:
    print("\n=== КЛАСИФІКАЦІЙНІ МОДЕЛІ (Anastasiia) ===\n")
    os.makedirs(MODEL_DIR, exist_ok=True)
    feature_cols: List[str] = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
    ]
    labeled = df.withColumn(
        "label",
        (F.col("precipitation") < F.lit(DRY_DAY_PRECIPITATION_THRESHOLD)).cast("int"),
    ).dropna(subset=feature_cols)
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(labeled)
    labels = [r[0] for r in data.select("label").distinct().collect()]
    if len(labels) < 2:
        if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
            print("У підвибірці один клас — використовую збережену модель.")
            model = LogisticRegressionModel.load(MODEL_DIR)
            preds = model.transform(data)
            me = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction"
            )
            acc = me.setMetricName("accuracy").evaluate(preds)
            f1 = me.setMetricName("f1").evaluate(preds)
            prec = me.setMetricName("weightedPrecision").evaluate(preds)
            rec = me.setMetricName("weightedRecall").evaluate(preds)
            be = BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="rawPrediction"
            )
            auc_roc = be.setMetricName("areaUnderROC").evaluate(preds)
            auc_pr = be.setMetricName("areaUnderPR").evaluate(preds)
            print(
                f"Dry days (Chennai) → Acc: {acc:.3f} | F1: {f1:.3f} | P: {prec:.3f} | R: {rec:.3f}"
            )
            print(f"AUC-ROC: {auc_roc:.3f} | AUC-PR: {auc_pr:.3f}")
            cm = (
                preds.groupBy("label", "prediction")
                .count()
                .orderBy("label", "prediction")
            )
            print("Confusion matrix (label, prediction, count):")
            cm.show(truncate=False)
        else:
            print(
                "У підвибірці один клас і немає збереженої моделі — пропускаю ML-блок."
            )
        return
    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"Завантажую модель з {MODEL_DIR} ...")
        model = LogisticRegressionModel.load(MODEL_DIR)
        preds = model.transform(data)
        me = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction"
        )
        acc = me.setMetricName("accuracy").evaluate(preds)
        f1 = me.setMetricName("f1").evaluate(preds)
        prec = me.setMetricName("weightedPrecision").evaluate(preds)
        rec = me.setMetricName("weightedRecall").evaluate(preds)
        be = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction"
        )
        auc_roc = be.setMetricName("areaUnderROC").evaluate(preds)
        auc_pr = be.setMetricName("areaUnderPR").evaluate(preds)
        print(
            f"Dry days (Chennai) → Acc: {acc:.3f} | F1: {f1:.3f} | P: {prec:.3f} | R: {rec:.3f}"
        )
        print(f"AUC-ROC: {auc_roc:.3f} | AUC-PR: {auc_pr:.3f}")
        cm = preds.groupBy("label", "prediction").count().orderBy("label", "prediction")
        print("Confusion matrix (label, prediction, count):")
        cm.show(truncate=False)
        return
    train, test = data.randomSplit([0.8, 0.2], seed=42)

    model: LogisticRegressionModel = LogisticRegression(
        featuresCol="features", labelCol="label"
    ).fit(train)
    model.write().overwrite().save(MODEL_DIR)
    preds = model.transform(test)
    me = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    acc = me.setMetricName("accuracy").evaluate(preds)
    f1 = me.setMetricName("f1").evaluate(preds)
    prec = me.setMetricName("weightedPrecision").evaluate(preds)
    rec = me.setMetricName("weightedRecall").evaluate(preds)
    be = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction"
    )
    auc_roc = be.setMetricName("areaUnderROC").evaluate(preds)
    auc_pr = be.setMetricName("areaUnderPR").evaluate(preds)
    print(
        f"Dry days (Chennai) → Acc: {acc:.3f} | F1: {f1:.3f} | P: {prec:.3f} | R: {rec:.3f}"
    )
    print(f"AUC-ROC: {auc_roc:.3f} | AUC-PR: {auc_pr:.3f}")
    cm = preds.groupBy("label", "prediction").count().orderBy("label", "prediction")
    print("Confusion matrix (label, prediction, count):")
    cm.show(truncate=False)


class BusinessAnalytics:
    def __init__(self, df: DataFrame):
        self.df = df
        self.spark = df.sparkSession
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def _ensure_city(self) -> None:
        if "city" not in self.df.columns:
            d = self.df.withColumn("_src", F.input_file_name())
            d = d.withColumn(
                "city",
                F.regexp_extract(F.col("_src"), r"([^/\\]+?)(?:_[^/\\]*|)\.csv$", 1),
            ).drop("_src")
            self.df = d

    @staticmethod
    def _is_monsoon(col_date: Column) -> Column:
        m = F.month(col_date)
        return (m >= F.lit(6)) & (m <= F.lit(9))

    def _write_csv(self, df: DataFrame, name: str) -> None:
        out = f"{RESULTS_DIR}/{name}"
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(out)

    def run_all(self) -> None:
        self._ensure_city()
        df = self.df
        monsoon = df.filter(self._is_monsoon(F.col("date"))).withColumn(
            "year", F.year("date")
        )
        top_humid = (
            monsoon.groupBy("city", "year")
            .agg(F.avg("relative_humidity_2m").alias("avg_humidity"))
            .withColumn(
                "rn",
                F.row_number().over(
                    Window.partitionBy("city").orderBy(F.desc("avg_humidity"))
                ),
            )
            .filter(F.col("rn") == 1)
            .drop("rn")
            .orderBy("city")
        )
        print("\n[1] Найвологіший рік у мусон:")
        top_humid.show(truncate=False)
        if top_humid.head(1):
            self._write_csv(top_humid, "top_humid")
        delhi = df.filter(F.col("city") == "Delhi").withColumn(
            "is_spike",
            (F.col("pressure_msl") > F.lit(PRESSURE_SPIKE_THRESHOLD)).cast("int"),
        )
        spikes = (
            delhi.withColumn("ym", F.date_format("date", "yyyy-MM"))
            .groupBy("ym")
            .agg(
                F.avg("is_spike").alias("share_spike"),
                F.sum("is_spike").alias("spike_hours"),
            )
            .orderBy("ym")
        )
        print("\n[2] Спайки тиску в Delhi:")
        spikes.show(50, truncate=False)
        if spikes.head(1):
            self._write_csv(spikes, "delhi_pressure_spikes")
        cities8 = [r["city"] for r in df.select("city").distinct().limit(8).collect()]

        def daypart_col(h: Column) -> Column:
            return (
                F.when((h >= 9) & (h <= 18), F.lit("work"))
                .when((h >= 22) | (h <= 6), F.lit("night"))
                .otherwise(F.lit("other"))
            )

        wind_base = (
            df.filter(F.col("city").isin(cities8))
            .withColumn("hour", F.hour("date"))
            .withColumn("daypart", daypart_col(F.col("hour")))
        )
        wind_cmp = (
            wind_base.groupBy("city", "daypart")
            .agg(F.avg("wind_speed_100m").alias("avg_wind_100m"))
            .filter(F.col("daypart").isin("work", "night"))
            .orderBy("city", "daypart")
        )
        print("\n[3] Вітер 100м: робочі vs ніч (перші 8 міст):")
        wind_cmp.show(truncate=False)
        if wind_cmp.head(1):
            self._write_csv(wind_cmp, "wind_work_vs_night")
        bhopal = (
            df.filter(F.col("city") == "Bhopal")
            .withColumn("date_d", F.to_date("date"))
            .withColumn(
                "is_humid",
                (F.col("relative_humidity_2m") > F.lit(HUMIDITY_THRESHOLD)).cast("int"),
            )
        )
        bhopal_daily = (
            bhopal.groupBy("date_d")
            .agg(F.max("is_humid").alias("humid_day"))
            .orderBy("date_d")
        )
        w = Window.orderBy("date_d")
        grp = F.sum(F.when(bhopal_daily["humid_day"] == 0, 1).otherwise(0)).over(w)
        series = (
            bhopal_daily.withColumn("grp", grp)
            .groupBy("grp")
            .agg(
                F.sum("humid_day").alias("len_run"),
                F.min("date_d").alias("start"),
                F.max("date_d").alias("end"),
            )
            .filter(F.col("len_run") >= 3)
            .orderBy(F.desc("len_run"))
        )
        print("\n[4] Bhopal: серії вологості (>=3 дні):")
        series.show(50, truncate=False)
        if series.head(1):
            self._write_csv(series, "bhopal_humidity_runs")
        monsoon_cloud = (
            df.filter(self._is_monsoon(F.col("date")))
            .groupBy("city")
            .agg(F.avg("cloud_cover").alias("avg_cloud"))
            .orderBy("city")
        )
        print("\n[5] Хмарність у мусон (середня по містах):")
        monsoon_cloud.show(truncate=False)
        if monsoon_cloud.head(1):
            self._write_csv(monsoon_cloud, "monsoon_cloud")
        lat_map: List[tuple] = []
        if lat_map:
            lat_df = self.spark.createDataFrame(lat_map, ["city", "lat"])
            cloud_lat = monsoon_cloud.join(lat_df, "city", "inner").orderBy("lat")
            print(
                "Кореляція(lat, avg_cloud):",
                cloud_lat.select(F.corr("lat", "avg_cloud")).first()[0],
            )
        chn = (
            df.filter(F.col("city") == "Chennai")
            .withColumn("date_d", F.to_date("date"))
            .withColumn(
                "is_dry",
                (F.col("precipitation") < F.lit(DRY_DAY_PRECIPITATION_THRESHOLD)).cast(
                    "int"
                ),
            )
        )
        chn_daily = (
            chn.groupBy("date_d")
            .agg(F.min("is_dry").alias("dry_day"))
            .orderBy("date_d")
        )
        w2 = Window.orderBy("date_d")
        grp2 = F.sum(F.when(chn_daily["dry_day"] == 0, 1).otherwise(0)).over(w2)
        dry_runs = (
            chn_daily.withColumn("grp", grp2)
            .groupBy("grp")
            .agg(
                F.sum("dry_day").alias("len_run"),
                F.min("date_d").alias("start"),
                F.max("date_d").alias("end"),
            )
            .filter(F.col("len_run") >= 3)
            .orderBy(F.desc("len_run"))
        )
        print("\n[6] Chennai: сухі періоди (>=3 дні):")
        dry_runs.show(50, truncate=False)
        if dry_runs.head(1):
            self._write_csv(dry_runs, "chennai_dry_runs")
