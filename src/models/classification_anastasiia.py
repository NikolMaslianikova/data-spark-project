from pyspark.sql import DataFrame, functions as F, Column
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# === Константи порогів для бізнес-логіки ===
HUMIDITY_THRESHOLD = 80.0
PRESSURE_SPIKE_THRESHOLD = 1015.0
WIND_SPEED_AT_100_METERS_THRESHOLD = 20.0
CLOUD_COVER_THRESHOLD = 70.0
DRY_DAY_PRECIPITATION_THRESHOLD = 1.0


# === Базова утиліта для однієї ML-моделі (класифікація) ===
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
    Лишаємо лише ОДИН запуск ML-моделі, як домовлялися.
    """
    print("\n=== КЛАСИФІКАЦІЙНІ МОДЕЛІ (Anastasiia) ===\n")

    # Приклад: класифікація «сухих» годин/днів у Chennai (або загалом за даними)
    run_classification(
        df,
        label_expr=(F.col("precipitation") < DRY_DAY_PRECIPITATION_THRESHOLD).cast("int"),
        feature_cols=["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        name="Dry days (Chennai)",
    )


# === Клас для всіх 6 бізнес-відповідей без додаткових модулів/конфігів ===
class BusinessAnalytics:
    """
    Обчислює 6 бізнес-відповідей чистим Spark (агрегації/вікна).
    Публічний інтерфейс: BusinessAnalytics(df).run_all()
    """

    def __init__(self, df: DataFrame):
        self.df = df
        self.spark = df.sparkSession

    # ---- helpers (приватні) ----
    def _ensure_city(self) -> None:
        """
        Якщо у вхідному df немає колонки city — витягуємо її з імені вхідного файлу.
        Очікуємо патерн .../CityName_*.csv або .../CityName.csv
        """
        if "city" not in self.df.columns:
            d = self.df.withColumn("_src", F.input_file_name())
            d = d.withColumn(
                "city",
                F.regexp_extract(F.col("_src"), r"/([^/]+?)(?:_[^/]*|)\.csv$", 1),
            ).drop("_src")
            self.df = d

    @staticmethod
    def _is_monsoon(col_date: Column) -> Column:
        # Базово: червень–вересень
        m = F.month(col_date)
        return (m >= F.lit(6)) & (m <= F.lit(9))

    # ---- public ----
    def run_all(self) -> None:
        self._ensure_city()
        df = self.df

        # 1) Найвологіший рік у мусон (Guwahati, Gandhinagar)
        monsoon = df.filter(
            self._is_monsoon(F.col("date"))
            & F.col("city").isin("Guwahati", "Gandhinagar")
        )
        top_humid = (
            monsoon.withColumn("year", F.year("date"))
            .groupBy("city", "year")
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

        # 2) Спайки тиску (Delhi)
        delhi = df.filter(F.col("city") == "Delhi").withColumn(
            "is_spike", (F.col("pressure_msl") > F.lit(PRESSURE_SPIKE_THRESHOLD)).cast("int")
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
        spikes.show(24, truncate=False)

        # 3) Вітер 100м: робочі 9–18 vs ніч 22–6 (A–H)
        def _part_of_day(h: Column) -> Column:
            return (
                F.when((h >= 9) & (h <= 18), F.lit("work"))
                .when((h >= 22) | (h <= 6), F.lit("night"))
                .otherwise(F.lit("other"))
            )

        a_to_h = (
            df.filter(F.col("city").isin("A", "B", "C", "D", "E", "F", "G", "H"))
            .withColumn("hour", F.hour("date"))
            .withColumn("daypart", _part_of_day(F.col("hour")))
        )
        wind_cmp = (
            a_to_h.groupBy("city", "daypart")
            .agg(F.avg("wind_speed_100m").alias("avg_wind_100m"))
            .filter(F.col("daypart").isin("work", "night"))
            .orderBy("city", "daypart")
        )
        print("\n[3] Вітер 100м: робочі vs ніч (A–H):")
        wind_cmp.show(truncate=False)

        # 4) Bhopal: серії підвищеної вологості (>=3 дні)
        bhopal = (
            df.filter(F.col("city") == "Bhopal")
            .withColumn("date_d", F.to_date("date"))
            .withColumn(
                "is_humid", (F.col("relative_humidity_2m") > F.lit(HUMIDITY_THRESHOLD)).cast("int")
            )
        )
        # Переведемо годинні у добові (чи був хоч один "вологий" час у добі)
        bhopal_daily = (
            bhopal.groupBy("date_d")
            .agg(F.max("is_humid").alias("humid_day"))
            .orderBy("date_d")
        )
        w  = Window.orderBy("date_d")
        grp = F.sum(F.when(bhopal_daily["humid_day"]==0, 1).otherwise(0)).over(w)
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
        series.show(truncate=False)

        # 5) Широтна залежність хмарності у мусон (A–H)
        lat_map = [
            ("A", 10.0),
            ("B", 12.0),
            ("C", 15.0),
            ("D", 18.0),
            ("E", 20.0),
            ("F", 23.0),
            ("G", 26.0),
            ("H", 28.0),
        ]
        lat_df = self.spark.createDataFrame(lat_map, ["city", "lat"])
        cloud = (
            df.filter(
                self._is_monsoon(F.col("date"))
                & F.col("city").isin("A", "B", "C", "D", "E", "F", "G", "H")
            )
            .groupBy("city")
            .agg(F.avg("cloud_cover").alias("avg_cloud"))
            .join(lat_df, "city", "left")
            .orderBy("lat")
        )
        print("\n[5] Хмарність vs широта у мусон (A–H):")
        cloud.show(truncate=False)
        print("Кореляція(lat, avg_cloud):", cloud.select(F.corr("lat", "avg_cloud")).first()[0])

        # 6) Chennai: довгі «сухі» періоди (>=3 дні, precipitation < 1.0)
        chn = (
            df.filter(F.col("city") == "Chennai")
            .withColumn("date_d", F.to_date("date"))
            .withColumn(
                "is_dry", (F.col("precipitation") < F.lit(DRY_DAY_PRECIPITATION_THRESHOLD)).cast("int")
            )
        )
        # Добова умова «сухий день», якщо протягом доби не перевищено поріг
        chn_daily = (
            chn.groupBy("date_d").agg(F.min("is_dry").alias("dry_day")).orderBy("date_d")
        )
        w2 = Window.orderBy("date_d")
        grp2 = F.sum(F.when(chn_daily["dry_day"]==0, 1).otherwise(0)).over(w2)
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
        dry_runs.show(truncate=False)
