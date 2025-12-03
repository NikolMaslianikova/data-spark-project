from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = (
    SparkSession.builder
    .appName("Weather_Regression_3_Tasks")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


data_dir = "/Users/nikmas/PycharmProjects/BigData/data"

selected_files = [
    "Abiramam.csv",
    "Allur.csv",
    "Arantangi.csv",
    "Avadi.csv",
    "Bommayapalaiyam.csv",
    "Chidambaram.csv",
]
paths = [f"{data_dir}/{f}" for f in selected_files]

weather_schema = T.StructType([
    T.StructField("_c0", T.IntegerType(), True),
    T.StructField("date", T.TimestampType(), True),
    T.StructField("temperature_2m", T.DoubleType(), True),
    T.StructField("relative_humidity_2m", T.DoubleType(), True),
    T.StructField("dew_point_2m", T.DoubleType(), True),
    T.StructField("apparent_temperature", T.DoubleType(), True),
    T.StructField("precipitation", T.DoubleType(), True),
    T.StructField("rain", T.DoubleType(), True),
    T.StructField("snowfall", T.DoubleType(), True),
    T.StructField("snow_depth", T.DoubleType(), True),
    T.StructField("pressure_msl", T.DoubleType(), True),
    T.StructField("surface_pressure", T.DoubleType(), True),
    T.StructField("cloud_cover", T.DoubleType(), True),
    T.StructField("cloud_cover_low", T.DoubleType(), True),
    T.StructField("cloud_cover_mid", T.DoubleType(), True),
    T.StructField("cloud_cover_high", T.DoubleType(), True),
    T.StructField("wind_speed_10m", T.DoubleType(), True),
    T.StructField("wind_speed_100m", T.DoubleType(), True),
    T.StructField("wind_direction_10m", T.DoubleType(), True),
    T.StructField("wind_direction_100m", T.DoubleType(), True),
    T.StructField("wind_gusts_10m", T.DoubleType(), True),
])

df = (
    spark.read
    .option("header", True)
    .schema(weather_schema)
    .csv(paths)
)

print("Схема вхідних даних:")
df.printSchema()

df = df.dropna()
print("Кількість рядків після dropna:", df.count())

MAX_ROWS = 200_000
df = df.limit(MAX_ROWS)
print(f"Кількість рядків після limit({MAX_ROWS}):", df.count())

# ------------------ 3. Одна регресійна задача ------------------ #

def run_regression_task(target_col: str, task_name: str):
    print("\n" + "=" * 80)
    print(f"=== РЕГРЕСІЙНА ЗАДАЧА: {task_name} (target = {target_col}) ===")

    feature_cols = [
        c for c in df.columns
        if c not in ["_c0", "date", target_col]
    ]
    print("Використані ознаки:", feature_cols)

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_vec"
    )

    scaler = StandardScaler(
        inputCol="features_vec",
        outputCol="features",
        withStd=True,
        withMean=True
    )

    df_assembled = assembler.transform(df)
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled)

    reg_df = df_scaled.select(
        F.col(target_col).alias("label"),
        F.col("features")
    )

    train_full, test_reg = reg_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train FULL rows: {train_full.count()}, Test rows: {test_reg.count()}")

    TRAIN_FRACTION_FOR_TREES = 0.2
    train_small = train_full.sample(
        withReplacement=False,
        fraction=TRAIN_FRACTION_FOR_TREES,
        seed=42
    )
    print(f"Train SMALL rows (для RF/GBT): {train_small.count()}")

    evaluator_rmse = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="r2"
    )

    def train_and_eval_reg(model, name: str, use_small: bool = False):
        train_df = train_small if use_small else train_full
        print(f"\n--- Модель: {name} (train_rows = {train_df.count()}) ---")
        fitted = model.fit(train_df)
        preds = fitted.transform(test_reg)

        preds.select("label", "prediction").show(10)

        rmse = evaluator_rmse.evaluate(preds)
        r2 = evaluator_r2.evaluate(preds)

        print(f"{name} -> RMSE: {rmse:.4f}, R^2: {r2:.4f}")

    # 1) Linear Regression — на повному train
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=20,
        regParam=0.1,
        elasticNetParam=0.0
    )

    # 2) Random Forest — на підвибірці
    rf_reg = RandomForestRegressor(
        featuresCol="features",
        labelCol="label",
        numTrees=30,
        maxDepth=8,
        seed=42
    )

    # 3) GBT — теж на підвибірці
    gbt_reg = GBTRegressor(
        featuresCol="features",
        labelCol="label",
        maxIter=20,
        maxDepth=5,
        stepSize=0.1,
        seed=42
    )

    train_and_eval_reg(lr, "LinearRegression", use_small=False)
    train_and_eval_reg(rf_reg, "RandomForestRegressor", use_small=True)
    train_and_eval_reg(gbt_reg, "GBTRegressor", use_small=True)


# ------------------ 4. Три регресійні бізнес-питання ------------------ #

# 1) Прогноз реальної температури повітря
run_regression_task(
    target_col="temperature_2m",
    task_name="Прогноз температури повітря (temperature_2m)"
)

# 2) Прогноз “температури за відчуттями”
run_regression_task(
    target_col="apparent_temperature",
    task_name="Прогноз 'відчутної' температури (apparent_temperature)"
)

# 3) Прогноз швидкості вітру на 100 м
run_regression_task(
    target_col="wind_speed_100m",
    task_name="Прогноз швидкості вітру на 100 м (wind_speed_100m)"
)

print("\nГотово. Усі 3 регресійні задачі відпрацювали (з підвибіркою для RF/GBT).")

spark.stop()
