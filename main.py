import pandas as pd
from pyspark.sql import SparkSession

from src.io_utils import load_weather_data
from src.models.classification_anastasiia import run_all_classifications

if __name__ == "__main__":
    spark = SparkSession.builder.appName("WeatherInDocker").getOrCreate()

    print()
    print("=== ЕТАП ВИДОБУВАННЯ ДАНИХ ===")

    df = load_weather_data(spark, "/app/data/*.csv")

    print("ЗАГАЛЬНА ІНФОРМАЦІЯ")
    print(f"Кількість рядків: {df.count():,}")
    print(f"Кількість колонок: {len(df.columns)}")

    print("СПИСОК УСІХ КОЛОНОК:")
    for i, c in enumerate(df.columns, start=1):
        print(f"{i:>2}. {c}")
    print()

    print("СХЕМА DataFrame:")
    df.printSchema()

    print("ПЕРШІ 5 РЯДКІВ ДАНИХ:")
    df_safe = df.withColumn("date", df["date"].cast("string"))
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.precision", 3)
    pandas_df = df_safe.limit(5).toPandas()

    print("-" * 120)
    print(pandas_df.to_string(index=False))
    print("-" * 120)

    print("Завантаження даних успішно завершено.")

    run_all_classifications(df)

    spark.stop()
