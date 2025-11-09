from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from src.io_utils import load_weather_data
import pandas as pd

if __name__ == "__main__":
    spark = SparkSession.builder.appName("WeatherInDocker").getOrCreate()

    print("\n=== ЕТАП ВИДОБУВАННЯ ДАНИХ ===\n")

    df = load_weather_data(spark, "./data/archive/w_d_1/*.csv")

    print("ЗАГАЛЬНА ІНФОРМАЦІЯ")
    print(f"Кількість рядків: {df.count():,}")
    print(f"Кількість колонок: {len(df.columns)}\n")

    print("СПИСОК УСІХ КОЛОНОК:")
    for i, c in enumerate(df.columns, start=1):
        print(f"{i:>2}. {c}")
    print()

    print("СХЕМА DataFrame:")
    df.printSchema()

    print("\nПЕРШІ 5 РЯДКІВ ДАНИХ:\n")
    df_safe = df.withColumn("date", df["date"].cast("string"))
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.precision", 3)
    pandas_df = df_safe.limit(5).toPandas()

    print("-" * 120)
    print(pandas_df.to_string(index=False))
    print("-" * 120)

    print("\n=== ОПИСОВА СТАТИСТИКА ЧИСЛОВИХ ПОЛІВ ===\n")
    numeric_columns = [
        field.name for field in df.schema.fields if isinstance(field.dataType, DoubleType)
    ]

    if numeric_columns:
        stats_df = df.select(numeric_columns).summary("count", "mean", "stddev", "min", "max")
        stats_pdf = stats_df.toPandas().set_index("summary").T

        print(stats_pdf.round(3).to_string())
        print()

        stats_numeric = stats_pdf.apply(pd.to_numeric, errors="coerce")
        counts = stats_numeric["count"].iloc[0] if "count" in stats_numeric else None
        mean_series = stats_numeric["mean"].sort_values(ascending=False)
        std_series = stats_numeric["stddev"].sort_values(ascending=False)
        min_series = stats_numeric["min"].sort_values(ascending=True)
        max_series = stats_numeric["max"].sort_values(ascending=False)

        print("АНАЛІЗ СТАТИСТИКИ:")
        if counts:
            print(f"- Кількість спостережень у числових колонках: {int(counts):,}")

        top_means = ", ".join(
            f"{col} ({value:.2f})" for col, value in mean_series.head(3).items()
        )
        print(f"- Топ-3 середні значення: {top_means}")

        top_std = ", ".join(
            f"{col} ({value:.2f})" for col, value in std_series.head(3).items()
        )
        print(f"- Найбільша варіативність (STD): {top_std}")

        min_values = ", ".join(
            f"{col} ({value:.2f})" for col, value in min_series.head(3).items()
        )
        print(f"- Найнижчі мінімальні значення: {min_values}")

        max_values = ", ".join(
            f"{col} ({value:.2f})" for col, value in max_series.head(3).items()
        )
        print(f"- Найвищі максимальні значення: {max_values}")

        zero_min_cols = [
            col for col, value in min_series.items() if pd.notna(value) and value == 0.0
        ]
        if zero_min_cols:
            print(f"- Нульові значення присутні у: {', '.join(zero_min_cols)}")
    else:
        print("У наборі даних немає числових стовпців для аналізу.")

    print("\nЗавантаження та аналіз даних успішно завершено.\n")

    spark.stop()
