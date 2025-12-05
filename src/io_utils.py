from pyspark.sql import SparkSession
from pyspark.sql.functions import col, input_file_name, regexp_extract
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType
)


def get_weather_schema():
    return StructType([
        StructField("empty", StringType(), True),
        StructField("date", TimestampType(), True),
        StructField("temperature_2m", DoubleType(), True),
        StructField("relative_humidity_2m", DoubleType(), True),
        StructField("dew_point_2m", DoubleType(), True),
        StructField("apparent_temperature", DoubleType(), True),
        StructField("precipitation", DoubleType(), True),
        StructField("rain", DoubleType(), True),
        StructField("snowfall", DoubleType(), True),
        StructField("snow_depth", DoubleType(), True),
        StructField("pressure_msl", DoubleType(), True),
        StructField("surface_pressure", DoubleType(), True),
        StructField("cloud_cover", DoubleType(), True),
        StructField("cloud_cover_low", DoubleType(), True),
        StructField("cloud_cover_mid", DoubleType(), True),
        StructField("cloud_cover_high", DoubleType(), True),
        StructField("wind_speed_10m", DoubleType(), True),
        StructField("wind_speed_100m", DoubleType(), True),
        StructField("wind_direction_10m", DoubleType(), True),
        StructField("wind_direction_100m", DoubleType(), True),
        StructField("wind_gusts_10m", DoubleType(), True)
    ])


def load_weather_data(spark: SparkSession, data_path: str, extract_city=True):
    """
    Зчитує погодні дані з файлів CSV згідно з визначеною схемою.
    
    Args:
        spark: SparkSession
        data_path: Шлях до CSV файлів (може бути glob pattern, наприклад "/app/data/*.csv")
        extract_city: Чи витягувати назву міста з імені файлу (за замовчуванням True)
    """
    schema = get_weather_schema()
    # Використовуємо PERMISSIVE mode для обробки помилок у даних
    df = spark.read.csv(data_path, header=True, schema=schema, mode="PERMISSIVE")

    df = df.withColumn("date", col("date").cast("timestamp"))
    
    # Витягуємо назву міста з імені файлу
    if extract_city:
        df = df.withColumn("input_file", input_file_name())
        df = df.withColumn(
            "city",
            regexp_extract(col("input_file"), r"([^/]+)\.csv$", 1)
        )
        df = df.drop("input_file")
    
    return df
