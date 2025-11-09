from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def get_weather_schema() -> StructType:
    return StructType(
        [
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
            StructField("wind_gusts_10m", DoubleType(), True),
        ]
    )


def load_weather_data(spark: SparkSession, data_path: str) -> DataFrame:
    """
    Зчитує погодні дані з файлів CSV згідно з визначеною схемою.
    """
    schema = get_weather_schema()
    df = spark.read.csv(data_path, header=True, schema=schema)

    return df.withColumn("date", col("date").cast("timestamp"))
