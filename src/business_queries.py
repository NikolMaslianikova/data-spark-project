# src/business_queries.py

from pyspark.sql import SparkSession, DataFrame


def register_weather_view(df: DataFrame) -> None:
    """
    Реєструє DataFrame як тимчасовий view 'weather' для SQL-запитів.
    Очікується, що в df вже є колонка `city`.
    """
    df.createOrReplaceTempView("weather")


SQL_QUERIES = {
    # 1. Тренд вологості по містах (filters + group by)
    "q1_trend_humidity_by_city": """
        SELECT
          city,
          year(date)  AS year,
          month(date) AS month,
          avg(relative_humidity_2m) AS avg_humidity
        FROM weather
        WHERE date BETWEEN '2010-01-01' AND '2024-12-31'
        GROUP BY city, year(date), month(date)
        ORDER BY city, year, month
    """,

    # 2. Сезонна структура опадів (filters + group by)
    "q2_seasonal_precipitation": """
        WITH weather_with_season AS (
          SELECT
            city,
            date,
            precipitation,
            CASE
              WHEN month(date) BETWEEN 6 AND 9 THEN 'monsoon'
              WHEN month(date) IN (10,11)      THEN 'post_monsoon'
              WHEN month(date) IN (12,1,2)     THEN 'winter'
              ELSE 'summer'
            END AS season
          FROM weather
        )
        SELECT
          city,
          season,
          avg(precipitation) AS avg_precip,
          sum(precipitation) AS total_precip
        FROM weather_with_season
        GROUP BY city, season
        ORDER BY city, season
    """,

    # 3. Комфортні дні (filters + group by)
    "q3_comfort_days": """
        WITH comfort_rows AS (
          SELECT
            city,
            to_date(date) AS day,
            CASE
              WHEN temperature_2m BETWEEN 20 AND 30
               AND relative_humidity_2m BETWEEN 30 AND 70
               AND precipitation < 1
               AND wind_speed_10m <= 20
              THEN 1 ELSE 0
            END AS is_comfort
          FROM weather
        ),
        daily_comfort AS (
          SELECT
            city,
            day,
            avg(is_comfort) AS comfort_ratio
          FROM comfort_rows
          GROUP BY city, day
        )
        SELECT
          city,
          count(*) AS comfort_days
        FROM daily_comfort
        WHERE comfort_ratio >= 0.7
        GROUP BY city
        ORDER BY comfort_days DESC
    """,

    # 4. Жаркі ночі (filters + group by)
    "q4_hot_nights": """
        WITH night_temps AS (
          SELECT
            city,
            to_date(date) AS day,
            min(temperature_2m) AS min_night_temp
          FROM weather
          WHERE hour(date) BETWEEN 0 AND 5
          GROUP BY city, to_date(date)
        )
        SELECT
          city,
          count(*) AS hot_nights
        FROM night_temps
        WHERE min_night_temp >= 30
        GROUP BY city
        ORDER BY hot_nights DESC
    """,

    # 5. Стрибки денних максимумів (group by + window + filters)
    "q5_temp_jumps": """
        WITH daily_t AS (
          SELECT
            city,
            to_date(date) AS day,
            max(temperature_2m) AS t_max
          FROM weather
          GROUP BY city, to_date(date)
        ),
        with_lag AS (
          SELECT
            city,
            day,
            t_max,
            lag(t_max) OVER (PARTITION BY city ORDER BY day) AS prev_t_max
          FROM daily_t
        )
        SELECT
          city,
          count(*) AS jump_days
        FROM with_lag
        WHERE prev_t_max IS NOT NULL
          AND abs(t_max - prev_t_max) >= 5
        GROUP BY city
        ORDER BY jump_days DESC
    """,

    # 6. Місяці з опадами > норми на 30% (group by + window + filters)
    "q6_months_precip_above_norm": """
        WITH monthly_precip AS (
          SELECT
            city,
            year(date)  AS year,
            month(date) AS month,
            sum(precipitation) AS monthly_precip
          FROM weather
          GROUP BY city, year(date), month(date)
        ),
        with_norm AS (
          SELECT
            city,
            year,
            month,
            monthly_precip,
            avg(monthly_precip) OVER (PARTITION BY city, month) AS month_norm
          FROM monthly_precip
        )
        SELECT
          city,
          year,
          month,
          monthly_precip,
          month_norm
        FROM with_norm
        WHERE monthly_precip > 1.3 * month_norm
        ORDER BY city, year, month
    """,

    # 7. Похмурі бездощові дні (filters + group by)
    "q7_cloudy_dry_days": """
        WITH daily_cloudy AS (
          SELECT
            city,
            to_date(date) AS day,
            avg(cloud_cover)   AS avg_cloud,
            sum(precipitation) AS total_precip
          FROM weather
          GROUP BY city, to_date(date)
        )
        SELECT
          city,
          count(*) AS cloudy_dry_days
        FROM daily_cloudy
        WHERE avg_cloud >= 70
          AND total_precip = 0
        GROUP BY city
        ORDER BY cloudy_dry_days DESC
    """,

    # 8. Мокрі тижні (window + filters)
    "q8_wet_weeks": """
        WITH daily_precip AS (
          SELECT
            city,
            to_date(date) AS day,
            sum(precipitation) AS daily_precip
          FROM weather
          GROUP BY city, to_date(date)
        ),
        rolling_7 AS (
          SELECT
            city,
            day,
            daily_precip,
            sum(daily_precip) OVER (
              PARTITION BY city
              ORDER BY day
              ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) AS precip_7d
          FROM daily_precip
        )
        SELECT
          city,
          count(*) AS wet_week_points
        FROM rolling_7
        WHERE precip_7d >= 50      -- поріг, можеш змінити
        GROUP BY city
        ORDER BY wet_week_points DESC
    """,

    # 9. Вітер для вітроенергетики (group by)
    "q9_wind_for_energy": """
        SELECT
          city,
          avg(wind_speed_100m)        AS avg_wind_100m,
          stddev_pop(wind_speed_100m) AS std_wind_100m
        FROM weather
        GROUP BY city
        ORDER BY avg_wind_100m DESC
    """,

    # 10. Дуже вологі дні / частка по місту (join між агрегатами)
    "q10_very_humid_share": """
        WITH humid_days AS (
          SELECT
            city,
            to_date(date) AS day
          FROM weather
          WHERE relative_humidity_2m >= 80
          GROUP BY city, to_date(date)
        ),
        total_days AS (
          SELECT
            city,
            count(DISTINCT to_date(date)) AS total_days
          FROM weather
          GROUP BY city
        ),
        humid_counts AS (
          SELECT
            city,
            count(*) AS very_humid_days
          FROM humid_days
          GROUP BY city
        )
        SELECT
          t.city,
          t.total_days,
          h.very_humid_days,
          (h.very_humid_days * 1.0) / t.total_days AS humid_share
        FROM total_days t
        JOIN humid_counts h
          ON t.city = h.city
        ORDER BY humid_share DESC
    """,

    # 11. JOIN між комфортними днями та жаркими ночами (join + filters + group by)
    "q11_comfort_vs_hot_nights": """
        WITH comfort_rows AS (
          SELECT
            city,
            to_date(date) AS day,
            CASE
              WHEN temperature_2m BETWEEN 20 AND 30
               AND relative_humidity_2m BETWEEN 30 AND 70
               AND precipitation < 1
               AND wind_speed_10m <= 20
              THEN 1 ELSE 0
            END AS is_comfort
          FROM weather
          WHERE year(date) BETWEEN 2019 AND 2021   -- 🔹 звузили по роках
        ),
        daily_comfort AS (
          SELECT
            city,
            day,
            avg(is_comfort) AS comfort_ratio
          FROM comfort_rows
          GROUP BY city, day
        ),
        hot_nights AS (
          SELECT
            city,
            to_date(date) AS day,
            min(temperature_2m) AS min_night_temp
          FROM weather
          WHERE hour(date) BETWEEN 0 AND 5
            AND year(date) BETWEEN 2019 AND 2021   -- 🔹 те саме обмеження
          GROUP BY city, to_date(date)
        )
        SELECT
          c.city,
          count(*) AS days_with_comfort_and_hot_night
        FROM daily_comfort c
        JOIN hot_nights h
          ON c.city = h.city AND c.day = h.day
        WHERE c.comfort_ratio >= 0.7
          AND h.min_night_temp >= 30
        GROUP BY c.city
        ORDER BY days_with_comfort_and_hot_night DESC
    """,


    # 12. Вітер: робочі години vs ніч (filters + group by)
    "q12_wind_day_vs_night": """
        WITH wind_by_period AS (
          SELECT
            city,
            CASE
              WHEN hour(date) BETWEEN 9 AND 18 THEN 'day_shift'
              ELSE 'night_shift'
            END AS period,
            wind_speed_10m
          FROM weather
        )
        SELECT
          city,
          period,
          avg(wind_speed_10m) AS avg_wind_10m
        FROM wind_by_period
        GROUP BY city, period
        ORDER BY city, period
    """
}


def run_query(spark: SparkSession, query_key: str, show_rows: int = 20) -> DataFrame:
    """
    Виконує один SQL-запит з SQL_QUERIES і повертає DataFrame.
    """
    if query_key not in SQL_QUERIES:
        raise ValueError(f"Невідомий ключ запиту: {query_key}")

    sql_text = SQL_QUERIES[query_key]
    df = spark.sql(sql_text)
    print(f"\n=== Результат для {query_key} ===")
    df.show(show_rows, truncate=False)
    return df


def run_all_queries(spark: SparkSession, show_rows: int = 20):
    """
    Виконує всі бізнес-запити по черзі.
    """
    for key in sorted(SQL_QUERIES.keys()):
        run_query(spark, key, show_rows=show_rows)
