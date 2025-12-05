"""
Бізнес-запити для аналізу погодних даних
Використання: window functions, group by, join, filters
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, count, countDistinct, avg, max, min, sum, year, month, dayofmonth,
    window, row_number, rank, dense_rank, lag, lead, stddev, 
    date_format, quarter, lit, abs, round, expr, filter, array
)
from pyspark.sql.window import Window


def query_1_rainy_days_monsoon(spark, df):
    """
    Запит 1: Які міста (A–H) мають найбільше дощових днів у сезон мусонів 
    (черв–вер), і кого варто включати до топ-10 пріоритетів для операцій?
    
    Використання: GROUP BY, FILTER, WINDOW FUNCTION (rank)
    """
    # Припускаємо, що є колонка 'city' (може бути додана з назви файлу або іншого джерела)
    # Фільтруємо місяці мусонів (червень-вересень = 6-9)
    monsoon_df = df.filter(
        (month(col("date")).between(6, 9)) &
        (col("precipitation") > 0) &
        (col("city").rlike("^[A-H]"))  # Міста A-H
    )
    
    # Групуємо по містах і рахуємо дощові дні
    rainy_days = monsoon_df.groupBy("city").agg(
        count("*").alias("rainy_days")
    )
    
    # Використовуємо window function для ранжування
    window_spec = Window.orderBy(col("rainy_days").desc())
    ranked = rainy_days.withColumn("rank", rank().over(window_spec))
    
    # Топ-10 міст
    top_10 = ranked.filter(col("rank") <= 10).orderBy(col("rank"))
    
    return top_10


def query_2_pressure_deviation(spark, df):
    """
    Запит 2: Наскільки середньомісячний тиск у Ahmedabad відхиляється від 
    Hyderabad у 2015–2024, і чи потрібні різні пороги попереджень для міст?
    
    Використання: GROUP BY, JOIN, FILTER, WINDOW FUNCTION
    """
    # Фільтруємо роки 2015-2024 та міста Ahmedabad і Hyderabad
    filtered_df = df.filter(
        (year(col("date")).between(2015, 2024)) &
        (col("city").isin(["Ahmedabad", "Hyderabad"])) &
        (col("pressure_msl").isNotNull())
    )
    
    # Групуємо по місту та місяцю, рахуємо середньомісячний тиск
    monthly_pressure = filtered_df.groupBy(
        "city", 
        year(col("date")).alias("year"),
        month(col("date")).alias("month")
    ).agg(
        avg("pressure_msl").alias("avg_pressure")
    )
    
    # Створюємо два окремі датафрейми для кожного міста
    ahmedabad = monthly_pressure.filter(col("city") == "Ahmedabad").select(
        col("year"), col("month"), col("avg_pressure").alias("ahmedabad_pressure")
    )
    
    hyderabad = monthly_pressure.filter(col("city") == "Hyderabad").select(
        col("year"), col("month"), col("avg_pressure").alias("hyderabad_pressure")
    )
    
    # JOIN для порівняння
    comparison = ahmedabad.join(
        hyderabad, 
        ["year", "month"], 
        "inner"
    ).withColumn(
        "pressure_deviation", 
        col("ahmedabad_pressure") - col("hyderabad_pressure")
    ).withColumn(
        "abs_deviation",
        abs(col("pressure_deviation"))
    )
    
    # Використовуємо window function для середнього відхилення
    window_spec = Window.partitionBy()
    result = comparison.withColumn(
        "avg_deviation", 
        avg("abs_deviation").over(window_spec)
    )
    
    return result.select(
        "year", "month", "ahmedabad_pressure", "hyderabad_pressure",
        "pressure_deviation", "abs_deviation", "avg_deviation"
    )


def query_3_extreme_heat_days(spark, df):
    """
    Запит 3: Як часто у містах (A–H) трапляються дні «екстремальної спеки» 
    (≥40 °C), і чи треба змінювати графіки персоналу/постачання?
    
    Використання: FILTER, GROUP BY, WINDOW FUNCTION
    """
    # Фільтруємо дні з температурою >= 40°C та міста A-H
    extreme_heat = df.filter(
        (col("temperature_2m") >= 40) &
        (col("city").rlike("^[A-H]"))
    )
    
    # Групуємо по містах та роках
    heat_by_city_year = extreme_heat.groupBy(
        "city",
        year(col("date")).alias("year")
    ).agg(
        count("*").alias("extreme_heat_days")
    )
    
    # Використовуємо window function для середньої кількості днів по містах
    window_spec = Window.partitionBy("city")
    result = heat_by_city_year.withColumn(
        "avg_heat_days_per_year",
        avg("extreme_heat_days").over(window_spec)
    ).withColumn(
        "max_heat_days",
        max("extreme_heat_days").over(window_spec)
    )
    
    # Сортуємо по середній кількості днів
    return result.orderBy(col("avg_heat_days_per_year").desc())


def query_4_temperature_peaks_hyderabad(spark, df):
    """
    Запит 4: Чи фіксуються у Hyderabad послідовні піки/провали 7-денного 
    ковзного середнього температури в квітні–червні, і як під це коригувати 
    тарифи/потужності?
    
    Використання: WINDOW FUNCTION (rolling average), FILTER, LAG/LEAD
    """
    # Фільтруємо Hyderabad та місяці квітень-червень (4-6)
    hyderabad_df = df.filter(
        (col("city") == "Hyderabad") &
        (month(col("date")).between(4, 6)) &
        (col("temperature_2m").isNotNull())
    ).select(
        "date", "temperature_2m"
    ).orderBy("date")
    
    # Використовуємо window function для 7-денного ковзного середнього
    window_spec = Window.orderBy("date").rowsBetween(-6, 0)
    rolling_avg = hyderabad_df.withColumn(
        "rolling_avg_7d",
        avg("temperature_2m").over(window_spec)
    )
    
    # Використовуємо LAG та LEAD для виявлення піків та провалів
    window_lag = Window.orderBy("date")
    peaks_valleys = rolling_avg.withColumn(
        "prev_avg", lag("rolling_avg_7d", 1).over(window_lag)
    ).withColumn(
        "next_avg", lead("rolling_avg_7d", 1).over(window_lag)
    ).withColumn(
        "is_peak",
        when(
            (col("rolling_avg_7d") > col("prev_avg")) &
            (col("rolling_avg_7d") > col("next_avg")),
            lit(1)
        ).otherwise(lit(0))
    ).withColumn(
        "is_valley",
        when(
            (col("rolling_avg_7d") < col("prev_avg")) &
            (col("rolling_avg_7d") < col("next_avg")),
            lit(1)
        ).otherwise(lit(0))
    )
    
    return peaks_valleys.filter(
        (col("is_peak") == 1) | (col("is_valley") == 1)
    )


def query_5_stormy_days_coastal_vs_inland(spark, df):
    """
    Запит 5: Чи відрізняється частота «штормових» днів у прибережних проти 
    внутрішніх міст (A–H), і як це врахувати в планах реагування?
    
    Використання: JOIN (з таблицею класифікації міст), GROUP BY, FILTER
    """
    # Припускаємо, що є таблиця з класифікацією міст (прибережні/внутрішні)
    # Створюємо датафрейм з класифікацією (в реальності може бути з іншого джерела)
    coastal_cities = ["A", "B", "C", "D"]  # Приклад
    inland_cities = ["E", "F", "G", "H"]   # Приклад
    
    # Визначаємо штормові дні (високі опади + сильний вітер)
    stormy_days = df.filter(
        (col("city").rlike("^[A-H]")) &
        (col("precipitation") > 10) &  # Високі опади
        (col("wind_speed_10m") > 15)   # Сильний вітер
    )
    
    # Додаємо класифікацію міста
    stormy_with_type = stormy_days.withColumn(
        "city_type",
        when(col("city").isin(coastal_cities), lit("coastal"))
        .otherwise(lit("inland"))
    )
    
    # Групуємо по типу міста та рахуємо штормові дні
    stormy_stats = stormy_with_type.groupBy("city_type").agg(
        count("*").alias("stormy_days"),
        countDistinct("city").alias("num_cities")
    ).withColumn(
        "avg_stormy_days_per_city",
        col("stormy_days") / col("num_cities")
    )
    
    return stormy_stats


def query_6_temperature_delta_from_state_avg(spark, df):
    """
    Запит 6: Чи є для міст (A–H) систематична дельта температури від 
    середнього по всіх містах A-H, і як її використати для таргетування інвестицій 
    у стійкість?
    
    Використання: WINDOW FUNCTION, GROUP BY, JOIN
    """
    # Фільтруємо міста A-H (якщо назви міст починаються з A-H)
    # Або можна використати конкретні міста, якщо вони відомі
    cities_ah = df.filter(
        (col("city").rlike("^[A-H]")) &
        (col("temperature_2m").isNotNull())
    )
    
    # Рахуємо середню температуру по всіх містах A-H (загальне середнє)
    overall_avg_temp = cities_ah.groupBy(
        year(col("date")).alias("year"),
        month(col("date")).alias("month")
    ).agg(
        avg("temperature_2m").alias("overall_avg_temp")
    )
    
    # Рахуємо середню температуру по кожному місті
    city_avg_temp = cities_ah.groupBy(
        "city",
        year(col("date")).alias("year"),
        month(col("date")).alias("month")
    ).agg(
        avg("temperature_2m").alias("city_avg_temp")
    )
    
    # JOIN для порівняння температури міста з загальним середнім
    comparison = city_avg_temp.join(
        overall_avg_temp,
        ["year", "month"],
        "inner"
    ).withColumn(
        "temperature_delta",
        col("city_avg_temp") - col("overall_avg_temp")
    )
    
    # Використовуємо window function для середньої дельти по містах
    window_spec = Window.partitionBy("city")
    result = comparison.withColumn(
        "avg_delta",
        avg("temperature_delta").over(window_spec)
    ).withColumn(
        "stddev_delta",
        stddev("temperature_delta").over(window_spec)
    )
    
    # Групуємо по містах для фінального результату
    city_delta_summary = result.groupBy("city").agg(
        avg("avg_delta").alias("systematic_delta"),
        avg("stddev_delta").alias("delta_volatility")
    ).orderBy(abs(col("systematic_delta")).desc())
    
    return city_delta_summary


# Функція для виконання всіх запитів
def run_all_queries(spark, df):
    """
    Виконує всі бізнес-запити та повертає результати
    """
    results = {}
    
    print("\n=== Запит 1: Дощові дні в сезон мусонів ===")
    results['query_1'] = query_1_rainy_days_monsoon(spark, df)
    results['query_1'].show(10)
    
    print("\n=== Запит 2: Відхилення тиску Ahmedabad vs Hyderabad ===")
    results['query_2'] = query_2_pressure_deviation(spark, df)
    results['query_2'].show(20)
    
    print("\n=== Запит 3: Дні екстремальної спеки ===")
    results['query_3'] = query_3_extreme_heat_days(spark, df)
    results['query_3'].show(20)
    
    print("\n=== Запит 4: Піки/провали температури в Hyderabad ===")
    results['query_4'] = query_4_temperature_peaks_hyderabad(spark, df)
    results['query_4'].show(20)
    
    print("\n=== Запит 5: Штормові дні: прибережні vs внутрішні ===")
    results['query_5'] = query_5_stormy_days_coastal_vs_inland(spark, df)
    results['query_5'].show()
    
    print("\n=== Запит 6: Дельта температури від середнього по штату ===")
    results['query_6'] = query_6_temperature_delta_from_state_avg(spark, df)
    results['query_6'].show(20)
    
    return results


if __name__ == "__main__":
    from src.io_utils import load_weather_data
    
    spark = SparkSession.builder.appName("BusinessQueries").getOrCreate()
    
    print("\n=== ЗАВАНТАЖЕННЯ ДАНИХ ===\n")
    # Завантажуємо дані (назва міста автоматично витягується з імені файлу)
    df = load_weather_data(spark, "/app/data/*.csv", extract_city=True)
    
    print(f"Завантажено {df.count():,} рядків")
    print(f"Унікальних міст: {df.select('city').distinct().count()}")
    
    # Примітка: Для запитів, що потребують колонку 'state', 
    # можна додати маппінг міст до штатів з окремого джерела
    # Наразі запити, що використовують 'state', можуть потребувати додаткових даних
    
    # Виконуємо запити
    results = run_all_queries(spark, df)
    
    spark.stop()