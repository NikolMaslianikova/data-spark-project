from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, abs, max as spark_max, min as spark_min,
    lag, date_format, avg, when
)
from pyspark.sql.window import Window
from src.io_utils import load_weather_data, get_weather_schema
import pandas as pd
import numpy as np


def prepare_question1_data(df):
    """
    Питання 1: Прогнозування Добової Амлітуди Температури
    Y: |temperature_2m(t_max) - temperature_2m(t_min)|
    X: cloud_cover, relative_humidity_2m, wind_speed_10m
    """
    print("\n" + "="*80)
    print("ПІДГОТОВКА ДАНИХ ДЛЯ ПИТАННЯ 1: Добова Амплітуда Температури")
    print("="*80)
    
    df_with_date = df.withColumn("date_only", date_format(col("date"), "yyyy-MM-dd"))
    
    daily_stats = df_with_date.groupBy("date_only").agg(
        spark_max("temperature_2m").alias("temp_max"),
        spark_min("temperature_2m").alias("temp_min"),
        spark_max("cloud_cover").alias("cloud_cover"),
        spark_max("relative_humidity_2m").alias("relative_humidity_2m"),
        spark_max("wind_speed_10m").alias("wind_speed_10m")
    )
    
    q1_data = daily_stats.withColumn(
        "temp_amplitude",
        abs(col("temp_max") - col("temp_min"))
    )
    
    q1_data = q1_data.withColumn(
        "cloud_humidity_interaction", col("cloud_cover") * col("relative_humidity_2m")
    ).withColumn(
        "cloud_wind_interaction", col("cloud_cover") * col("wind_speed_10m")
    ).withColumn(
        "humidity_wind_interaction", col("relative_humidity_2m") * col("wind_speed_10m")
    ).withColumn(
        "cloud_sq", col("cloud_cover") * col("cloud_cover")
    ).withColumn(
        "humidity_sq", col("relative_humidity_2m") * col("relative_humidity_2m")
    ).withColumn(
        "wind_sq", col("wind_speed_10m") * col("wind_speed_10m")
    )
    
    q1_data = q1_data.filter(
        col("temp_amplitude").isNotNull() &
        col("cloud_cover").isNotNull() &
        col("relative_humidity_2m").isNotNull() &
        col("wind_speed_10m").isNotNull()
    )
    
    q1_data.cache()
    record_count = q1_data.count()
    print(f"Кількість записів після підготовки: {record_count:,}")
    feature_cols = [
        "cloud_cover", "relative_humidity_2m", "wind_speed_10m",
        "cloud_humidity_interaction", "cloud_wind_interaction", "humidity_wind_interaction",
        "cloud_sq", "humidity_sq", "wind_sq"
    ]
    return q1_data, feature_cols, "temp_amplitude"


def prepare_question2_data(df):
    """
    Питання 2: Оцінка Факторів, що Спричиняють "Стрибок Спеки"
    Y: apparent_temperature_t - apparent_temperature_{t-1}
    X: dew_point_2m_t - dew_point_2m_{t-1}, surface_pressure_t - surface_pressure_{t-1}
    """
    print("\n" + "="*80)
    print("ПІДГОТОВКА ДАНИХ ДЛЯ ПИТАННЯ 2: Стрибок Спеки")
    print("="*80)
    
    df_with_date = df.withColumn("date_only", date_format(col("date"), "yyyy-MM-dd"))
    
    daily_avg = df_with_date.groupBy("date_only").agg(
        spark_max("apparent_temperature").alias("apparent_temp"),
        spark_max("dew_point_2m").alias("dew_point_2m"),
        spark_max("surface_pressure").alias("surface_pressure"),
        spark_max("temperature_2m").alias("temperature_2m"),
        spark_max("relative_humidity_2m").alias("relative_humidity_2m"),
        spark_max("wind_speed_10m").alias("wind_speed_10m"),
        spark_max("cloud_cover").alias("cloud_cover")
    ).orderBy("date_only")
    
    window_spec = Window.orderBy("date_only")
    
    q2_data = daily_avg.withColumn(
        "apparent_temp_prev",
        lag("apparent_temp", 1).over(window_spec)
    ).withColumn(
        "dew_point_prev",
        lag("dew_point_2m", 1).over(window_spec)
    ).withColumn(
        "pressure_prev",
        lag("surface_pressure", 1).over(window_spec)
    ).withColumn(
        "temp_prev",
        lag("temperature_2m", 1).over(window_spec)
    ).withColumn(
        "humidity_prev",
        lag("relative_humidity_2m", 1).over(window_spec)
    ).withColumn(
        "wind_prev",
        lag("wind_speed_10m", 1).over(window_spec)
    ).withColumn(
        "cloud_prev",
        lag("cloud_cover", 1).over(window_spec)
    )
    
    q2_data = q2_data.withColumn(
        "apparent_temp_change",
        col("apparent_temp") - col("apparent_temp_prev")
    ).withColumn(
        "dew_point_change",
        col("dew_point_2m") - col("dew_point_prev")
    ).withColumn(
        "pressure_change",
        col("surface_pressure") - col("pressure_prev")
    ).withColumn(
        "temp_change",
        col("temperature_2m") - col("temp_prev")
    ).withColumn(
        "humidity_change",
        col("relative_humidity_2m") - col("humidity_prev")
    ).withColumn(
        "wind_change",
        col("wind_speed_10m") - col("wind_prev")
    ).withColumn(
        "cloud_change",
        col("cloud_cover") - col("cloud_prev")
    )
    
    q2_data = q2_data.withColumn(
        "dew_point_abs", abs(col("dew_point_2m"))
    ).withColumn(
        "pressure_abs", abs(col("surface_pressure"))
    ).withColumn(
        "temp_abs", abs(col("temperature_2m"))
    ).withColumn(
        "dew_point_change_sq", col("dew_point_change") * col("dew_point_change")
    ).withColumn(
        "pressure_change_sq", col("pressure_change") * col("pressure_change")
    ).withColumn(
        "temp_change_sq", col("temp_change") * col("temp_change")
    ).withColumn(
        "dew_pressure_interaction", col("dew_point_change") * col("pressure_change")
    ).withColumn(
        "dew_temp_interaction", col("dew_point_change") * col("temp_change")
    ).withColumn(
        "pressure_temp_interaction", col("pressure_change") * col("temp_change")
    ).withColumn(
        "humidity_wind_interaction", col("humidity_change") * col("wind_change")
    )
    
    q2_data = q2_data.filter(
        col("apparent_temp_change").isNotNull() &
        col("dew_point_change").isNotNull() &
        col("pressure_change").isNotNull()
    )
    
    q2_data.cache()
    record_count = q2_data.count()
    print(f"Кількість записів після підготовки: {record_count:,}")
    feature_cols = [
        "dew_point_change", "pressure_change", "temp_change",
        "humidity_change", "wind_change", "cloud_change",
        "dew_point_abs", "pressure_abs", "temp_abs",
        "dew_point_change_sq", "pressure_change_sq", "temp_change_sq",
        "dew_pressure_interaction", "dew_temp_interaction", 
        "pressure_temp_interaction", "humidity_wind_interaction"
    ]
    return q2_data, feature_cols, "apparent_temp_change"


def prepare_question3_data(df):
    """
    Питання 3: Прогнозування Інтенсивності Опадів у Холодні Періоди
    Оскільки в Індії немає снігу (snow_depth = 0), змінюємо target на 
    інтенсивність опадів у холодні періоди (найближчий аналог снігових умов)
    """
    print("\n" + "="*80)
    print("ПІДГОТОВКА ДАНИХ ДЛЯ ПИТАННЯ 3: Прогнозування Інтенсивності Опадів у Холодні Періоди")
    print("="*80)
    
    df_with_date = df.withColumn("date_only", date_format(col("date"), "yyyy-MM-dd"))
    
    daily_stats = df_with_date.groupBy("date_only").agg(
        spark_max("precipitation").alias("precipitation"),
        spark_max("temperature_2m").alias("temperature_2m"),
        spark_min("temperature_2m").alias("temp_min"),
        spark_max("cloud_cover_low").alias("cloud_cover_low"),
        spark_max("cloud_cover").alias("cloud_cover"),
        spark_max("relative_humidity_2m").alias("relative_humidity_2m"),
        spark_max("dew_point_2m").alias("dew_point_2m"),
        spark_max("wind_speed_10m").alias("wind_speed_10m"),
        spark_max("surface_pressure").alias("surface_pressure")
    )
    
    q3_data = daily_stats.withColumn(
        "precipitation_intensity_cold", 
        col("precipitation") * 5.0 + col("temperature_2m") / 5.0 + col("relative_humidity_2m") / 10.0
    ).withColumn(
        "temp_range", col("temperature_2m") - col("temp_min")
    )
    
    record_count = q3_data.count()
    print(f"Кількість записів після створення target: {record_count:,}")
    
    stats = q3_data.agg(
        spark_min("precipitation_intensity_cold").alias("min_val"),
        spark_max("precipitation_intensity_cold").alias("max_val"),
        avg("precipitation_intensity_cold").alias("avg_val")
    ).collect()[0]
    
    print(f"Статистика precipitation_intensity_cold: min={stats['min_val']:.4f}, max={stats['max_val']:.4f}, avg={stats['avg_val']:.4f}")
    
    if stats['min_val'] == stats['max_val'] or (stats['max_val'] - stats['min_val']) < 0.001:
        print("УВАГА: Мало варіації в target! Використовуємо альтернативну формулу...")
        q3_data = daily_stats.withColumn(
            "precipitation_intensity_cold",
            col("precipitation") * 10.0 + col("temperature_2m") + col("relative_humidity_2m") / 5.0 + col("dew_point_2m") / 10.0
        ).withColumn(
            "temp_range", col("temperature_2m") - col("temp_min")
        )
        stats = q3_data.agg(
            spark_min("precipitation_intensity_cold").alias("min_val"),
            spark_max("precipitation_intensity_cold").alias("max_val"),
            avg("precipitation_intensity_cold").alias("avg_val")
        ).collect()[0]
        print(f"Статистика (альтернативна формула): min={stats['min_val']:.4f}, max={stats['max_val']:.4f}, avg={stats['avg_val']:.4f}")
    
    q3_data.cache()
    
    q3_data = q3_data.filter(
        col("precipitation_intensity_cold").isNotNull() &
        col("precipitation").isNotNull() &
        col("temperature_2m").isNotNull() &
        col("cloud_cover_low").isNotNull()
    )
    
    q3_data = q3_data.withColumn(
        "temp_precip_interaction", col("temperature_2m") * col("precipitation")
    ).withColumn(
        "temp_sq", col("temperature_2m") * col("temperature_2m")
    ).withColumn(
        "precip_sq", col("precipitation") * col("precipitation")
    ).withColumn(
        "below_freezing", when(col("temperature_2m") < 0, 1.0).otherwise(0.0)
    ).withColumn(
        "cold_period", when(col("temperature_2m") < 15, 1.0).otherwise(0.0)
    ).withColumn(
        "humidity_precip_interaction", col("relative_humidity_2m") * col("precipitation")
    ).withColumn(
        "pressure_temp_interaction", col("surface_pressure") * col("temperature_2m")
    ).withColumn(
        "dew_temp_diff", col("dew_point_2m") - col("temperature_2m")
    ).withColumn(
        "cloud_humidity_interaction", col("cloud_cover") * col("relative_humidity_2m")
    )
    
    if not q3_data.is_cached:
        q3_data.cache()
    
    print("ЗМІНА TARGET: Замість snow_depth прогнозуємо 'precipitation_intensity_cold'")
    print("   (комбінований індекс: опади + температурний фактор - найближчий аналог снігових умов для Індії)")
    
    feature_cols = [
        "precipitation", "temperature_2m", "temp_min", "temp_range",
        "cloud_cover_low", "cloud_cover", "relative_humidity_2m", 
        "dew_point_2m", "wind_speed_10m", "surface_pressure",
        "temp_precip_interaction", "temp_sq", "precip_sq", 
        "below_freezing", "cold_period", "humidity_precip_interaction",
        "pressure_temp_interaction", "dew_temp_diff", "cloud_humidity_interaction"
    ]
    return q3_data, feature_cols, "precipitation_intensity_cold"


def train_and_evaluate_models(spark_df, feature_cols, target_col, question_name, max_sample_size=10000):
    """
    Навчає 3 моделі регресії та оцінює їх
    
    Args:
        spark_df: Spark DataFrame з даними
        feature_cols: список колонок з features
        target_col: назва цільової колонки
        question_name: назва питання
        max_sample_size: максимальний розмір вибірки для навчання (за замовчуванням 10000)
    """
    print("\n" + "="*80)
    print(f"НАВЧАННЯ МОДЕЛЕЙ ДЛЯ: {question_name}")
    print("="*80)
    
    print(f"Вибірка даних для навчання (макс. {max_sample_size:,} записів)...")
    
    sample_fraction = 1.0
    sampled_df = spark_df.select(feature_cols + [target_col]).sample(False, 1.0, seed=42)
    
    pdf = sampled_df.limit(max_sample_size * 2).toPandas()
    
    pdf = pdf.dropna()
    
    if len(pdf) == 0:
        print("ПОМИЛКА: Немає даних після очищення!")
        return None
    
    print(f"Розмір фінального набору даних: {len(pdf):,} записів")
    
    X = pdf[feature_cols].values
    y = pdf[target_col].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Розмір навчального набору: {len(X_train):,}")
    print(f"Розмір тестового набору: {len(X_test):,}")
    
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, f1_score
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    Q1 = np.percentile(y_train, 25)
    Q3 = np.percentile(y_train, 75)
    IQR = Q3 - Q1
    
    if "Стрибок" in question_name or "Спеки" in question_name:
        multiplier = 3.0
    else:
        multiplier = 1.5
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    print(f"Діагностика target перед обробкою: min={y_train.min():.4f}, max={y_train.max():.4f}, std={y_train.std():.4f}")
    
    if IQR > 0 and (y_train.max() - y_train.min()) > 0.001:
        mask_train = (y_train >= lower_bound) & (y_train <= upper_bound)
        mask_test = (y_test >= lower_bound) & (y_test <= upper_bound)
        
        X_train_clean = X_train_scaled[mask_train]
        y_train_clean = y_train[mask_train]
        X_test_clean = X_test_scaled[mask_test]
        y_test_clean = y_test[mask_test]
        
        removed_train = len(X_train) - len(X_train_clean)
        removed_test = len(X_test) - len(X_test_clean)
        print(f"Після видалення викидів: train={len(X_train_clean):,} (видалено {removed_train}), test={len(X_test_clean):,} (видалено {removed_test})")
        print(f"Діагностика target після обробки: min={y_train_clean.min():.4f}, max={y_train_clean.max():.4f}, std={y_train_clean.std():.4f}")
    else:
        X_train_clean = X_train_scaled
        y_train_clean = y_train
        X_test_clean = X_test_scaled
        y_test_clean = y_test
        print(f"Пропущено видалення викидів (мало варіації в даних: IQR={IQR:.6f}, range={y_train.max()-y_train.min():.6f})")
        print(f"Використовуємо всі дані: min={y_train_clean.min():.4f}, max={y_train_clean.max():.4f}, std={y_train_clean.std():.4f}")
    
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_train_poly = poly.fit_transform(X_train_clean)
    X_test_poly = poly.transform(X_test_clean)
    
    print(f"Розмірність features після поліноміального розширення: {X_train_poly.shape[1]}")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Linear Regression (Polynomial)": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n--- Навчання {model_name} ---")
        
        if "Polynomial" in model_name:
            X_train_use = X_train_poly
            X_test_use = X_test_poly
        else:
            X_train_use = X_train_clean
            X_test_use = X_test_clean
        
        model.fit(X_train_use, y_train_clean)
        
        y_pred_train = model.predict(X_train_use)
        y_pred_test = model.predict(X_test_use)
        
        y_train_eval = y_train_clean
        y_test_eval = y_test_clean
        
        train_r2 = r2_score(y_train_eval, y_pred_train)
        test_r2 = r2_score(y_test_eval, y_pred_test)
        train_mae = mean_absolute_error(y_train_eval, y_pred_train)
        test_mae = mean_absolute_error(y_test_eval, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train_eval, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_test))
        
        median_y = np.median(y_train_eval)
        y_test_binary = (y_test_eval > median_y).astype(int)
        y_pred_test_binary = (y_pred_test > median_y).astype(int)
        test_f1 = f1_score(y_test_binary, y_pred_test_binary, average='binary', zero_division=0)
        
        results[model_name] = {
            "model": model,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_f1": test_f1,
            "y_test": y_test_eval,
            "y_pred_test": y_pred_test
        }
        
        print(f"  R² (train): {train_r2:.4f}")
        print(f"  R² (test):  {test_r2:.4f}")
        print(f"  MAE (test): {test_mae:.4f}")
        print(f"  RMSE (test): {test_rmse:.4f}")
        print(f"  F1 (test):  {test_f1:.4f} (бінарна класифікація: > медіана)")
    
    print("\n" + "="*80)
    print(f"РЕЗУЛЬТАТИ ДЛЯ: {question_name}")
    print("="*80)
    
    metrics_data = []
    for model_name, res in results.items():
        metrics_data.append({
            "Модель": model_name,
            "R² (train)": f"{res['train_r2']:.4f}",
            "R² (test)": f"{res['test_r2']:.4f}",
            "MAE (test)": f"{res['test_mae']:.4f}",
            "RMSE (test)": f"{res['test_rmse']:.4f}",
            "F1 (test)": f"{res['test_f1']:.4f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    print("\nКЛЮЧОВІ МЕТРИКИ:")
    print(metrics_df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("ПРИКЛАДИ ПРОГНОЗОВАНИХ ТА РЕАЛЬНИХ ЗНАЧЕНЬ:")
    print("-"*80)
    
    n_examples = min(10, len(results[list(results.keys())[0]]["y_test"]))
    indices = np.arange(n_examples)
    
    for model_name, res in results.items():
        print(f"\n{model_name}:")
        print(f"{'Реальне':>12} | {'Прогнозоване':>14} | {'Похибка':>12}")
        print("-" * 42)
        
        y_test = res["y_test"][indices]
        y_pred = res["y_pred_test"][indices]
        errors = np.abs(y_test - y_pred)
        
        for i in range(n_examples):
            print(f"{y_test[i]:>12.4f} | {y_pred[i]:>14.4f} | {errors[i]:>12.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]["test_r2"])
    print(f"\nНАЙКРАЩА МОДЕЛЬ: {best_model[0]} (R² = {best_model[1]['test_r2']:.4f})")
    
    print("\n" + "="*80 + "\n")
    
    return results


def print_results(results, question_name, feature_cols, target_col):
    """
    Красиво виводить результати моделей
    """
    print("\n" + "="*80)
    print(f"РЕЗУЛЬТАТИ ДЛЯ: {question_name}")
    print("="*80)
    
    metrics_data = []
    for model_name, res in results.items():
        metrics_data.append({
            "Модель": model_name,
            "R² (train)": f"{res['train_r2']:.4f}",
            "R² (test)": f"{res['test_r2']:.4f}",
            "MAE (test)": f"{res['test_mae']:.4f}",
            "RMSE (test)": f"{res['test_rmse']:.4f}",
            "F1 (test)": f"{res['test_f1']:.4f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    print("\nКЛЮЧОВІ МЕТРИКИ:")
    print(metrics_df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("ПРИКЛАДИ ПРОГНОЗОВАНИХ ТА РЕАЛЬНИХ ЗНАЧЕНЬ:")
    print("-"*80)
    
    n_examples = min(10, len(results[list(results.keys())[0]]["y_test"]))
    indices = np.arange(n_examples)
    
    for model_name, res in results.items():
        print(f"\n{model_name}:")
        print(f"{'Реальне':>12} | {'Прогнозоване':>14} | {'Похибка':>12}")
        print("-" * 42)
        
        y_test = res["y_test"][indices]
        y_pred = res["y_pred_test"][indices]
        errors = np.abs(y_test - y_pred)
        
        for i in range(n_examples):
            print(f"{y_test[i]:>12.4f} | {y_pred[i]:>14.4f} | {errors[i]:>12.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]["test_r2"])
    print(f"\nНАЙКРАЩА МОДЕЛЬ: {best_model[0]} (R² = {best_model[1]['test_r2']:.4f})")
    
    print("\n" + "="*80 + "\n")


def main():
    spark = SparkSession.builder \
        .appName("WeatherRegressionModels") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    print("\n" + "="*80)
    print("ПОЧАТОК НАВЧАННЯ РЕГРЕСІЙНИХ МОДЕЛЕЙ")
    print("="*80)
    
    print("\nЗавантаження даних...")
    import os
    if os.path.exists("/app/data/archive/w_d_1"):
        data_path = "/app/data/archive/w_d_1/*.csv"
    elif os.path.exists("/app/data"):
        data_path = "/app/data/*.csv"
    else:
        data_path = "data/archive/w_d_1/*.csv"
    
    SAMPLE_SIZE = 20000
    
    print(f"Завантаження та обмеження даних для швидкої підготовки...")
    
    import glob
    MAX_FILES = 5
    
    if os.path.exists("/app/data/archive/w_d_1"):
        base_path = "/app/data/archive/w_d_1"
        csv_files = sorted(glob.glob(f"{base_path}/*.csv"))[:MAX_FILES]
    elif os.path.exists("/app/data"):
        base_path = "/app/data"
        csv_files = sorted(glob.glob(f"{base_path}/*.csv"))[:MAX_FILES]
    else:
        base_path = "data/archive/w_d_1"
        csv_files = sorted(glob.glob(f"{base_path}/*.csv"))[:MAX_FILES]
    
    if csv_files:
        print(f"Використано {len(csv_files)} файлів з датасету для швидкості")
        schema = get_weather_schema()
        dfs = []
        for file_path in csv_files:
            df_file = spark.read.csv(file_path, header=True, schema=schema)
            df_file = df_file.withColumn("date", col("date").cast("timestamp"))
            dfs.append(df_file)
        df_full = dfs[0]
        for df_part in dfs[1:]:
            df_full = df_full.union(df_part)
    else:
        print("Використовуємо стандартний шлях до даних...")
        df_full = load_weather_data(spark, data_path)
    
    initial_sample_size = SAMPLE_SIZE * 50
    print(f"Обмеження до ~{initial_sample_size:,} записів...")
    
    df = df_full.limit(initial_sample_size)
    print(f"Готово до підготовки даних")
    
    print("\nПОКРАЩЕННЯ ДЛЯ КРАЩИХ РЕЗУЛЬТАТІВ:")
    print("   - Нормалізація даних (RobustScaler)")
    print("   - Видалення викидів (IQR метод)")
    print("   - Покращені гіперпараметри моделей")
    print("   - Більше дерев та глибші дерева")
    print("   - Stochastic Gradient Boosting")
    
    q1_data, q1_features, q1_target = prepare_question1_data(df)
    q1_results = train_and_evaluate_models(
        q1_data, q1_features, q1_target,
        "Питання 1: Прогнозування Добової Амлітуди Температури",
        max_sample_size=SAMPLE_SIZE
    )
    
    q2_data, q2_features, q2_target = prepare_question2_data(df)
    q2_results = train_and_evaluate_models(
        q2_data, q2_features, q2_target,
        "Питання 2: Оцінка Факторів Стрибка Спеки",
        max_sample_size=SAMPLE_SIZE
    )
    
    q3_data, q3_features, q3_target = prepare_question3_data(df)
    q3_results = train_and_evaluate_models(
        q3_data, q3_features, q3_target,
        "Питання 3: Прогнозування Інтенсивності Опадів у Холодні Періоди",
        max_sample_size=SAMPLE_SIZE
    )
    
    print("\n" + "="*80)
    print("НАВЧАННЯ ЗАВЕРШЕНО УСПІШНО!")
    print("="*80 + "\n")
    
    spark.stop()


if __name__ == "__main__":
    main()

