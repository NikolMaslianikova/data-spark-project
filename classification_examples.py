"""
Приклади використання окремих функцій класифікації
Для тестування та демонстрації окремих компонентів
"""

from pyspark.sql import SparkSession
from classification_models import (
    prepare_data_for_classification,
    train_binary_classification_models,
    train_multiclass_classification_models,
    hyperparameter_tuning_example
)
from src.io_utils import load_weather_data


def example_binary_classification_only():
    """
    Приклад запуску тільки бінарної класифікації
    """
    spark = SparkSession.builder.appName("BinaryClassificationExample").getOrCreate()
    
    # Завантаження даних
    df = load_weather_data(spark, "/app/data/*.csv", extract_city=True)
    prepared_df = prepare_data_for_classification(spark, df)
    
    # Тільки екстремальна спека
    results = train_binary_classification_models(spark, prepared_df, 'extreme_heat')
    
    print("\nНайкращі результати:")
    best_model = max(results.items(), key=lambda x: x[1]['F1-score'])
    print(f"Модель: {best_model[0]}")
    print(f"F1-score: {best_model[1]['F1-score']}")
    
    spark.stop()
    return results


def example_multiclass_classification_only():
    """
    Приклад запуску тільки мультикласової класифікації
    """
    spark = SparkSession.builder.appName("MulticlassClassificationExample").getOrCreate()
    
    # Завантаження даних
    df = load_weather_data(spark, "/app/data/*.csv", extract_city=True)
    prepared_df = prepare_data_for_classification(spark, df)
    
    # Тільки категорії температури
    results = train_multiclass_classification_models(spark, prepared_df, 'temp_category')
    
    print("\nНайкращі результати:")
    best_model = max(results.items(), key=lambda x: x[1]['F1-score'])
    print(f"Модель: {best_model[0]}")
    print(f"F1-score: {best_model[1]['F1-score']}")
    
    spark.stop()
    return results


def example_hyperparameter_tuning_only():
    """
    Приклад запуску тільки налаштування гіперпараметрів
    """
    spark = SparkSession.builder.appName("HyperparameterTuningExample").getOrCreate()
    
    # Завантаження даних
    df = load_weather_data(spark, "/app/data/*.csv", extract_city=True)
    prepared_df = prepare_data_for_classification(spark, df)
    
    # Налаштування для екстремальної спеки
    tuned_model = hyperparameter_tuning_example(spark, prepared_df, 'extreme_heat')
    
    spark.stop()
    return tuned_model


def example_custom_target_variable():
    """
    Приклад створення власної цільової змінної
    """
    from pyspark.sql.functions import col, when
    
    spark = SparkSession.builder.appName("CustomTargetExample").getOrCreate()
    
    # Завантаження даних
    df = load_weather_data(spark, "/app/data/*.csv", extract_city=True)
    
    # Створюємо власну цільову змінну: "комфортна погода"
    # Комфортна погода: температура 20-28°C, вологість <70%, без опадів
    df_custom = df.withColumn(
        "comfortable_weather",
        when(
            (col("temperature_2m").between(20, 28)) &
            (col("relative_humidity_2m") < 70) &
            (col("precipitation") == 0),
            1
        ).otherwise(0)
    )
    
    # Підготовка даних
    prepared_df = prepare_data_for_classification(spark, df_custom)
    
    # Навчання моделей
    results = train_binary_classification_models(spark, prepared_df, 'comfortable_weather')
    
    print("\nРезультати для 'комфортної погоди':")
    for model_name, metrics in results.items():
        print(f"{model_name}: F1-score = {metrics['F1-score']}")
    
    spark.stop()
    return results


def example_feature_importance_analysis():
    """
    Приклад аналізу важливості ознак для Random Forest
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import RandomForestClassifier
    from classification_models import create_feature_pipeline
    
    spark = SparkSession.builder.appName("FeatureImportanceExample").getOrCreate()
    
    # Завантаження та підготовка даних
    df = load_weather_data(spark, "/app/data/*.csv", extract_city=True)
    prepared_df = prepare_data_for_classification(spark, df)
    
    # Підготовка pipeline
    feature_pipeline = create_feature_pipeline()
    
    # Random Forest модель
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="extreme_heat",
        numTrees=100,
        seed=42
    )
    
    pipeline = Pipeline(stages=feature_pipeline.getStages() + [rf])
    
    # Навчання
    model = pipeline.fit(prepared_df)
    rf_model = model.stages[-1]
    
    # Важливість ознак
    feature_names = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "apparent_temperature", "precipitation", "pressure_msl",
        "surface_pressure", "cloud_cover", "wind_speed_10m",
        "wind_speed_100m", "wind_direction_10m", "wind_gusts_10m",
        "year", "month", "day", "is_monsoon",
        "temp_humidity_interaction", "pressure_wind_interaction"
    ]
    
    importance_scores = rf_model.featureImportances.toArray()
    
    print("\nВажливість ознак для прогнозування екстремальної спеки:")
    print("-" * 50)
    
    feature_importance = list(zip(feature_names, importance_scores))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for feature, importance in feature_importance[:10]:  # Топ-10
        print(f"{feature:<30} {importance:.4f}")
    
    spark.stop()
    return feature_importance


def example_prediction_on_new_data():
    """
    Приклад використання навченої моделі для прогнозування на нових даних
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import RandomForestClassifier
    from classification_models import create_feature_pipeline
    from pyspark.sql.functions import col, lit
    
    spark = SparkSession.builder.appName("PredictionExample").getOrCreate()
    
    # Завантаження та підготовка даних для навчання
    df = load_weather_data(spark, "/app/data/*.csv", extract_city=True)
    prepared_df = prepare_data_for_classification(spark, df)
    
    # Розділення на train/test
    train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
    
    # Навчання моделі
    feature_pipeline = create_feature_pipeline()
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="extreme_heat",
        numTrees=100,
        seed=42
    )
    
    pipeline = Pipeline(stages=feature_pipeline.getStages() + [rf])
    model = pipeline.fit(train_df)
    
    # Прогнозування на тестових даних
    predictions = model.transform(test_df)
    
    # Показуємо кілька прикладів прогнозів
    print("\nПриклади прогнозів:")
    print("-" * 80)
    print(f"{'Температура':<12} {'Вологість':<10} {'Опади':<8} {'Реальне':<8} {'Прогноз':<8} {'Ймовірність':<12}")
    print("-" * 80)
    
    sample_predictions = predictions.select(
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "extreme_heat", "prediction", "probability"
    ).limit(10).collect()
    
    for row in sample_predictions:
        prob_1 = row.probability[1] if len(row.probability) > 1 else 0.0
        print(f"{row.temperature_2m:<12.1f} {row.relative_humidity_2m:<10.1f} "
              f"{row.precipitation:<8.1f} {int(row.extreme_heat):<8} "
              f"{int(row.prediction):<8} {prob_1:<12.3f}")
    
    spark.stop()
    return model


if __name__ == "__main__":
    print("Виберіть приклад для запуску:")
    print("1. Тільки бінарна класифікація")
    print("2. Тільки мультикласова класифікація") 
    print("3. Тільки налаштування гіперпараметрів")
    print("4. Власна цільова змінна")
    print("5. Аналіз важливості ознак")
    print("6. Прогнозування на нових даних")
    
    # За замовчуванням запускаємо аналіз важливості ознак
    print("\nЗапускається аналіз важливості ознак...")
    example_feature_importance_analysis()
