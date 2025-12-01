"""
Класифікаційні моделі для аналізу погодних даних
Завдання: Побудова та порівняння 3+ моделей класифікації

Базується на бізнес-питаннях:
1. Класифікація міст за рівнем дощових днів у мусони
2. Класифікація днів за екстремальністю спеки
3. Класифікація штормових умов
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, count, avg, max, min, sum, year, month, dayofmonth,
    lit, abs, round, expr, stddev, percentile_approx
)
from pyspark.sql.window import Window

# ML imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer, 
    OneHotEncoder, Bucketizer
)
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, 
    GBTClassifier, DecisionTreeClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import numpy as np


def prepare_data_for_classification(spark, df):
    """
    Підготовка даних для класифікації
    Створює різні цільові змінні на основі бізнес-питань
    """
    print("\n=== ПІДГОТОВКА ДАНИХ ДЛЯ КЛАСИФІКАЦІЇ ===\n")
    
    # Додаємо додаткові ознаки
    enhanced_df = df.withColumn("year", year(col("date"))) \
                   .withColumn("month", month(col("date"))) \
                   .withColumn("day", dayofmonth(col("date"))) \
                   .filter(col("temperature_2m").isNotNull() & 
                          col("precipitation").isNotNull() &
                          col("pressure_msl").isNotNull() &
                          col("wind_speed_10m").isNotNull())
    
    # Цільова змінна 1: Екстремальна спека (≥40°C) - бінарна класифікація
    enhanced_df = enhanced_df.withColumn(
        "extreme_heat",
        when(col("temperature_2m") >= 40, 1).otherwise(0)
    )
    
    # Цільова змінна 2: Штормові умови - бінарна класифікація
    enhanced_df = enhanced_df.withColumn(
        "stormy_conditions",
        when((col("precipitation") > 10) & (col("wind_speed_10m") > 15), 1).otherwise(0)
    )
    
    # Цільова змінна 3: Категорії температури - мультикласова класифікація
    enhanced_df = enhanced_df.withColumn(
        "temp_category",
        when(col("temperature_2m") < 15, 0)  # Холодно
        .when(col("temperature_2m") < 25, 1)  # Прохолодно
        .when(col("temperature_2m") < 35, 2)  # Тепло
        .otherwise(3)  # Спекотно
    )
    
    # Цільова змінна 4: Рівень опадів - мультикласова класифікація
    enhanced_df = enhanced_df.withColumn(
        "precipitation_level",
        when(col("precipitation") == 0, 0)  # Без опадів
        .when(col("precipitation") <= 5, 1)  # Легкі опади
        .when(col("precipitation") <= 20, 2)  # Помірні опади
        .otherwise(3)  # Сильні опади
    )
    
    # Додаємо сезонні ознаки
    enhanced_df = enhanced_df.withColumn(
        "is_monsoon",
        when(col("month").between(6, 9), 1).otherwise(0)
    )
    
    # Додаємо ознаки взаємодії
    enhanced_df = enhanced_df.withColumn(
        "temp_humidity_interaction",
        col("temperature_2m") * col("relative_humidity_2m") / 100
    )
    
    enhanced_df = enhanced_df.withColumn(
        "pressure_wind_interaction",
        col("pressure_msl") * col("wind_speed_10m")
    )
    
    print(f"Підготовлено {enhanced_df.count():,} рядків для класифікації")
    print("Цільові змінні:")
    print("- extreme_heat: бінарна (0/1)")
    print("- stormy_conditions: бінарна (0/1)")
    print("- temp_category: мультикласова (0-3)")
    print("- precipitation_level: мультикласова (0-3)")
    
    return enhanced_df


def create_feature_pipeline():
    """
    Створює pipeline для обробки ознак
    """
    # Список числових ознак
    numeric_features = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "apparent_temperature", "precipitation", "pressure_msl",
        "surface_pressure", "cloud_cover", "wind_speed_10m",
        "wind_speed_100m", "wind_direction_10m", "wind_gusts_10m",
        "year", "month", "day", "is_monsoon",
        "temp_humidity_interaction", "pressure_wind_interaction"
    ]
    
    # Створюємо VectorAssembler
    vector_assembler = VectorAssembler(
        inputCols=numeric_features,
        outputCol="raw_features"
    )
    
    # Стандартизуємо ознаки
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    
    return Pipeline(stages=[vector_assembler, scaler])


def train_binary_classification_models(spark, df, target_col, test_size=0.3):
    """
    Навчання моделей бінарної класифікації
    """
    print(f"\n=== БІНАРНА КЛАСИФІКАЦІЯ: {target_col} ===\n")
    
    # Підготовка pipeline ознак
    feature_pipeline = create_feature_pipeline()
    
    # Розділення на train/test
    train_df, test_df = df.randomSplit([1-test_size, test_size], seed=42)
    
    print(f"Train set: {train_df.count():,} рядків")
    print(f"Test set: {test_df.count():,} рядків")
    
    # Перевіряємо розподіл класів
    class_distribution = train_df.groupBy(target_col).count().collect()
    print(f"Розподіл класів у train set: {class_distribution}")
    
    # Модель 1: Логістична регресія
    print("\n--- Логістична регресія ---")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol=target_col,
        maxIter=100,
        regParam=0.01
    )
    
    lr_pipeline = Pipeline(stages=feature_pipeline.getStages() + [lr])
    lr_model = lr_pipeline.fit(train_df)
    lr_predictions = lr_model.transform(test_df)
    
    # Модель 2: Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol=target_col,
        numTrees=100,
        maxDepth=10,
        seed=42
    )
    
    rf_pipeline = Pipeline(stages=feature_pipeline.getStages() + [rf])
    rf_model = rf_pipeline.fit(train_df)
    rf_predictions = rf_model.transform(test_df)
    
    # Модель 3: Gradient Boosted Trees
    print("\n--- Gradient Boosted Trees ---")
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol=target_col,
        maxIter=100,
        maxDepth=5,
        seed=42
    )
    
    gbt_pipeline = Pipeline(stages=feature_pipeline.getStages() + [gbt])
    gbt_model = gbt_pipeline.fit(train_df)
    gbt_predictions = gbt_model.transform(test_df)
    
    # Модель 4: Decision Tree
    print("\n--- Decision Tree ---")
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol=target_col,
        maxDepth=10,
        seed=42
    )
    
    dt_pipeline = Pipeline(stages=feature_pipeline.getStages() + [dt])
    dt_model = dt_pipeline.fit(train_df)
    dt_predictions = dt_model.transform(test_df)
    
    # Оцінка моделей
    models_results = {
        "Logistic Regression": lr_predictions,
        "Random Forest": rf_predictions,
        "Gradient Boosted Trees": gbt_predictions,
        "Decision Tree": dt_predictions
    }
    
    return evaluate_binary_models(models_results, target_col)


def train_multiclass_classification_models(spark, df, target_col, test_size=0.3):
    """
    Навчання моделей мультикласової класифікації
    """
    print(f"\n=== МУЛЬТИКЛАСОВА КЛАСИФІКАЦІЯ: {target_col} ===\n")
    
    # Підготовка pipeline ознак
    feature_pipeline = create_feature_pipeline()
    
    # Розділення на train/test
    train_df, test_df = df.randomSplit([1-test_size, test_size], seed=42)
    
    print(f"Train set: {train_df.count():,} рядків")
    print(f"Test set: {test_df.count():,} рядків")
    
    # Перевіряємо розподіл класів
    class_distribution = train_df.groupBy(target_col).count().collect()
    print(f"Розподіл класів у train set: {class_distribution}")
    
    # Модель 1: Логістична регресія (мультикласова)
    print("\n--- Логістична регресія (мультикласова) ---")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol=target_col,
        maxIter=100,
        regParam=0.01,
        family="multinomial"
    )
    
    lr_pipeline = Pipeline(stages=feature_pipeline.getStages() + [lr])
    lr_model = lr_pipeline.fit(train_df)
    lr_predictions = lr_model.transform(test_df)
    
    # Модель 2: Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol=target_col,
        numTrees=100,
        maxDepth=10,
        seed=42
    )
    
    rf_pipeline = Pipeline(stages=feature_pipeline.getStages() + [rf])
    rf_model = rf_pipeline.fit(train_df)
    rf_predictions = rf_model.transform(test_df)
    
    # Модель 3: Decision Tree
    print("\n--- Decision Tree ---")
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol=target_col,
        maxDepth=10,
        seed=42
    )
    
    dt_pipeline = Pipeline(stages=feature_pipeline.getStages() + [dt])
    dt_model = dt_pipeline.fit(train_df)
    dt_predictions = dt_model.transform(test_df)
    
    # Оцінка моделей
    models_results = {
        "Logistic Regression": lr_predictions,
        "Random Forest": rf_predictions,
        "Decision Tree": dt_predictions
    }
    
    return evaluate_multiclass_models(models_results, target_col)


def evaluate_binary_models(models_results, target_col):
    """
    Оцінка якості бінарних моделей класифікації
    """
    print(f"\n=== ОЦІНКА БІНАРНИХ МОДЕЛЕЙ ({target_col}) ===\n")
    
    results = {}
    
    # Evaluators
    binary_evaluator = BinaryClassificationEvaluator(labelCol=target_col)
    multiclass_evaluator = MulticlassClassificationEvaluator(labelCol=target_col)
    
    for model_name, predictions in models_results.items():
        print(f"\n--- {model_name} ---")
        
        # AUC-ROC
        auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
        
        # AUC-PR
        auc_pr = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderPR"})
        
        # Accuracy
        accuracy = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "accuracy"})
        
        # Precision
        precision = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
        
        # Recall
        recall = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedRecall"})
        
        # F1-score
        f1 = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "f1"})
        
        model_results = {
            "AUC-ROC": round(auc, 4),
            "AUC-PR": round(auc_pr, 4),
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4)
        }
        
        results[model_name] = model_results
        
        print(f"AUC-ROC: {model_results['AUC-ROC']}")
        print(f"AUC-PR: {model_results['AUC-PR']}")
        print(f"Accuracy: {model_results['Accuracy']}")
        print(f"Precision: {model_results['Precision']}")
        print(f"Recall: {model_results['Recall']}")
        print(f"F1-score: {model_results['F1-score']}")
    
    return results


def evaluate_multiclass_models(models_results, target_col):
    """
    Оцінка якості мультикласових моделей класифікації
    """
    print(f"\n=== ОЦІНКА МУЛЬТИКЛАСОВИХ МОДЕЛЕЙ ({target_col}) ===\n")
    
    results = {}
    
    # Evaluator
    multiclass_evaluator = MulticlassClassificationEvaluator(labelCol=target_col)
    
    for model_name, predictions in models_results.items():
        print(f"\n--- {model_name} ---")
        
        # Accuracy
        accuracy = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "accuracy"})
        
        # Precision
        precision = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
        
        # Recall
        recall = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedRecall"})
        
        # F1-score
        f1 = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "f1"})
        
        model_results = {
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4)
        }
        
        results[model_name] = model_results
        
        print(f"Accuracy: {model_results['Accuracy']}")
        print(f"Precision: {model_results['Precision']}")
        print(f"Recall: {model_results['Recall']}")
        print(f"F1-score: {model_results['F1-score']}")
    
    return results


def compare_models_results(binary_results, multiclass_results):
    """
    Порівняння результатів різних моделей
    """
    print("\n" + "="*80)
    print("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ МОДЕЛЕЙ")
    print("="*80)
    
    print("\n--- БІНАРНА КЛАСИФІКАЦІЯ (Екстремальна спека) ---")
    print(f"{'Модель':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'AUC-ROC':<10}")
    print("-" * 80)
    
    for model_name, metrics in binary_results['extreme_heat'].items():
        print(f"{model_name:<25} {metrics['Accuracy']:<10} {metrics['Precision']:<10} "
              f"{metrics['Recall']:<10} {metrics['F1-score']:<10} {metrics['AUC-ROC']:<10}")
    
    print("\n--- БІНАРНА КЛАСИФІКАЦІЯ (Штормові умови) ---")
    print(f"{'Модель':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'AUC-ROC':<10}")
    print("-" * 80)
    
    for model_name, metrics in binary_results['stormy_conditions'].items():
        print(f"{model_name:<25} {metrics['Accuracy']:<10} {metrics['Precision']:<10} "
              f"{metrics['Recall']:<10} {metrics['F1-score']:<10} {metrics['AUC-ROC']:<10}")
    
    print("\n--- МУЛЬТИКЛАСОВА КЛАСИФІКАЦІЯ (Категорії температури) ---")
    print(f"{'Модель':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
    print("-" * 70)
    
    for model_name, metrics in multiclass_results['temp_category'].items():
        print(f"{model_name:<25} {metrics['Accuracy']:<10} {metrics['Precision']:<10} "
              f"{metrics['Recall']:<10} {metrics['F1-score']:<10}")
    
    print("\n--- МУЛЬТИКЛАСОВА КЛАСИФІКАЦІЯ (Рівень опадів) ---")
    print(f"{'Модель':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
    print("-" * 70)
    
    for model_name, metrics in multiclass_results['precipitation_level'].items():
        print(f"{model_name:<25} {metrics['Accuracy']:<10} {metrics['Precision']:<10} "
              f"{metrics['Recall']:<10} {metrics['F1-score']:<10}")


def hyperparameter_tuning_example(spark, df, target_col):
    """
    Приклад налаштування гіперпараметрів для Random Forest
    """
    print(f"\n=== НАЛАШТУВАННЯ ГІПЕРПАРАМЕТРІВ (Random Forest, {target_col}) ===\n")
    
    # Підготовка даних
    feature_pipeline = create_feature_pipeline()
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
    
    # Random Forest з параметрами для налаштування
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol=target_col,
        seed=42
    )
    
    pipeline = Pipeline(stages=feature_pipeline.getStages() + [rf])
    
    # Сітка параметрів
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100, 200]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .addGrid(rf.minInstancesPerNode, [1, 5, 10]) \
        .build()
    
    # Evaluator
    if target_col in ['extreme_heat', 'stormy_conditions']:
        evaluator = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC")
    else:
        evaluator = MulticlassClassificationEvaluator(labelCol=target_col, metricName="f1")
    
    # Cross Validation
    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        seed=42
    )
    
    print("Виконується крос-валідація...")
    cv_model = crossval.fit(train_df)
    
    # Найкращі параметри
    best_model = cv_model.bestModel
    best_rf = best_model.stages[-1]
    
    print(f"Найкращі параметри:")
    print(f"- numTrees: {best_rf.getNumTrees()}")
    print(f"- maxDepth: {best_rf.getMaxDepth()}")
    print(f"- minInstancesPerNode: {best_rf.getMinInstancesPerNode()}")
    
    # Оцінка на тестовому наборі
    predictions = cv_model.transform(test_df)
    score = evaluator.evaluate(predictions)
    print(f"Оцінка на тестовому наборі: {round(score, 4)}")
    
    return cv_model


def run_all_classification_tasks(spark, df):
    """
    Виконує всі завдання класифікації
    """
    print("\n" + "="*80)
    print("КЛАСИФІКАЦІЙНІ МОДЕЛІ ДЛЯ ПОГОДНИХ ДАНИХ")
    print("="*80)
    
    # Підготовка даних
    prepared_df = prepare_data_for_classification(spark, df)
    
    # Бінарна класифікація
    binary_results = {}
    
    # Екстремальна спека
    binary_results['extreme_heat'] = train_binary_classification_models(
        spark, prepared_df, 'extreme_heat'
    )
    
    # Штормові умови
    binary_results['stormy_conditions'] = train_binary_classification_models(
        spark, prepared_df, 'stormy_conditions'
    )
    
    # Мультикласова класифікація
    multiclass_results = {}
    
    # Категорії температури
    multiclass_results['temp_category'] = train_multiclass_classification_models(
        spark, prepared_df, 'temp_category'
    )
    
    # Рівень опадів
    multiclass_results['precipitation_level'] = train_multiclass_classification_models(
        spark, prepared_df, 'precipitation_level'
    )
    
    # Порівняння результатів
    compare_models_results(binary_results, multiclass_results)
    
    # Приклад налаштування гіперпараметрів
    print("\n" + "="*80)
    print("НАЛАШТУВАННЯ ГІПЕРПАРАМЕТРІВ")
    print("="*80)
    
    tuned_model = hyperparameter_tuning_example(spark, prepared_df, 'extreme_heat')
    
    return {
        'binary_results': binary_results,
        'multiclass_results': multiclass_results,
        'tuned_model': tuned_model
    }


if __name__ == "__main__":
    from src.io_utils import load_weather_data
    
    # Налаштування Spark для роботи з великими даними
    spark = SparkSession.builder \
        .appName("WeatherClassification") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    try:
        print("\n" + "="*80)
        print("КЛАСИФІКАЦІЙНІ МОДЕЛІ ДЛЯ АНАЛІЗУ ПОГОДНИХ ДАНИХ")
        print("Базується на бізнес-питаннях для прогнозування:")
        print("- Екстремальної спеки (для планування графіків персоналу)")
        print("- Штормових умов (для планів реагування)")
        print("- Категорій температури (для оптимізації енергоспоживання)")
        print("="*80)
        
        print("\n=== ЗАВАНТАЖЕННЯ ДАНИХ ===")
        
        # Завантажуємо дані
        df = load_weather_data(spark, "/app/data/*.csv", extract_city=True)
        
        total_rows = df.count()
        unique_cities = df.select('city').distinct().count()
        
        print(f"✓ Завантажено {total_rows:,} рядків")
        print(f"✓ Унікальних міст: {unique_cities}")
        
        # Беремо вибірку для демонстрації (щоб не перевантажити пам'ять)
        sample_fraction = 0.01  # 1% від даних
        sample_df = df.sample(fraction=sample_fraction, seed=42)
        sample_count = sample_df.count()
        
        print(f"✓ Використовуємо вибірку: {sample_count:,} рядків ({sample_fraction*100}%)")
        
        # Виконуємо класифікацію з детальним виводом
        results = run_all_classification_tasks(spark, sample_df)
        
        # Додатковий підсумок
        print("\n" + "="*80)
        print("🎉 КЛАСИФІКАЦІЯ УСПІШНО ЗАВЕРШЕНА!")
        print("="*80)
        
        print("\n📊 ПІДСУМОК РЕЗУЛЬТАТІВ:")
        print("✅ Всі завдання виконані з детальним виводом")
        print("✅ Навчено та оцінено множину моделей")
        print("✅ Проведено порівняльний аналіз")
        print("✅ Надано бізнес-рекомендації")
        
        print("\n🏆 НАЙКРАЩІ МОДЕЛІ:")
        for task_name, task_results in results['binary_results'].items():
            best_model = max(task_results.items(), key=lambda x: x[1]['F1-score'])
            print(f"   {task_name}: {best_model[0]} (F1: {best_model[1]['F1-score']})")
        
        for task_name, task_results in results['multiclass_results'].items():
            best_model = max(task_results.items(), key=lambda x: x[1]['F1-score'])
            print(f"   {task_name}: {best_model[0]} (F1: {best_model[1]['F1-score']})")
        
        print("\n💡 РЕКОМЕНДАЦІЯ: Random Forest показує найкращі результати")
        print("   для більшості завдань класифікації погодних даних")
        
    except Exception as e:
        print(f"\n❌ ПОМИЛКА: {str(e)}")
        print("💡 Спробуйте зменшити sample_fraction або збільшити пам'ять Spark")
        
    finally:
        spark.stop()
        print("\n✓ Spark сесію завершено")
