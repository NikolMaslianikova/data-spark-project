"""
Оптимізована версія класифікаційних моделей
Працює з обмеженим набором даних для демонстрації всіх завдань
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, count, avg, max, min, sum, year, month, dayofmonth,
    lit, abs, round, expr, stddev, percentile_approx
)
from pyspark.sql.window import Window

# ML imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, 
    GBTClassifier, DecisionTreeClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)

import time


def prepare_sample_data(spark, df, sample_fraction=0.01):
    """
    Підготовка вибірки даних для класифікації
    """
    print("\n" + "="*60)
    print("ЗАВДАННЯ 1: ПОПЕРЕДНЯ ОБРОБКА ДАНИХ")
    print("="*60)
    
    print("1.1. Вибірка даних для оптимізації...")
    # Беремо вибірку для швидкої демонстрації
    sampled_df = df.sample(fraction=sample_fraction, seed=42)
    
    print("1.2. Очищення даних...")
    # Фільтруємо тільки повні записи
    clean_df = sampled_df.filter(
        col("temperature_2m").isNotNull() & 
        col("precipitation").isNotNull() &
        col("pressure_msl").isNotNull() &
        col("wind_speed_10m").isNotNull() &
        col("relative_humidity_2m").isNotNull()
    )
    
    print("1.3. Створення нових ознак...")
    # Додаємо додаткові ознаки
    enhanced_df = clean_df.withColumn("year", year(col("date"))) \
                         .withColumn("month", month(col("date"))) \
                         .withColumn("day", dayofmonth(col("date")))
    
    # Сезонні ознаки
    enhanced_df = enhanced_df.withColumn(
        "is_monsoon",
        when(col("month").between(6, 9), 1).otherwise(0)
    )
    
    # Ознаки взаємодії
    enhanced_df = enhanced_df.withColumn(
        "temp_humidity_interaction",
        col("temperature_2m") * col("relative_humidity_2m") / 100
    )
    
    print("1.4. Створення цільових змінних...")
    
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
    
    final_count = enhanced_df.count()
    print(f"✅ ЗАВДАННЯ 1 ВИКОНАНО: Підготовлено {final_count:,} рядків")
    print(f"   - Очищено дані від пропусків")
    print(f"   - Створено {enhanced_df.columns.__len__()} ознак")
    print(f"   - Створено 3 цільові змінні")
    
    # Показуємо розподіл цільових змінних
    print("\n1.5. Розподіл цільових змінних:")
    extreme_heat_dist = enhanced_df.groupBy("extreme_heat").count().collect()
    stormy_dist = enhanced_df.groupBy("stormy_conditions").count().collect()
    temp_cat_dist = enhanced_df.groupBy("temp_category").count().collect()
    
    print(f"   Екстремальна спека: {extreme_heat_dist}")
    print(f"   Штормові умови: {stormy_dist}")
    print(f"   Категорії температури: {temp_cat_dist}")
    
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
        "year", "month", "day", "is_monsoon", "temp_humidity_interaction"
    ]
    
    # VectorAssembler
    vector_assembler = VectorAssembler(
        inputCols=numeric_features,
        outputCol="raw_features"
    )
    
    # StandardScaler
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    
    return Pipeline(stages=[vector_assembler, scaler])


def train_and_evaluate_models(spark, df, target_col, task_type="binary"):
    """
    Навчання та оцінка моделей
    """
    print(f"\n" + "="*60)
    print(f"ЗАВДАННЯ 2-4: НАВЧАННЯ ТА ОЦІНКА МОДЕЛЕЙ ({target_col})")
    print("="*60)
    
    print("2.1. Розділення даних на train/test...")
    # Розділення на train/test
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
    
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"   Train set: {train_count:,} рядків")
    print(f"   Test set: {test_count:,} рядків")
    
    # Підготовка pipeline ознак
    feature_pipeline = create_feature_pipeline()
    
    print("\n2.2. Вибір мінімум 3 різних моделей...")
    
    # Модель 1: Логістична регресія
    print("   - Логістична регресія")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol=target_col,
        maxIter=50,  # Зменшено для швидкості
        regParam=0.01
    )
    
    # Модель 2: Random Forest
    print("   - Random Forest")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol=target_col,
        numTrees=50,  # Зменшено для швидкості
        maxDepth=10,
        seed=42
    )
    
    # Модель 3: Decision Tree
    print("   - Decision Tree")
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol=target_col,
        maxDepth=10,
        seed=42
    )
    
    # Модель 4: Gradient Boosted Trees (тільки для бінарної класифікації)
    models = [
        ("Logistic Regression", lr),
        ("Random Forest", rf),
        ("Decision Tree", dt)
    ]
    
    if task_type == "binary":
        print("   - Gradient Boosted Trees")
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol=target_col,
            maxIter=20,  # Зменшено для швидкості
            maxDepth=5,
            seed=42
        )
        models.append(("Gradient Boosted Trees", gbt))
    
    print(f"✅ ЗАВДАННЯ 2 ВИКОНАНО: Обрано {len(models)} моделей")
    
    print("\n3.1. Навчання моделей...")
    trained_models = {}
    predictions = {}
    
    for model_name, model in models:
        print(f"   Навчання {model_name}...")
        start_time = time.time()
        
        # Створюємо pipeline
        pipeline = Pipeline(stages=feature_pipeline.getStages() + [model])
        
        # Навчаємо модель
        trained_model = pipeline.fit(train_df)
        
        # Робимо прогнози
        model_predictions = trained_model.transform(test_df)
        
        end_time = time.time()
        print(f"     Час навчання: {end_time - start_time:.2f} секунд")
        
        trained_models[model_name] = trained_model
        predictions[model_name] = model_predictions
    
    print(f"✅ ЗАВДАННЯ 3 ВИКОНАНО: Навчено {len(models)} моделей")
    
    print("\n4.1. Оцінка якості моделей...")
    results = evaluate_models(predictions, target_col, task_type)
    
    print(f"✅ ЗАВДАННЯ 4 ВИКОНАНО: Оцінено якість всіх моделей")
    
    return results


def evaluate_models(predictions, target_col, task_type):
    """
    Оцінка якості моделей з усіма метриками
    """
    print("\n" + "="*60)
    print("ЗАВДАННЯ 5: ОЦІНКА ЯКОСТІ МОДЕЛЕЙ")
    print("="*60)
    
    results = {}
    
    # Evaluators
    if task_type == "binary":
        binary_evaluator = BinaryClassificationEvaluator(labelCol=target_col)
    multiclass_evaluator = MulticlassClassificationEvaluator(labelCol=target_col)
    
    print("5.1. Розрахунок метрик для кожної моделі...")
    
    for model_name, model_predictions in predictions.items():
        print(f"\n   --- {model_name} ---")
        
        # Accuracy
        accuracy = multiclass_evaluator.evaluate(
            model_predictions, 
            {multiclass_evaluator.metricName: "accuracy"}
        )
        
        # Precision (weighted)
        precision = multiclass_evaluator.evaluate(
            model_predictions, 
            {multiclass_evaluator.metricName: "weightedPrecision"}
        )
        
        # Recall (weighted)
        recall = multiclass_evaluator.evaluate(
            model_predictions, 
            {multiclass_evaluator.metricName: "weightedRecall"}
        )
        
        # F1-score
        f1 = multiclass_evaluator.evaluate(
            model_predictions, 
            {multiclass_evaluator.metricName: "f1"}
        )
        
        model_results = {
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4)
        }
        
        # Додаткові метрики для бінарної класифікації
        if task_type == "binary":
            auc_roc = binary_evaluator.evaluate(
                model_predictions, 
                {binary_evaluator.metricName: "areaUnderROC"}
            )
            auc_pr = binary_evaluator.evaluate(
                model_predictions, 
                {binary_evaluator.metricName: "areaUnderPR"}
            )
            model_results["AUC-ROC"] = round(auc_roc, 4)
            model_results["AUC-PR"] = round(auc_pr, 4)
        
        results[model_name] = model_results
        
        # Виводимо результати
        print(f"     Accuracy:  {model_results['Accuracy']}")
        print(f"     Precision: {model_results['Precision']}")
        print(f"     Recall:    {model_results['Recall']}")
        print(f"     F1-score:  {model_results['F1-score']}")
        if task_type == "binary":
            print(f"     AUC-ROC:   {model_results['AUC-ROC']}")
            print(f"     AUC-PR:    {model_results['AUC-PR']}")
    
    print(f"\n✅ ЗАВДАННЯ 5 ВИКОНАНО: Розраховано всі метрики")
    print("   - Accuracy: загальна точність класифікації")
    print("   - Precision: точність позитивних прогнозів")
    print("   - Recall: повнота виявлення позитивних випадків")
    print("   - F1-score: гармонічне середнє Precision та Recall")
    if task_type == "binary":
        print("   - AUC-ROC: площа під ROC-кривою")
        print("   - AUC-PR: площа під Precision-Recall кривою")
    
    return results


def compare_algorithms(all_results):
    """
    Порівняння результатів різних алгоритмів
    """
    print("\n" + "="*80)
    print("ЗАВДАННЯ 6: ПОРІВНЯННЯ РЕЗУЛЬТАТІВ РІЗНИХ АЛГОРИТМІВ")
    print("="*80)
    
    for task_name, results in all_results.items():
        print(f"\n6.{list(all_results.keys()).index(task_name) + 1}. {task_name.upper()}")
        print("-" * 70)
        
        # Заголовок таблиці
        if "extreme_heat" in task_name or "stormy" in task_name:
            print(f"{'Модель':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'AUC-ROC':<10}")
        else:
            print(f"{'Модель':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
        print("-" * 70)
        
        # Сортуємо за F1-score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['F1-score'], reverse=True)
        
        for model_name, metrics in sorted_results:
            if "extreme_heat" in task_name or "stormy" in task_name:
                print(f"{model_name:<25} {metrics['Accuracy']:<10} {metrics['Precision']:<10} "
                      f"{metrics['Recall']:<10} {metrics['F1-score']:<10} {metrics.get('AUC-ROC', 'N/A'):<10}")
            else:
                print(f"{model_name:<25} {metrics['Accuracy']:<10} {metrics['Precision']:<10} "
                      f"{metrics['Recall']:<10} {metrics['F1-score']:<10}")
        
        # Визначаємо найкращу модель
        best_model = sorted_results[0]
        print(f"\n   🏆 НАЙКРАЩА МОДЕЛЬ: {best_model[0]} (F1-score: {best_model[1]['F1-score']})")
    
    print(f"\n✅ ЗАВДАННЯ 6 ВИКОНАНО: Порівняно результати всіх алгоритмів")
    print("   - Результати відсортовано за F1-score")
    print("   - Визначено найкращі моделі для кожної задачі")


def run_complete_classification(spark, df):
    """
    Виконує повний цикл класифікації з усіма завданнями
    """
    print("\n" + "="*80)
    print("ПОВНИЙ ЦИКЛ КЛАСИФІКАЦІЙНОГО АНАЛІЗУ ПОГОДНИХ ДАНИХ")
    print("="*80)
    print("Базується на бізнес-питаннях для прогнозування:")
    print("- Екстремальної спеки (для планування графіків персоналу)")
    print("- Штормових умов (для планів реагування)")
    print("- Категорій температури (для оптимізації енергоспоживання)")
    
    # ЗАВДАННЯ 1: Підготовка даних
    prepared_df = prepare_sample_data(spark, df, sample_fraction=0.005)  # 0.5% для демонстрації
    
    all_results = {}
    
    # ЗАВДАННЯ 2-6: Бінарна класифікація - Екстремальна спека
    print(f"\n{'='*80}")
    print("БІНАРНА КЛАСИФІКАЦІЯ: ЕКСТРЕМАЛЬНА СПЕКА")
    print(f"{'='*80}")
    results_heat = train_and_evaluate_models(spark, prepared_df, 'extreme_heat', 'binary')
    all_results['extreme_heat'] = results_heat
    
    # ЗАВДАННЯ 2-6: Бінарна класифікація - Штормові умови
    print(f"\n{'='*80}")
    print("БІНАРНА КЛАСИФІКАЦІЯ: ШТОРМОВІ УМОВИ")
    print(f"{'='*80}")
    results_storm = train_and_evaluate_models(spark, prepared_df, 'stormy_conditions', 'binary')
    all_results['stormy_conditions'] = results_storm
    
    # ЗАВДАННЯ 2-6: Мультикласова класифікація - Категорії температури
    print(f"\n{'='*80}")
    print("МУЛЬТИКЛАСОВА КЛАСИФІКАЦІЯ: КАТЕГОРІЇ ТЕМПЕРАТУРИ")
    print(f"{'='*80}")
    results_temp = train_and_evaluate_models(spark, prepared_df, 'temp_category', 'multiclass')
    all_results['temp_category'] = results_temp
    
    # ЗАВДАННЯ 6: Порівняння всіх результатів
    compare_algorithms(all_results)
    
    print("\n" + "="*80)
    print("🎉 ВСІ ЗАВДАННЯ УСПІШНО ВИКОНАНІ!")
    print("="*80)
    print("✅ 1. Попередня обробка даних - ВИКОНАНО")
    print("✅ 2. Вибір мінімум 3 різних моделей - ВИКОНАНО")
    print("✅ 3. Навчання моделей - ВИКОНАНО")
    print("✅ 4. Аналіз процесу навчання - ВИКОНАНО")
    print("✅ 5. Оцінка якості (Accuracy, Precision, Recall, F1-score) - ВИКОНАНО")
    print("✅ 6. Порівняння результатів різних алгоритмів - ВИКОНАНО")
    print("\nРезультати показують ефективність різних алгоритмів")
    print("для вирішення бізнес-задач на основі погодних даних.")
    
    return all_results


if __name__ == "__main__":
    from src.io_utils import load_weather_data
    
    # Налаштування Spark для оптимізації пам'яті
    spark = SparkSession.builder \
        .appName("WeatherClassificationOptimized") \
        .config("spark.driver.memory", "2g") \
        .config("spark.driver.maxResultSize", "1g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    print("Завантаження даних...")
    df = load_weather_data(spark, "w_d_1/*.csv", extract_city=True)
    
    print(f"Загальна кількість рядків: {df.count():,}")
    
    # Виконуємо повну класифікацію
    results = run_complete_classification(spark, df)
    
    spark.stop()
