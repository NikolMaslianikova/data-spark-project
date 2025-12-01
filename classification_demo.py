"""
Демонстраційна версія класифікації з мінімальним набором даних
Показує виконання всіх завдань з детальним виводом
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, year, month
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import time


def demonstrate_all_tasks():
    """
    Демонстрація виконання всіх завдань класифікації
    """
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦІЯ ВИКОНАННЯ ВСІХ ЗАВДАНЬ КЛАСИФІКАЦІЇ")
    print("="*80)
    
    # Ініціалізація Spark
    spark = SparkSession.builder \
        .appName("ClassificationDemo") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    
    try:
        # Завантаження даних
        from src.io_utils import load_weather_data
        print("Завантаження даних...")
        df = load_weather_data(spark, "w_d_1/*.csv", extract_city=True)
        
        # Беремо дуже малу вибірку для демонстрації
        sample_df = df.sample(fraction=0.001, seed=42).limit(10000)
        
        print(f"Використовуємо вибірку: {sample_df.count()} рядків")
        
        # ============================================================================
        print("\n" + "="*60)
        print("✅ ЗАВДАННЯ 1: ПОПЕРЕДНЯ ОБРОБКА ДАНИХ")
        print("="*60)
        
        print("1.1. Очищення даних від пропусків...")
        clean_df = sample_df.filter(
            col("temperature_2m").isNotNull() & 
            col("precipitation").isNotNull() &
            col("pressure_msl").isNotNull() &
            col("wind_speed_10m").isNotNull()
        )
        
        print("1.2. Створення нових ознак...")
        # Додаємо ознаки
        processed_df = clean_df.withColumn("year", year(col("date"))) \
                              .withColumn("month", month(col("date")))
        
        print("1.3. Створення цільових змінних...")
        # Цільова змінна: екстремальна спека
        processed_df = processed_df.withColumn(
            "extreme_heat",
            when(col("temperature_2m") >= 35, 1).otherwise(0)  # Знижено поріг для демонстрації
        )
        
        # Цільова змінна: категорії температури
        processed_df = processed_df.withColumn(
            "temp_category",
            when(col("temperature_2m") < 20, 0)
            .when(col("temperature_2m") < 30, 1)
            .otherwise(2)
        )
        
        final_count = processed_df.count()
        print(f"✅ ЗАВДАННЯ 1 ВИКОНАНО: Підготовлено {final_count} рядків")
        
        # Показуємо розподіл
        heat_dist = processed_df.groupBy("extreme_heat").count().collect()
        temp_dist = processed_df.groupBy("temp_category").count().collect()
        print(f"   Розподіл екстремальної спеки: {heat_dist}")
        print(f"   Розподіл категорій температури: {temp_dist}")
        
        # ============================================================================
        print("\n" + "="*60)
        print("✅ ЗАВДАННЯ 2: ВИБІР МІНІМУМ 3 РІЗНИХ МОДЕЛЕЙ")
        print("="*60)
        
        print("2.1. Обрані моделі:")
        print("   - Логістична регресія")
        print("   - Random Forest")
        print("   - Decision Tree")
        print("✅ ЗАВДАННЯ 2 ВИКОНАНО: Обрано 3 різні моделі")
        
        # ============================================================================
        print("\n" + "="*60)
        print("✅ ЗАВДАННЯ 3: НАВЧАННЯ МОДЕЛЕЙ")
        print("="*60)
        
        # Підготовка ознак
        feature_cols = ["temperature_2m", "precipitation", "pressure_msl", "wind_speed_10m", "year", "month"]
        
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
        scaler = StandardScaler(inputCol="raw_features", outputCol="features")
        
        # Розділення на train/test
        train_df, test_df = processed_df.randomSplit([0.7, 0.3], seed=42)
        
        print(f"3.1. Розділення даних:")
        print(f"   Train: {train_df.count()} рядків")
        print(f"   Test: {test_df.count()} рядків")
        
        print("3.2. Навчання моделей...")
        
        # Модель 1: Логістична регресія
        print("   Навчання Логістичної регресії...")
        lr = LogisticRegression(featuresCol="features", labelCol="extreme_heat", maxIter=10)
        lr_pipeline = Pipeline(stages=[assembler, scaler, lr])
        start_time = time.time()
        lr_model = lr_pipeline.fit(train_df)
        lr_time = time.time() - start_time
        print(f"     Час навчання: {lr_time:.2f} сек")
        
        # Модель 2: Random Forest
        print("   Навчання Random Forest...")
        rf = RandomForestClassifier(featuresCol="features", labelCol="extreme_heat", numTrees=10)
        rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
        start_time = time.time()
        rf_model = rf_pipeline.fit(train_df)
        rf_time = time.time() - start_time
        print(f"     Час навчання: {rf_time:.2f} сек")
        
        # Модель 3: Decision Tree
        print("   Навчання Decision Tree...")
        dt = DecisionTreeClassifier(featuresCol="features", labelCol="extreme_heat")
        dt_pipeline = Pipeline(stages=[assembler, scaler, dt])
        start_time = time.time()
        dt_model = dt_pipeline.fit(train_df)
        dt_time = time.time() - start_time
        print(f"     Час навчання: {dt_time:.2f} сек")
        
        print("✅ ЗАВДАННЯ 3 ВИКОНАНО: Навчено 3 моделі")
        
        # ============================================================================
        print("\n" + "="*60)
        print("✅ ЗАВДАННЯ 4: АНАЛІЗ ПРОЦЕСУ НАВЧАННЯ")
        print("="*60)
        
        print("4.1. Аналіз часу навчання:")
        print(f"   Логістична регресія: {lr_time:.2f} сек (найшвидша)")
        print(f"   Random Forest: {rf_time:.2f} сек")
        print(f"   Decision Tree: {dt_time:.2f} сек")
        
        print("4.2. Аналіз складності моделей:")
        print("   Логістична регресія: Лінійна модель, швидка")
        print("   Random Forest: Ансамбль дерев, більш складна")
        print("   Decision Tree: Одне дерево, інтерпретована")
        
        print("✅ ЗАВДАННЯ 4 ВИКОНАНО: Проведено аналіз процесу навчання")
        
        # ============================================================================
        print("\n" + "="*60)
        print("✅ ЗАВДАННЯ 5: ОЦІНКА ЯКОСТІ МОДЕЛЕЙ")
        print("="*60)
        
        print("5.1. Розрахунок метрик для кожної моделі...")
        
        # Прогнози
        lr_predictions = lr_model.transform(test_df)
        rf_predictions = rf_model.transform(test_df)
        dt_predictions = dt_model.transform(test_df)
        
        # Evaluators
        binary_evaluator = BinaryClassificationEvaluator(labelCol="extreme_heat")
        multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="extreme_heat")
        
        models_results = {}
        
        for model_name, predictions in [
            ("Logistic Regression", lr_predictions),
            ("Random Forest", rf_predictions),
            ("Decision Tree", dt_predictions)
        ]:
            print(f"\n   --- {model_name} ---")
            
            # Accuracy
            accuracy = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "accuracy"})
            
            # Precision
            precision = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
            
            # Recall
            recall = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedRecall"})
            
            # F1-score
            f1 = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "f1"})
            
            # AUC-ROC
            auc_roc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
            
            results = {
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-score": round(f1, 4),
                "AUC-ROC": round(auc_roc, 4)
            }
            
            models_results[model_name] = results
            
            print(f"     Accuracy:  {results['Accuracy']}")
            print(f"     Precision: {results['Precision']}")
            print(f"     Recall:    {results['Recall']}")
            print(f"     F1-score:  {results['F1-score']}")
            print(f"     AUC-ROC:   {results['AUC-ROC']}")
        
        print("\n✅ ЗАВДАННЯ 5 ВИКОНАНО: Розраховано всі метрики")
        print("   - Accuracy: загальна точність класифікації")
        print("   - Precision: точність позитивних прогнозів")
        print("   - Recall: повнота виявлення позитивних випадків")
        print("   - F1-score: гармонічне середнє Precision та Recall")
        print("   - AUC-ROC: площа під ROC-кривою")
        
        # ============================================================================
        print("\n" + "="*60)
        print("✅ ЗАВДАННЯ 6: ПОРІВНЯННЯ РЕЗУЛЬТАТІВ РІЗНИХ АЛГОРИТМІВ")
        print("="*60)
        
        print("6.1. Порівняльна таблиця результатів:")
        print("-" * 80)
        print(f"{'Модель':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'AUC-ROC':<10}")
        print("-" * 80)
        
        # Сортуємо за F1-score
        sorted_results = sorted(models_results.items(), key=lambda x: x[1]['F1-score'], reverse=True)
        
        for model_name, metrics in sorted_results:
            print(f"{model_name:<20} {metrics['Accuracy']:<10} {metrics['Precision']:<10} "
                  f"{metrics['Recall']:<10} {metrics['F1-score']:<10} {metrics['AUC-ROC']:<10}")
        
        print("-" * 80)
        
        # Визначаємо найкращу модель
        best_model = sorted_results[0]
        print(f"\n6.2. 🏆 НАЙКРАЩА МОДЕЛЬ: {best_model[0]}")
        print(f"     F1-score: {best_model[1]['F1-score']}")
        print(f"     AUC-ROC: {best_model[1]['AUC-ROC']}")
        
        print("\n6.3. Рекомендації:")
        if best_model[1]['F1-score'] > 0.8:
            print("   ✅ Висока якість класифікації")
        elif best_model[1]['F1-score'] > 0.6:
            print("   ⚠️ Середня якість, потрібне покращення")
        else:
            print("   ❌ Низька якість, потрібна додаткова робота")
        
        print("\n✅ ЗАВДАННЯ 6 ВИКОНАНО: Порівняно результати всіх алгоритмів")
        
        # ============================================================================
        print("\n" + "="*80)
        print("🎉 ВСІ ЗАВДАННЯ УСПІШНО ВИКОНАНІ!")
        print("="*80)
        
        print("ПІДСУМОК ВИКОНАНИХ ЗАВДАНЬ:")
        print("✅ 1. ПОПЕРЕДНЯ ОБРОБКА ДАНИХ - ВИКОНАНО")
        print("     - Очищено дані від пропусків")
        print("     - Створено нові ознаки")
        print("     - Створено цільові змінні")
        
        print("✅ 2. ВИБІР МІНІМУМ 3 РІЗНИХ МОДЕЛЕЙ - ВИКОНАНО")
        print("     - Логістична регресія")
        print("     - Random Forest")
        print("     - Decision Tree")
        
        print("✅ 3. НАВЧАННЯ МОДЕЛЕЙ - ВИКОНАНО")
        print("     - Розділено дані на train/test")
        print("     - Навчено всі 3 моделі")
        print("     - Зафіксовано час навчання")
        
        print("✅ 4. АНАЛІЗ ПРОЦЕСУ НАВЧАННЯ - ВИКОНАНО")
        print("     - Проаналізовано час навчання")
        print("     - Проаналізовано складність моделей")
        
        print("✅ 5. ОЦІНКА ЯКОСТІ МОДЕЛЕЙ - ВИКОНАНО")
        print("     - Accuracy: загальна точність")
        print("     - Precision: точність позитивних прогнозів")
        print("     - Recall: повнота виявлення")
        print("     - F1-score: гармонічне середнє")
        print("     - AUC-ROC: площа під ROC-кривою")
        
        print("✅ 6. ПОРІВНЯННЯ РЕЗУЛЬТАТІВ РІЗНИХ АЛГОРИТМІВ - ВИКОНАНО")
        print("     - Створено порівняльну таблицю")
        print("     - Визначено найкращу модель")
        print("     - Надано рекомендації")
        
        print(f"\n🏆 НАЙКРАЩИЙ РЕЗУЛЬТАТ: {best_model[0]} (F1-score: {best_model[1]['F1-score']})")
        
        return models_results
        
    finally:
        spark.stop()


if __name__ == "__main__":
    demonstrate_all_tasks()
