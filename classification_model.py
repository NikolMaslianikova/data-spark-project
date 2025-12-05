"""
Класифікаційна модель для погодних даних у PySpark
Задачі класифікації на основі бізнес-питань:
1. Екстремальна спека (≥40°C) - Бізнес-питання 3
2. Штормові дні (високі опади + сильний вітер) - Бізнес-питання 5
3. Висока вологість (≥80%) - Бізнес-питання 6

Для кожної задачі використовуються 3 моделі:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting (GBT)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, year, month, dayofmonth, dayofweek, 
    count, lit, rand, monotonically_increasing_id, mean as spark_mean, stddev, lag, abs as spark_abs
)
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, Imputer
)
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import time
import os
import glob
from src.io_utils import load_weather_data


# Визначення 3 задач класифікації на основі бізнес-питань
# Вибрано питання з кращим розподілом класів для якісного навчання моделей
CLASSIFICATION_TASKS = {
    "task_1_rainy_days": {
        "name": "Дощові дні",
        "business_question": "Які міста мають найбільше дощових днів у сезон мусонів (черв–вер)?",
        "target_col": "rainy_day",
        "description": "Класифікація днів як 'дощовий' (опади > 0.1mm) або 'сухий'"
    },
    "task_2_temperature_anomaly": {
        "name": "Аномальна температура",
        "business_question": "Чи фіксуються послідовні піки/провали температури в квітні–червні?",
        "target_col": "temperature_anomaly",
        "description": "Класифікація днів як 'аномальна температура' (відхилення > 1σ) або 'нормальна'"
    },
    "task_3_temperature_spike": {
        "name": "Піки та провали температури",
        "business_question": "Чи фіксуються послідовні піки/провали 7-денного ковзного середнього температури в квітні–червні, і як під це коригувати тарифи/потужності?",
        "target_col": "temperature_spike",
        "description": "Класифікація днів як 'пік/провал температури' (різка зміна день-до-дня > 3°C) або 'нормальна зміна'"
    }
}


def create_target_variable(df, task_config):
    """
    Створює цільову змінну на основі конфігурації задачі
    
    Args:
        df: DataFrame з даними
        task_config: Словник з конфігурацією задачі
    """
    target_col = task_config["target_col"]
    
    if target_col == "rainy_day":
        # Задача 1: Дощові дні (опади > 0.1mm) - зменшено поріг для кращого балансу
        df_with_target = df.withColumn(
            target_col,
            when(col("precipitation") > 0.1, 1.0).otherwise(0.0)
        )
    elif target_col == "temperature_anomaly":
        # Задача 2: Аномальна температура (відхилення від середньої > 2σ)
        # Спочатку обчислюємо середнє та стандартне відхилення
        temp_stats = df.select(
            spark_mean("temperature_2m").alias("mean_temp"),
            stddev("temperature_2m").alias("std_temp")
        ).collect()[0]
        
        mean_temp = temp_stats["mean_temp"] if temp_stats["mean_temp"] is not None else 25.0
        std_temp = temp_stats["std_temp"] if temp_stats["std_temp"] is not None else 5.0
        
        # Аномалія = відхилення більше ніж 1 стандартне відхилення (зменшено для кращого балансу)
        df_with_target = df.withColumn(
            target_col,
            when(
                (col("temperature_2m") > mean_temp + 1 * std_temp) | 
                (col("temperature_2m") < mean_temp - 1 * std_temp),
                1.0
            ).otherwise(0.0)
        )
    elif target_col == "temperature_spike":
        # Задача 3: Піки та провали температури (різка зміна день-до-дня > 3°C)
        # Спочатку обчислюємо зміну температури від попереднього дня
        # Сортуємо по місту та даті для коректного обчислення lag (щоб не змішувати дані різних міст)
        if "city" in df.columns:
            window_spec = Window.partitionBy("city").orderBy("date")
        else:
            window_spec = Window.orderBy("date")
        
        df_with_lag = df.withColumn(
            "prev_temperature",
            lag("temperature_2m", 1).over(window_spec)
        ).withColumn(
            "temp_change",
            spark_abs(col("temperature_2m") - col("prev_temperature"))
        )
        
        # Пік/провал = зміна температури більше 3°C (поріг зменшено для кращого балансу)
        # Враховуємо тільки дні, де є попередня температура (не перший день для кожного міста)
        df_with_target = df_with_lag.withColumn(
            target_col,
            when(
                (col("prev_temperature").isNotNull()) & (col("temp_change") > 3.0),
                1.0
            ).otherwise(0.0)
        ).drop("prev_temperature", "temp_change")
    else:
        raise ValueError(f"Невідома задача: {target_col}")
    
    return df_with_target


def preprocess_data(df, task_config):
    """
    Попередня обробка даних для конкретної задачі класифікації:
    1. Видалення порожніх колонок
    2. Створення цільової змінної для задачі
    3. Витягування датних ознак
    4. Обробка пропущених значень
    
    Args:
        df: DataFrame з даними
        task_config: Словник з конфігурацією задачі
    """
    task_name = task_config["name"]
    target_col = task_config["target_col"]
    
    print(f"\n=== ЕТАП 1: ПОПЕРЕДНЯ ОБРОБКА ДАНИХ - {task_name} ===\n")
    
    # Видаляємо порожню колонку
    df_clean = df.drop("empty")
    
    # Створюємо цільову змінну для цієї задачі
    df_with_target = create_target_variable(df_clean, task_config)
    
    # Витягуємо датні ознаки
    df_features = df_with_target.withColumn(
        "year", year(col("date"))
    ).withColumn(
        "month", month(col("date"))
    ).withColumn(
        "day", dayofmonth(col("date"))
    ).withColumn(
        "day_of_week", dayofweek(col("date"))
    )
    
    # Вибираємо найважливіші числові ознаки для моделі
    feature_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "pressure_msl",
        "precipitation",
        "wind_speed_10m",
        "month",
        "day_of_week"
    ]
    
    # Перевіряємо наявність колонок
    available_features = [c for c in feature_columns if c in df_features.columns]
    print(f"Використовуємо {len(available_features)} ознак: {available_features}")
    
    # Фільтруємо рядки з відсутньою цільовою змінною та необхідними полями
    filter_conditions = [col(target_col).isNotNull()]
    
    # Додаємо умови фільтрації залежно від задачі
    if target_col == "rainy_day":
        filter_conditions.append(col("precipitation").isNotNull())
    elif target_col == "temperature_anomaly":
        filter_conditions.append(col("temperature_2m").isNotNull())
    elif target_col == "temperature_spike":
        filter_conditions.append(col("temperature_2m").isNotNull())
        filter_conditions.append(col("date").isNotNull())  # Потрібна дата для сортування
    
    # Об'єднуємо всі умови фільтрації
    from functools import reduce
    from operator import and_ as op_and
    final_filter = reduce(op_and, filter_conditions)
    df_filtered = df_features.filter(final_filter)
    
    print(f"\nРозподіл цільової змінної '{target_col}' (буде показано після обробки)...")
    
    return df_filtered, available_features, target_col


def create_feature_pipeline(feature_columns, use_scaler=True):
    """
    Створює pipeline для обробки ознак:
    1. Imputer для заповнення пропущених значень
    2. VectorAssembler для об'єднання ознак
    3. StandardScaler для нормалізації (опціонально)
    
    Args:
        use_scaler: Чи використовувати StandardScaler (вимкнути для швидкості)
    """
    # Imputer для заповнення пропущених значень середнім значенням
    # Використовуємо стратегію "mean" для швидшої обробки
    imputer = Imputer(
        inputCols=feature_columns,
        outputCols=[f"{col}_imputed" for col in feature_columns],
        strategy="mean"
    )
    
    # Створюємо список імпутованих колонок
    imputed_columns = [f"{col}_imputed" for col in feature_columns]
    
    # VectorAssembler для об'єднання ознак у вектор
    assembler = VectorAssembler(
        inputCols=imputed_columns,
        outputCol="features_raw",
        handleInvalid="keep"
    )
    
    # StandardScaler для нормалізації ознак (опціонально)
    if use_scaler:
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        return [imputer, assembler, scaler]
    else:
        # Якщо скалер не використовується, використовуємо features напряму
        assembler_final = VectorAssembler(
            inputCols=imputed_columns,
            outputCol="features",
            handleInvalid="keep"
        )
        return [imputer, assembler_final]


def calculate_class_weights(train_df, target_col):
    """
    Обчислює ваги класів для балансування незбалансованих даних
    
    Args:
        train_df: Тренувальний DataFrame
        target_col: Назва цільової колонки
    
    Returns:
        dict: Словник з вагами для кожного класу
    """
    # Підраховуємо кількість прикладів кожного класу
    class_counts = train_df.groupBy(target_col).agg(
        count("*").alias("count")
    ).collect()
    
    total_samples = sum(row['count'] for row in class_counts)
    n_classes = len(class_counts)
    
    # Обчислюємо ваги: загальна кількість / (кількість класів * кількість прикладів класу)
    class_weights = {}
    for row in class_counts:
        class_label = float(row[target_col])
        class_count = row['count']
        # Зворотна частота для балансування
        weight = total_samples / (n_classes * class_count)
        class_weights[class_label] = weight
    
    print(f"Обчислені ваги класів: {class_weights}")
    return class_weights


def add_class_weights(df, target_col, class_weights):
    """
    Додає колонку з вагами класів до DataFrame
    
    Args:
        df: DataFrame
        target_col: Назва цільової колонки
        class_weights: Словник з вагами класів
    
    Returns:
        DataFrame з доданою колонкою class_weight
    """
    # Створюємо умову для додавання ваг
    conditions = None
    for class_label, weight in class_weights.items():
        condition = when(col(target_col) == class_label, lit(weight))
        if conditions is None:
            conditions = condition
        else:
            conditions = conditions.when(col(target_col) == class_label, lit(weight))
    
    # Додаємо колонку з вагами
    df_with_weights = df.withColumn("class_weight", conditions.otherwise(lit(1.0)))
    return df_with_weights


def balance_dataset_undersampling(df, target_col, max_ratio=3.0, seed=42):
    """
    Балансує датасет через undersampling majority class
    Обмежує кількість прикладів majority class до max_ratio * minority class
    
    Args:
        df: DataFrame
        target_col: Назва цільової колонки
        max_ratio: Максимальне співвідношення majority/minority (за замовчуванням 3:1)
        seed: Seed для відтворюваності
    
    Returns:
        Збалансований DataFrame
    """
    # Підраховуємо кількість прикладів кожного класу
    class_counts = df.groupBy(target_col).agg(
        count("*").alias("count")
    ).orderBy("count").collect()
    
    if len(class_counts) < 2:
        return df  # Якщо тільки один клас, повертаємо як є
    
    # Знаходимо minority та majority класи
    minority_count = class_counts[0]['count']
    minority_class = class_counts[0][target_col]
    majority_count = class_counts[-1]['count']
    majority_class = class_counts[-1][target_col]
    
    # Обчислюємо максимальну кількість прикладів majority class
    max_majority_samples = int(minority_count * max_ratio)
    
    # Якщо majority class має більше прикладів, ніж потрібно, робимо undersampling
    if majority_count > max_majority_samples:
        print(f"\n=== БАЛАНСУВАННЯ ДАТАСЕТУ ===")
        print(f"До балансування:")
        print(f"  Minority class ({minority_class}): {minority_count} прикладів")
        print(f"  Majority class ({majority_class}): {majority_count} прикладів")
        print(f"  Співвідношення: {majority_count/minority_count:.2f}:1")
        
        # Отримуємо majority class дані
        majority_df = df.filter(col(target_col) == majority_class)
        minority_df = df.filter(col(target_col) == minority_class)
        
        # Вибираємо випадкову вибірку з majority class
        sample_fraction = max_majority_samples / majority_count
        majority_sampled = majority_df.sample(False, sample_fraction, seed=seed).limit(max_majority_samples)
        
        # Об'єднуємо з minority class
        balanced_df = minority_df.union(majority_sampled)
        
        print(f"\nПісля балансування:")
        balanced_counts = balanced_df.groupBy(target_col).agg(
            count("*").alias("count")
        ).orderBy("count").collect()
        for row in balanced_counts:
            print(f"  Клас {row[target_col]}: {row['count']} прикладів")
        if len(balanced_counts) == 2:
            new_ratio = balanced_counts[1]['count'] / balanced_counts[0]['count']
            print(f"  Нове співвідношення: {new_ratio:.2f}:1")
        print("="*50)
        return balanced_df
    else:
        print(f"\nБалансування не потрібне: співвідношення класів {majority_count}:{minority_count} ({majority_count/minority_count:.2f}:1) прийнятне")
        return df


def stratified_split(df, target_col, train_ratio=0.8, seed=42, min_test_samples_per_class=1):
    """
    Стратифікований розподіл даних на тренувальний та тестовий набори
    Забезпечує, що обидва класи присутні в обох наборах
    
    Args:
        df: DataFrame
        target_col: Назва цільової колонки
        train_ratio: Частка тренувальних даних
        seed: Seed для відтворюваності
        min_test_samples_per_class: Мінімальна кількість прикладів кожного класу в тестовому наборі
    
    Returns:
        train_df, test_df: Тренувальний та тестовий DataFrames
    """
    # Розділяємо дані по класах
    train_dfs = []
    test_dfs = []
    
    # Отримуємо унікальні класи та їх кількість
    unique_classes = df.select(target_col).distinct().collect()
    class_counts = {}
    
    for row in unique_classes:
        class_label = row[target_col]
        class_df = df.filter(col(target_col) == class_label)
        class_count = class_df.count()
        class_counts[class_label] = class_count
        
        # Для дуже малих класів забезпечуємо мінімум у тестовому наборі
        if class_count <= min_test_samples_per_class * 2:
            # Якщо клас дуже малий, залишаємо мінімум у тестовому наборі
            test_count = min(min_test_samples_per_class, max(1, class_count // 2))
            if test_count > 0 and class_count > test_count:
                # Використовуємо sample для отримання випадкових прикладів
                test_df_class = class_df.sample(False, test_count / class_count, seed=seed).limit(test_count)
                # Створюємо список ID колонок для join (використовуємо всі колонки)
                # Для left_anti join потрібно використати всі колонки
                class_df_with_id = class_df.withColumn("_temp_id", monotonically_increasing_id())
                test_df_class_with_id = test_df_class.withColumn("_temp_id", monotonically_increasing_id())
                # Використовуємо left_anti join за temp_id
                train_df_class = class_df_with_id.join(
                    test_df_class_with_id.select("_temp_id"), 
                    on="_temp_id", 
                    how="left_anti"
                ).drop("_temp_id")
            else:
                # Якщо клас занадто малий, весь йде в тренувальний набір
                train_df_class = class_df
                test_df_class = class_df.limit(0)
        else:
            # Для великих класів використовуємо стандартний split
            class_train, class_test = class_df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
            train_df_class = class_train
            test_df_class = class_test
        
        train_dfs.append(train_df_class)
        test_dfs.append(test_df_class)
    
    # Об'єднуємо всі класи
    if len(train_dfs) > 0:
        train_df = train_dfs[0]
        for df_part in train_dfs[1:]:
            train_df = train_df.union(df_part)
    else:
        train_df = df.limit(0)
    
    if len(test_dfs) > 0:
        test_df = test_dfs[0]
        for df_part in test_dfs[1:]:
            test_df = test_df.union(df_part)
    else:
        test_df = df.limit(0)
    
    # Перевіряємо, що обидва класи є в тестовому наборі
    test_class_counts = test_df.groupBy(target_col).agg(count("*").alias("count")).collect()
    test_classes = {row[target_col]: row['count'] for row in test_class_counts}
    
    print(f"Розподіл класів у тестовому наборі: {test_classes}")
    
    return train_df, test_df


def train_logistic_regression(train_df, feature_pipeline_stages, target_col, class_weights=None):
    """
    Навчання моделі Logistic Regression
    
    Args:
        train_df: Тренувальний DataFrame
        feature_pipeline_stages: Стадії pipeline для обробки ознак
        target_col: Назва цільової колонки
    """
    print("\n=== НАВЧАННЯ МОДЕЛІ 1: LOGISTIC REGRESSION ===\n")
    
    start_time = time.time()
    
    # Створюємо модель Logistic Regression з class weights для балансування
    # Збільшуємо регуляризацію для запобігання перетренуванню
    lr_params = {
        "featuresCol": "features",
        "labelCol": target_col,
        "maxIter": 50,  # Зменшено для запобігання перетренуванню
        "regParam": 0.3,  # Ще більша регуляризація для запобігання перетренуванню
        "elasticNetParam": 0.1,  # Додаємо L1 регуляризацію
        "tol": 1e-4  # Збільшена толерантність для швидшого завершення
    }
    
    # Додаємо class weights, якщо вони надані
    if class_weights:
        # Перевіряємо, чи існує колонка class_weight
        if "class_weight" in train_df.columns:
            lr_params["weightCol"] = "class_weight"
            sorted_classes = sorted(class_weights.keys())
            weight_list = [class_weights[c] for c in sorted_classes]
            print(f"Використовуються ваги класів: {weight_list}")
        else:
            print("⚠ Попередження: Колонка class_weight не знайдена, ваги не використовуються")
    
    lr = LogisticRegression(**lr_params)
    
    # Створюємо pipeline
    pipeline_stages = list(feature_pipeline_stages) + [lr]
    pipeline = Pipeline(stages=pipeline_stages)
    
    # Навчаємо модель
    print("Початок навчання Logistic Regression...")
    model = pipeline.fit(train_df)
    
    training_time = time.time() - start_time
    print(f"Час навчання: {training_time:.2f} секунд")
    
    return model, training_time


def train_random_forest(train_df, feature_pipeline_stages, target_col, class_weights=None):
    """
    Навчання моделі Random Forest
    
    Args:
        train_df: Тренувальний DataFrame
        feature_pipeline_stages: Стадії pipeline для обробки ознак
        target_col: Назва цільової колонки
        class_weights: Словник з вагами класів
    """
    print("\n=== НАВЧАННЯ МОДЕЛІ 2: RANDOM FOREST ===\n")
    
    start_time = time.time()
    
    # Створюємо модель Random Forest з class weights
    # Зменшуємо глибину та кількість дерев для запобігання перетренуванню
    rf_params = {
        "featuresCol": "features",
        "labelCol": target_col,
        "numTrees": 20,  # Ще більше зменшено для запобігання перетренуванню
        "maxDepth": 3,   # Ще більше зменшено для запобігання перетренуванню
        "minInstancesPerNode": 10,  # Мінімум прикладів у вузлі
        "seed": 42
    }
    
    if class_weights:
        if "class_weight" in train_df.columns:
            rf_params["weightCol"] = "class_weight"
            sorted_classes = sorted(class_weights.keys())
            weight_list = [class_weights[c] for c in sorted_classes]
            print(f"Використовуються ваги класів: {weight_list}")
        else:
            print("⚠ Попередження: Колонка class_weight не знайдена, ваги не використовуються")
    
    rf = RandomForestClassifier(**rf_params)
    
    # Створюємо pipeline
    pipeline_stages = list(feature_pipeline_stages) + [rf]
    pipeline = Pipeline(stages=pipeline_stages)
    
    # Навчаємо модель
    print("Початок навчання Random Forest...")
    model = pipeline.fit(train_df)
    
    training_time = time.time() - start_time
    print(f"Час навчання: {training_time:.2f} секунд")
    
    return model, training_time


def train_gradient_boosting(train_df, feature_pipeline_stages, target_col, class_weights=None):
    """
    Навчання моделі Gradient Boosting
    
    Args:
        train_df: Тренувальний DataFrame
        feature_pipeline_stages: Стадії pipeline для обробки ознак
        target_col: Назва цільової колонки
        class_weights: Словник з вагами класів
    """
    print("\n=== НАВЧАННЯ МОДЕЛІ 3: GRADIENT BOOSTING ===\n")
    
    start_time = time.time()
    
    # Створюємо модель Gradient Boosting з class weights
    # Зменшуємо ітерації та глибину для запобігання перетренуванню
    gbt_params = {
        "featuresCol": "features",
        "labelCol": target_col,
        "maxIter": 20,  # Ще більше зменшено для запобігання перетренуванню
        "maxDepth": 2,  # Ще більше зменшено для запобігання перетренуванню
        "minInstancesPerNode": 10,  # Мінімум прикладів у вузлі
        "seed": 42
    }
    
    if class_weights:
        if "class_weight" in train_df.columns:
            gbt_params["weightCol"] = "class_weight"
            sorted_classes = sorted(class_weights.keys())
            weight_list = [class_weights[c] for c in sorted_classes]
            print(f"Використовуються ваги класів: {weight_list}")
        else:
            print("⚠ Попередження: Колонка class_weight не знайдена, ваги не використовуються")
    
    gbt = GBTClassifier(**gbt_params)
    
    # Створюємо pipeline
    pipeline_stages = list(feature_pipeline_stages) + [gbt]
    pipeline = Pipeline(stages=pipeline_stages)
    
    # Навчаємо модель
    print("Початок навчання Gradient Boosting...")
    model = pipeline.fit(train_df)
    
    training_time = time.time() - start_time
    print(f"Час навчання: {training_time:.2f} секунд")
    
    return model, training_time


def analyze_feature_importance(model, model_name, feature_columns):
    """
    Аналізує та виводить важливість ознак (feature importance) для моделі
    
    Args:
        model: Навчена модель (Pipeline)
        model_name: Назва моделі
        feature_columns: Список назв ознак
    """
    print(f"\n=== АНАЛІЗ ВАЖЛИВОСТІ ОЗНАК: {model_name} ===\n")
    
    try:
        # Отримуємо останній stage з pipeline (це сама модель класифікації)
        trained_model = model.stages[-1]
        
        # Для Random Forest та Gradient Boosting
        if hasattr(trained_model, 'featureImportances'):
            importances = trained_model.featureImportances
            
            # Конвертуємо в масив (може бути SparseVector або DenseVector)
            if hasattr(importances, 'toArray'):
                importances_array = importances.toArray()
            else:
                # Якщо це вже масив або список
                importances_array = list(importances) if hasattr(importances, '__iter__') else [float(importances)]
            
            # Створюємо список (назва, важливість)
            feature_importance_list = []
            num_features = min(len(feature_columns), len(importances_array))
            for i in range(num_features):
                feature_name = feature_columns[i]
                importance_value = float(importances_array[i])
                feature_importance_list.append((feature_name, importance_value))
            
            # Сортуємо за важливістю (від більшого до меншого)
            feature_importance_list.sort(key=lambda x: x[1], reverse=True)
            
            print("Важливість ознак (від найважливішої до найменш важливої):")
            print("-" * 60)
            print(f"{'Ознака':<30} {'Важливість':<15}")
            print("-" * 60)
            for feature_name, importance in feature_importance_list:
                print(f"{feature_name:<30} {importance:<15.6f}")
            
            # Показуємо топ-3 найважливіших ознак
            if len(feature_importance_list) >= 3:
                total_importance = sum(imp[1] for imp in feature_importance_list)
                print("\nТоп-3 найважливіші ознаки:")
                for i, (feature_name, importance) in enumerate(feature_importance_list[:3], 1):
                    percentage = (importance / total_importance) * 100 if total_importance > 0 else 0
                    print(f"  {i}. {feature_name}: {importance:.6f} ({percentage:.2f}%)")
        
        # Для Logistic Regression - використовуємо коефіцієнти
        elif hasattr(trained_model, 'coefficients'):
            coefficients = trained_model.coefficients
            
            # Конвертуємо коефіцієнти в масив (може бути SparseVector або DenseVector)
            if hasattr(coefficients, 'toArray'):
                coeff_array = coefficients.toArray()
            elif hasattr(coefficients, '__iter__'):
                coeff_array = list(coefficients)
            else:
                coeff_array = [float(coefficients)]
            
            # Створюємо список (назва, абсолютне значення коефіцієнта, коефіцієнт)
            feature_coeff_list = []
            num_features = min(len(feature_columns), len(coeff_array))
            for i in range(num_features):
                feature_name = feature_columns[i]
                coeff_value = float(coeff_array[i])
                feature_coeff_list.append((feature_name, abs(coeff_value), coeff_value))
            
            # Сортуємо за абсолютним значенням коефіцієнта
            feature_coeff_list.sort(key=lambda x: x[1], reverse=True)
            
            print("Коефіцієнти ознак (абсолютні значення, від найбільшого):")
            print("-" * 70)
            print(f"{'Ознака':<30} {'|Коефіцієнт|':<15} {'Коефіцієнт':<15}")
            print("-" * 70)
            for feature_name, abs_coeff, coeff in feature_coeff_list:
                print(f"{feature_name:<30} {abs_coeff:<15.6f} {coeff:<15.6f}")
            
            # Показуємо топ-3 найважливіших ознак
            if len(feature_coeff_list) >= 3:
                print("\nТоп-3 найважливіші ознаки (за абсолютним значенням коефіцієнта):")
                total_abs = sum(imp[1] for imp in feature_coeff_list)
                for i, (feature_name, abs_coeff, coeff) in enumerate(feature_coeff_list[:3], 1):
                    percentage = (abs_coeff / total_abs) * 100 if total_abs > 0 else 0
                    direction = "позитивний" if coeff > 0 else "негативний"
                    print(f"  {i}. {feature_name}: {abs_coeff:.6f} ({percentage:.2f}%, {direction} вплив)")
        
        else:
            print("⚠ Цей тип моделі не підтримує аналіз важливості ознак")
            
    except Exception as e:
        print(f"⚠ Помилка при аналізі важливості ознак: {e}")
        import traceback
        traceback.print_exc()


def evaluate_model(model, test_df, model_name, target_col, max_test_rows=50000, feature_columns=None):
    """
    Оцінка якості моделі:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Feature Importance (якщо доступно)
    
    Args:
        model: Навчена модель
        test_df: Тестовий DataFrame
        model_name: Назва моделі
        target_col: Назва цільової колонки
        max_test_rows: Максимальна кількість рядків для оцінки (для швидкості)
        feature_columns: Список назв ознак для аналізу важливості
    """
    print(f"\n=== ОЦІНКА МОДЕЛІ: {model_name} ===\n")
    
    # Аналізуємо важливість ознак перед оцінкою
    if feature_columns:
        analyze_feature_importance(model, model_name, feature_columns)
    
    # Обмежуємо тестовий набір для швидкої оцінки
    if max_test_rows:
        print(f"Обмежуємо тестовий набір до {max_test_rows:,} рядків для швидкої оцінки...")
        test_df_limited = test_df.limit(max_test_rows)
    else:
        test_df_limited = test_df
    
    # Перевіряємо, що тестовий набір не порожній
    try:
        test_check = test_df_limited.limit(1).count()
        if test_check == 0:
            print("⚠ Попередження: Тестовий набір порожній, пропускаємо оцінку")
            return None
    except Exception as e:
        print(f"⚠ Попередження при перевірці тестового набору: {e}")
        # Продовжуємо спробувати оцінити модель
    
    # Робимо прогнози
    predictions = model.transform(test_df_limited)
    
    # Перевіряємо розподіл класів у прогнозах та реальних значеннях
    print("Перевірка розподілу класів...")
    has_both_classes = False  # Ініціалізуємо як False за замовчуванням
    try:
        label_dist = predictions.groupBy(target_col).agg(count("*").alias("count")).collect()
        pred_dist = predictions.groupBy("prediction").agg(count("*").alias("count")).collect()
        
        print("Розподіл реальних значень:")
        for row in label_dist:
            print(f"  Клас {row[target_col]}: {row['count']} прикладів")
        
        print("Розподіл прогнозів:")
        for row in pred_dist:
            print(f"  Клас {row['prediction']}: {row['count']} прикладів")
        
        # Перевіряємо, чи є обидва класи
        unique_labels = set([float(row[target_col]) for row in label_dist])
        unique_predictions = set([float(row['prediction']) for row in pred_dist])
        has_both_classes = len(unique_labels) > 1 and len(unique_predictions) > 1
        
        if not has_both_classes:
            print("⚠ Попередження: Дані містять тільки один клас. Деякі метрики можуть бути недоступні.")
            print(f"   Унікальні реальні класи: {unique_labels}")
            print(f"   Унікальні прогнозовані класи: {unique_predictions}")
    except Exception as e:
        print(f"⚠ Помилка при перевірці розподілу: {e}")
        print("   Припускаємо, що є тільки один клас для безпеки")
        has_both_classes = False  # Безпечніше припустити False
    
    # Оцінювачі
    multiclass_evaluator = MulticlassClassificationEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    # Обчислюємо метрики
    accuracy = multiclass_evaluator.evaluate(predictions)
    
    # Precision
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    precision = precision_evaluator.evaluate(predictions)
    
    # Recall
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="weightedRecall"
    )
    recall = recall_evaluator.evaluate(predictions)
    
    # F1-score
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="f1"
    )
    f1_score = f1_evaluator.evaluate(predictions)
    
    # AUC-ROC (тільки якщо є обидва класи та правильна форма rawPrediction)
    auc_roc = 0.0  # Значення за замовчуванням
    
    # Завжди перевіряємо форму rawPrediction перед спробою обчислити AUC-ROC
    can_compute_auc = False
    if "rawPrediction" in predictions.columns:
        try:
            # Перевіряємо форму rawPrediction на першому рядку
            sample_pred = predictions.select("rawPrediction").limit(1).collect()
            if sample_pred and len(sample_pred) > 0:
                raw_pred_value = sample_pred[0]["rawPrediction"]
                # Перевіряємо, чи це вектор з 2 елементами (для бінарної класифікації)
                if hasattr(raw_pred_value, '__len__') and len(raw_pred_value) == 2:
                    can_compute_auc = True
                else:
                    print(f"⚠ rawPrediction має неправильну форму: {len(raw_pred_value) if hasattr(raw_pred_value, '__len__') else 'N/A'} елементів (очікується 2)")
                    print("   Це зазвичай означає, що модель навчена на даних з одним класом")
        except Exception as e:
            print(f"⚠ Помилка при перевірці форми rawPrediction: {e}")
    
    # Обчислюємо AUC-ROC тільки якщо є обидва класи І правильна форма rawPrediction
    if has_both_classes and can_compute_auc:
        try:
            binary_evaluator = BinaryClassificationEvaluator(
                labelCol=target_col,
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
            auc_roc = binary_evaluator.evaluate(predictions)
        except Exception as e:
            print(f"⚠ Не вдалося обчислити AUC-ROC: {e}")
            print("   Можлива причина: всі прогнози одного класу або проблема з rawPrediction")
            auc_roc = 0.0
    else:
        if not has_both_classes:
            print("⚠ AUC-ROC не обчислюється: дані містять тільки один клас")
        elif not can_compute_auc:
            print("⚠ AUC-ROC не обчислюється: rawPrediction має неправильну форму")
        auc_roc = 0.0
    
    # Виводимо результати
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1_score:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    
    # Показуємо матрицю плутанини (confusion matrix)
    print("\nМатриця плутанини (Confusion Matrix):")
    confusion_matrix = predictions.groupBy(target_col, "prediction").agg(
        count("*").alias("count")
    ).orderBy(target_col, "prediction")
    confusion_matrix.show()
    
    # Показуємо приклади прогнозів
    print("\nПриклади прогнозів (перші 10):")
    # Вибираємо колонки, які точно існують
    select_cols = ["date", "city", "temperature_2m", target_col, "prediction"]
    if "probability" in predictions.columns:
        select_cols.append("probability")
    predictions.select(*select_cols).show(10, truncate=False)
    
    # Переконаємося, що auc_roc має значення
    if auc_roc is None:
        auc_roc = 0.0
    
    return {
        "model_name": model_name,
        "target_col": target_col,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "auc_roc": auc_roc
    }


def compare_models(results):
    """
    Порівняння результатів різних моделей
    """
    print("\n" + "="*80)
    print("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ МОДЕЛЕЙ")
    print("="*80 + "\n")
    
    if not results:
        print("⚠ Помилка: Немає результатів для порівняння!")
        return
    
    # Створюємо таблицю порівняння
    # Перевіряємо, чи є інформація про задачу
    has_task_info = 'task_name' in results[0] if results else False
    
    if has_task_info:
        print(f"{'Задача':<20} {'Модель':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'AUC-ROC':<12}")
        print("-" * 120)
    for result in results:
            task_name = result.get('task_name', 'N/A')
            print(f"{task_name:<20} {result['model_name']:<35} "
                  f"{result['accuracy']:<12.4f} "
                  f"{result['precision']:<12.4f} "
                  f"{result['recall']:<12.4f} "
                  f"{result['f1_score']:<12.4f} "
                  f"{result['auc_roc']:<12.4f}")
    else:
        print(f"{'Модель':<50} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'AUC-ROC':<12}")
        print("-" * 100)
        for result in results:
            print(f"{result['model_name']:<50} "
              f"{result['accuracy']:<12.4f} "
              f"{result['precision']:<12.4f} "
              f"{result['recall']:<12.4f} "
              f"{result['f1_score']:<12.4f} "
              f"{result['auc_roc']:<12.4f}")
    
    # Знаходимо найкращу модель за кожною метрикою
    print("\n" + "="*80)
    print("НАЙКРАЩІ РЕЗУЛЬТАТИ:")
    print("="*80)
    
    if len(results) > 0:
        best_accuracy = max(results, key=lambda x: x['accuracy'])
        best_precision = max(results, key=lambda x: x['precision'])
        best_recall = max(results, key=lambda x: x['recall'])
        best_f1 = max(results, key=lambda x: x['f1_score'])
        best_auc = max(results, key=lambda x: x['auc_roc'])
    
    print(f"Найкраща Accuracy:  {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
    print(f"Найкраща Precision: {best_precision['model_name']} ({best_precision['precision']:.4f})")
    print(f"Найкращий Recall:   {best_recall['model_name']} ({best_recall['recall']:.4f})")
    print(f"Найкращий F1-score:  {best_f1['model_name']} ({best_f1['f1_score']:.4f})")
    print(f"Найкращий AUC-ROC:   {best_auc['model_name']} ({best_auc['auc_roc']:.4f})")
    
    # Загальна рекомендація
    print("\n" + "="*80)
    print("РЕКОМЕНДАЦІЯ:")
    print("="*80)
    
    # Використовуємо F1-score як основну метрику (баланс між precision і recall)
    best_overall = max(results, key=lambda x: x['f1_score'])
    print(f"Рекомендована модель: {best_overall['model_name']}")
    print(f"  - F1-score: {best_overall['f1_score']:.4f}")
    print(f"  - Accuracy: {best_overall['accuracy']:.4f}")
    print(f"  - Precision: {best_overall['precision']:.4f}")
    print(f"  - Recall: {best_overall['recall']:.4f}")


def main():
    """
    Головна функція для виконання класифікації
    """
    # Створюємо SparkSession з оптимізаціями для великої вибірки
    spark = SparkSession.builder \
        .appName("WeatherClassification") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.files.maxPartitionBytes", "134217728") \
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728") \
        .config("spark.default.parallelism", "4") \
        .config("spark.sql.files.maxRecordsPerFile", "50000") \
        .config("spark.memory.fraction", "0.7") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*80)
    print("КЛАСИФІКАЦІЙНА МОДЕЛЬ ДЛЯ ПОГОДНИХ ДАНИХ")
    print("3 БІЗНЕС-ПИТАННЯ × 3 МОДЕЛІ = 9 МОДЕЛЕЙ ЗАГАЛОМ")
    print("="*80)
    print("\nЗадачі класифікації:")
    for task_id, task_config in CLASSIFICATION_TASKS.items():
        print(f"  - {task_config['name']}: {task_config['description']}")
    print("="*80)
    
    # Завантажуємо дані для навчання на великій вибірці
    print("\n=== ЗАВАНТАЖЕННЯ ДАНИХ ===\n")
    print("Завантажуємо велику вибірку для кращих результатів...")
    
    MAX_ROWS = 100000  # Велика вибірка: 100 тисяч рядків для кращої якості моделей
    print(f"Обмежуємо дані до максимум {MAX_ROWS:,} рядків...")
    
    # ВАЖЛИВО: Завантажуємо дані для навчання
    # Для кращого розподілу класів завантажуємо багато файлів
    data_dir = "/app/data"
    file_name = "невизначено"
    df_raw = None  # Ініціалізуємо змінну
    
    try:
        # Використовуємо Python для пошуку файлів
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if csv_files:
            # Для кращого розподілу класів використовуємо багато файлів
            # Завантажуємо до 20 файлів для кращого розподілу класів
            num_files_to_load = min(20, len(csv_files))
            selected_files = csv_files[:num_files_to_load]
            file_names = [os.path.basename(f) for f in selected_files]
            print(f"Завантажуємо {num_files_to_load} файл(ів): {', '.join(file_names)}")
            
            if num_files_to_load == 1:
                data_path = selected_files[0]
                file_name = file_names[0]
            else:
                # Завантажуємо файли окремо та об'єднуємо їх
                print(f"Завантажуємо та об'єднуємо {num_files_to_load} файлів...")
                dfs = []
                for file_path in selected_files:
                    df_part = load_weather_data(spark, file_path, extract_city=True)
                    dfs.append(df_part)
                
                # Об'єднуємо всі DataFrame
                df_raw = dfs[0]
                for df_part in dfs[1:]:
                    df_raw = df_raw.union(df_part)
                
                file_name = f"{num_files_to_load} файлів"
        else:
            # Якщо не знайдено через glob, спробуємо завантажити всі файли
            # Spark підтримує glob patterns напряму
            print("⚠ Не знайдено файлів через Python glob, спробуємо через Spark...")
            data_path = os.path.join(data_dir, "*.csv")
            file_name = "всі файли (через Spark glob)"
            print(f"Шлях: {data_path}")
    except Exception as e:
        print(f"⚠ Помилка при пошуку файлів: {e}")
        print("Спробуємо завантажити через Spark glob pattern...")
        data_path = os.path.join(data_dir, "*.csv")
        file_name = "всі файли (через Spark glob)"
    
    # Завантажуємо дані з обмеженням (якщо ще не завантажені)
    if df_raw is None:
        print("Завантаження даних (це може зайняти хвилину)...")
        print("⚠ ВАЖЛИВО: Завантаження може бути повільним через великі файли")
        print("   Рекомендується використовувати один маленький файл для тестування")
        
        df_raw = load_weather_data(spark, data_path, extract_city=True)
    
    # Застосовуємо обмеження одразу після завантаження
    print("Застосування обмежень до даних...")
    # Використовуємо limit одразу (швидше ніж sample + limit)
    print(f"Обмежуємо до {MAX_ROWS} рядків...")
    df = df_raw.limit(MAX_ROWS)
    
    # Оптимізуємо партиції для великого набору даних
    print("Оптимізація партицій...")
    # Для великих даних використовуємо більше партицій для кращої продуктивності
    df = df.coalesce(4)  # 4 партиції для кращої продуктивності
    
    # Видаляємо посилання на великий датафрейм
    del df_raw
    
    # Перевіряємо, що дані завантажені
    print("Перевірка завантаження даних...")
    try:
        sample = df.take(1)  # Перевіряємо, що дані є
        if sample:
            print(f"✓ Завантажено дані з {file_name} (перевірено {len(sample)} рядок)")
        else:
            raise ValueError("Помилка: Дані не завантажені!")
    except Exception as e:
        print(f"⚠ Помилка при перевірці даних: {e}")
        raise
    
    # Зберігаємо всі результати для фінального порівняння
    all_results = []
    
    # Обробляємо кожну з 3 задач класифікації
    for task_id, task_config in CLASSIFICATION_TASKS.items():
        task_name = task_config["name"]
        target_col = task_config["target_col"]
        
        print("\n" + "="*80)
        print(f"ЗАДАЧА: {task_name.upper()}")
        print(f"Бізнес-питання: {task_config['business_question']}")
        print("="*80)
        
        # Попередня обробка даних для цієї задачі
        print("\nПочаток попередньої обробки даних...")
        df_processed, feature_columns, target_col = preprocess_data(df, task_config)
        
        # Оптимізуємо партиції після обробки
        print("Оптимізація партицій після обробки...")
        df_processed = df_processed.coalesce(4)  # 4 партиції для кращої продуктивності
        
        # Перевіряємо, що є достатньо даних після обробки
        print("Перевірка кількості даних після обробки...")
        try:
            sample_check = df_processed.limit(100).count()
            if sample_check == 0:
                print(f"⚠ Помилка: Після обробки не залишилося даних для задачі {task_name}")
                continue
            print(f"✓ Дані успішно оброблені (перевірено {sample_check} рядків)")
        except Exception as e:
            print(f"⚠ Попередження: {e}")
            continue
        
        # Перевіряємо розподіл цільової змінної
        print(f"\nПеревірка розподілу цільової змінної '{target_col}'...")
        target_stats = df_processed.groupBy(target_col).agg(
            count("*").alias("count")
        ).orderBy(target_col).collect()
        
        print("Розподіл цільової змінної:")
        for row in target_stats:
            print(f"  Клас {row[target_col]}: {row['count']} прикладів")
        
        # Перевіряємо, чи є обидва класи та достатня кількість прикладів
        unique_classes = [row[target_col] for row in target_stats]
        has_both_classes = len(set(unique_classes)) > 1
        
        # Перевіряємо мінімальну кількість прикладів для меншого класу
        min_class_count = min(row['count'] for row in target_stats)
        max_class_count = max(row['count'] for row in target_stats)
        class_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
        
        min_samples_required = 100  # Мінімум 100 прикладів для меншого класу
        min_train_samples_required = 50  # Мінімум 50 прикладів у тренувальному наборі
        max_acceptable_ratio = 5.0  # Максимальне прийнятне співвідношення класів (5:1)
        
        if not has_both_classes:
            print(f"\n❌ ПОМИЛКА: Дані містять тільки один клас для задачі {task_name}!")
            print("   Модель не може бути навчена. Пропускаємо цю задачу.\n")
            continue
        elif min_class_count < min_samples_required:
            print(f"\n❌ ПОМИЛКА: Менший клас має тільки {min_class_count} прикладів (мінімум {min_samples_required})!")
            print(f"   Це занадто мало для навчання моделі (потрібно мінімум {min_train_samples_required} у тренувальному наборі).")
            print("   Пропускаємо цю задачу.\n")
            continue
        elif class_ratio > max_acceptable_ratio:
            print(f"\n⚠ УВАГА: Сильний дисбаланс класів - співвідношення {class_ratio:.1f}:1 (максимум {max_acceptable_ratio}:1)")
            print("   Застосовуємо балансування датасету перед split...")
            # Балансуємо датасет перед split
            df_processed = balance_dataset_undersampling(df_processed, target_col, max_ratio=max_acceptable_ratio, seed=42)
            df_processed = df_processed.coalesce(4)  # Оптимізуємо партиції після балансування
            
            # Перераховуємо статистику після балансування
            target_stats = df_processed.groupBy(target_col).agg(
                count("*").alias("count")
            ).orderBy(target_col).collect()
            print("\nРозподіл після балансування:")
            for row in target_stats:
                print(f"  Клас {row[target_col]}: {row['count']} прикладів")
            print()
        else:
            print(f"✓ Дані містять обидва класи - співвідношення {class_ratio:.2f}:1 (прийнятне)\n")
        
        # Розділяємо дані на тренувальний та тестовий набори (80/20) зі стратифікацією
        print("\n=== РОЗДІЛЕННЯ ДАНИХ НА ТРЕНУВАЛЬНИЙ ТА ТЕСТОВИЙ НАБОРИ ===\n")
        print("Використовується стратифікований split для забезпечення балансу класів...")
        # Забезпечуємо мінімум 1 приклад кожного класу в тестовому наборі
        train_df, test_df = stratified_split(df_processed, target_col, train_ratio=0.8, seed=42, min_test_samples_per_class=1)
        train_df = train_df.coalesce(4)  # 4 партиції для кращої продуктивності
        test_df = test_df.coalesce(4)
        
        # Перевіряємо розподіл класів у тестовому наборі
        print("\nПеревірка розподілу класів у тестовому наборі...")
        test_target_stats = test_df.groupBy(target_col).agg(
            count("*").alias("count")
        ).orderBy(target_col).collect()
        
        print("Розподіл класів у тестовому наборі:")
        for row in test_target_stats:
            print(f"  Клас {row[target_col]}: {row['count']} прикладів")
        
        test_unique_classes = [row[target_col] for row in test_target_stats]
        test_min_class_count = min(row['count'] for row in test_target_stats) if test_target_stats else 0
        
        # Перевіряємо розподіл у тренувальному наборі
        train_target_stats = train_df.groupBy(target_col).agg(
            count("*").alias("count")
        ).orderBy(target_col).collect()
        
        train_min_class_count = min(row['count'] for row in train_target_stats) if train_target_stats else 0
        
        print(f"\nРозподіл класів у тренувальному наборі:")
        for row in train_target_stats:
            print(f"  Клас {row[target_col]}: {row['count']} прикладів")
        
        if len(set(test_unique_classes)) < 2:
            print(f"\n❌ ПОМИЛКА: Тестовий набір містить тільки один клас!")
            print("   Модель не зможе бути правильно оцінена. Пропускаємо цю задачу.\n")
            continue
        
        if train_min_class_count < min_train_samples_required:
            print(f"\n❌ ПОМИЛКА: У тренувальному наборі менший клас має тільки {train_min_class_count} прикладів!")
            print(f"   Це занадто мало для навчання моделі (мінімум {min_train_samples_required}).")
            print("   Пропускаємо цю задачу.\n")
            continue
        
        if test_min_class_count < 5:
            print(f"\n⚠ УВАГА: У тестовому наборі менший клас має тільки {test_min_class_count} прикладів!")
            print("   Оцінка моделі може бути неточною, але продовжуємо.\n")
        
        # Обчислюємо та додаємо class weights для балансування незбалансованих даних
        print("\nОбчислення ваг класів для балансування...")
        class_weights = calculate_class_weights(train_df, target_col)
        
        # Додаємо ваги до тренувального набору
        train_df = add_class_weights(train_df, target_col, class_weights)
        
        # Створюємо pipeline для обробки ознак
        print("\n=== СТВОРЕННЯ PIPELINE ДЛЯ ОБРОБКИ ОЗНАК ===\n")
        if not feature_columns:
            print(f"⚠ Помилка: Не знайдено жодної ознаки для задачі {task_name}")
            continue
        print(f"✓ Використовується {len(feature_columns)} ознак")
        
        USE_SCALER = True  # Увімкнено для кращої якості моделей
        print(f"✓ StandardScaler: {'увімкнено' if USE_SCALER else 'вимкнено'} (для кращої якості)")
        feature_pipeline_stages = create_feature_pipeline(feature_columns, use_scaler=USE_SCALER)
        
        # Навчаємо 3 моделі для цієї задачі
        print("\n" + "="*80)
        print(f"НАВЧАННЯ МОДЕЛЕЙ ДЛЯ ЗАДАЧІ: {task_name}")
        print("="*80)
        
        # 1. Logistic Regression
        lr_model, lr_time = train_logistic_regression(train_df, feature_pipeline_stages, target_col, class_weights)
        
        # 2. Random Forest
        rf_model, rf_time = train_random_forest(train_df, feature_pipeline_stages, target_col, class_weights)
        
        # 3. Gradient Boosting
        gbt_model, gbt_time = train_gradient_boosting(train_df, feature_pipeline_stages, target_col, class_weights)
        
        # Оцінюємо моделі
        print("\n" + "="*80)
        print(f"ОЦІНКА ЯКОСТІ МОДЕЛЕЙ ДЛЯ ЗАДАЧІ: {task_name}")
        print("="*80)
        
        task_results = []
        
        # Оцінка Logistic Regression
        print("\n" + "-"*80)
        lr_results = evaluate_model(lr_model, test_df, f"{task_name} - Logistic Regression", target_col, feature_columns=feature_columns)
        if lr_results:
            lr_results["training_time"] = lr_time
            lr_results["task_name"] = task_name
            task_results.append(lr_results)
            all_results.append(lr_results)
        
        # Оцінка Random Forest
        print("\n" + "-"*80)
        rf_results = evaluate_model(rf_model, test_df, f"{task_name} - Random Forest", target_col, feature_columns=feature_columns)
        if rf_results:
            rf_results["training_time"] = rf_time
            rf_results["task_name"] = task_name
            task_results.append(rf_results)
            all_results.append(rf_results)
        
        # Оцінка Gradient Boosting
        print("\n" + "-"*80)
        gbt_results = evaluate_model(gbt_model, test_df, f"{task_name} - Gradient Boosting", target_col, feature_columns=feature_columns)
        if gbt_results:
            gbt_results["training_time"] = gbt_time
            gbt_results["task_name"] = task_name
            task_results.append(gbt_results)
            all_results.append(gbt_results)
        
        # Порівняння моделей для цієї задачі
        if task_results:
            print("\n" + "="*80)
            print(f"ПОРІВНЯННЯ МОДЕЛЕЙ ДЛЯ ЗАДАЧІ: {task_name}")
            print("="*80)
            compare_models(task_results)
    
    # Фінальне порівняння всіх моделей по всіх задачах
    print("\n" + "="*80)
    print("ФІНАЛЬНЕ ПОРІВНЯННЯ ВСІХ МОДЕЛЕЙ ПО ВСІХ ЗАДАЧАХ")
    print("="*80)
    if all_results:
        compare_models(all_results)
    
    # Порівняння часу навчання
    print("\n" + "="*80)
    print("ПОРІВНЯННЯ ЧАСУ НАВЧАННЯ:")
    print("="*80)
    for result in all_results:
        print(f"{result['model_name']:<50} {result['training_time']:.2f} секунд")
    
    print("\n" + "="*80)
    print("АНАЛІЗ ЗАВЕРШЕНО")
    print(f"Всього навчено та оцінено: {len(all_results)} моделей")
    print("="*80 + "\n")
    
    # Очищаємо кеш перед закриттям (якщо було кешування)
    try:
        if df.is_cached:
            df.unpersist()
        if df_processed.is_cached:
            df_processed.unpersist()
    except:
        pass
    
    spark.stop()


if __name__ == "__main__":
    main()

