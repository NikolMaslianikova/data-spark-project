# Інструкції для запуску business_queries.py

## Варіант 1: Запуск через Docker (рекомендовано)

### Крок 1: Перебудувати Docker образ
Оскільки ми додали новий файл `business_queries.py`, потрібно перебудувати образ:

```bash
docker build -t my-spark-img .
```

### Крок 2: Запустити business_queries.py
Для запуску бізнес-запитів використовуйте:

**macOS/Linux:**
```bash
docker run -v /Users/yuliyatsyuvanyk/Desktop/data-spark-project/w_d_1:/app/data my-spark-img python business_queries.py
```

**Windows:**
```bash
docker run -v C:\path\to\w_d_1:/app/data my-spark-img python business_queries.py
```

## Варіант 2: Запуск окремого запиту

Якщо потрібно запустити тільки один запит, можна створити окремий скрипт або модифікувати `business_queries.py`.

Приклад для запуску тільки запиту 1:

```python
from pyspark.sql import SparkSession
from src.io_utils import load_weather_data
from business_queries import query_1_rainy_days_monsoon

spark = SparkSession.builder.appName("SingleQuery").getOrCreate()
df = load_weather_data(spark, "/app/data/*.csv", extract_city=True)
result = query_1_rainy_days_monsoon(spark, df)
result.show()
spark.stop()
```

## Варіант 3: Запуск локально (без Docker)

Якщо у вас встановлений PySpark локально:

```bash
# Переконайтеся, що ви в кореневій директорії проекту
python business_queries.py
```

Але потрібно змінити шлях до даних в коді:
```python
df = load_weather_data(spark, "/path/to/w_d_1/*.csv", extract_city=True)
```

## Примітки:

1. **Колонка 'city'**: Автоматично витягується з імені CSV файлу (наприклад, `Ahmedabad.csv` → `city = "Ahmedabad"`)

2. **Міста A-H**: Запити використовують регулярний вираз `^[A-H]` для фільтрації міст, які починаються з літер A-H. Якщо ваші міста мають інші назви, потрібно буде змінити фільтри в запитах.

3. **Колонка 'state'**: Деякі запити (наприклад, запит 6) спочатку використовували колонку 'state', але були адаптовані для роботи без неї. Якщо у вас є інформація про штати, можна додати маппінг міст до штатів.

4. **Час виконання**: Залежить від обсягу даних. Для великих наборів даних може знадобитися кілька хвилин.

## Очікуваний вивід:

Після запуску ви побачите результати всіх 6 запитів:
- Запит 1: Топ-10 міст за кількістю дощових днів у сезон мусонів
- Запит 2: Відхилення тиску між Ahmedabad та Hyderabad
- Запит 3: Статистика днів екстремальної спеки по містах
- Запит 4: Піки та провали температури в Hyderabad
- Запит 5: Порівняння штормових днів (прибережні vs внутрішні)
- Запит 6: Систематична дельта температури від середнього


