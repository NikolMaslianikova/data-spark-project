ARG OPENJDK_VERSION=8

FROM eclipse-temurin:${OPENJDK_VERSION}-jdk-jammy

ARG PYSPARK_VERSION=3.2.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 --no-cache-dir install pyspark==${PYSPARK_VERSION} pandas pyarrow


COPY main.py /app/main.py
COPY src/ /app/src/
WORKDIR /app

CMD ["python", "main.py"]