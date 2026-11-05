# Reproducible environment for ACED-Bench (CPU only; agents run on remote APIs).
#   docker build -t aced-bench .
#   docker run --rm aced-bench                                   # offline tests
#   docker run --rm -v "$PWD/out:/aced/out" aced-bench bash run_experiments.sh demo
FROM python:3.12.4-slim
WORKDIR /aced
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["bash", "run_tests.sh"]
