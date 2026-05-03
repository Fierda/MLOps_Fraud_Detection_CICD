from prometheus_client import start_http_server, Counter, Histogram, Gauge, Info
import time
import random
import psutil
import os

# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total requests to the model",
    ["method", "endpoint"],
)
FRAUD_DETECTED = Counter(
    "fraud_detected_total",
    "Total detected fraud transactions",
)
NORMAL_DETECTED = Counter(
    "normal_detected_total",
    "Total detected normal transactions",
)
ERROR_RATE = Counter(
    "http_error_responses_total",
    "Total error responses (4xx/5xx)",
    ["status_code"],
)

MODEL_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency time (seconds)",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0],
)

MEMORY_USAGE = Gauge(
    "system_memory_usage_bytes",
    "System memory usage (bytes)",
)
CPU_USAGE = Gauge(
    "system_cpu_usage_percent",
    "System CPU usage (%)",
)
PREDICTION_CONFIDENCE = Gauge(
    "model_prediction_confidence",
    "Confidence score of the last prediction",
)
ACTIVE_SESSIONS = Gauge(
    "active_user_sessions",
    "Number of active inference sessions",
)
DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Input data distribution anomaly score (0-1)",
)

MODEL_INFO = Info(
    "model",
    "Running Model Information",
)


def get_system_metrics():
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        return cpu, mem.used
    except Exception:
        return random.uniform(20.0, 85.0), random.uniform(500e6, 2e9)


def simulate_inference():

    MODEL_INFO.info({
        "name": "FraudModel",
        "version": "1",
        "framework": "scikit-learn",
        "developer": "Fierda",
    })

    print("[INFO] Starting Inference Simulation. Ctrl+C for stop.")

    while True:

        with MODEL_LATENCY.time():
            time.sleep(random.uniform(0.02, 0.15))

        REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
        ACTIVE_SESSIONS.set(random.randint(10, 100))

        cpu, mem = get_system_metrics()
        CPU_USAGE.set(cpu)
        MEMORY_USAGE.set(mem)

        if random.random() < 0.01:
            status = random.choice(["400", "500", "503"])
            ERROR_RATE.labels(status_code=status).inc()

        confidence = random.uniform(0.5, 0.99)
        PREDICTION_CONFIDENCE.set(confidence)

        if random.random() < 0.05:
            FRAUD_DETECTED.inc()
            DATA_DRIFT_SCORE.set(random.uniform(0.7, 1.0))
        else:
            NORMAL_DETECTED.inc()
            DATA_DRIFT_SCORE.set(random.uniform(0.05, 0.3))

        time.sleep(1)


if __name__ == "__main__":
    port = 8000
    start_http_server(port)
    print(f"[OK] Prometheus Exporter running on http://localhost:{port}/metrics")
    simulate_inference()
