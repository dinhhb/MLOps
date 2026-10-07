# End-to-End MLOps Pipeline

A reference MLOps pipeline that takes a ML model from versioned code and data all the way to a monitored production endpoint.

## Pipeline Overview

| # | Stage | Tool | Purpose |
|---|-------|------|---------|
| 1 | Version control | git + DVC | Code and data/model versions tracked |
| 2 | Experiment tracking | MLflow | Every training run's params/metrics logged |
| 3 | Containerization | Docker | Best run's model packaged, reproducible anywhere |
| 4 | CI/CD | GitHub Actions | Tests, builds, and pushes the container automatically |
| 5 | Serving | FastAPI | Container deployed as a live prediction endpoint |
| 6 | Monitoring | Evidently AI | Watches live traffic for data or concept drift |
| – | Orchestration | Airflow / Prefect | Schedules and chains stages 1–6 |
| – | Cloud platform | Azure / AWS / GCP | Managed home for stages 1–6 |


## Getting Started

### Prerequisites
- Python 3.10+
- Git and [DVC](https://dvc.org/)
- Docker
- A GitHub repository (for Actions)
- Access to a cloud account (Azure, AWS, or GCP)

### Setup

```bash
# Clone the repository
git clone https://github.com/dinhhb/MLOps.git
cd MLOps

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull versioned data and models
dvc pull
```

### Train and track experiments

```bash
mlflow ui                 # start the tracking UI at http://localhost:5000 or http://127.0.0.1:5000
python train.py       # logs params/metrics to MLflow
```

### Build and run the container

```bash
docker build -t mlops-model:latest .
docker run -p 8000:8000 mlops-model:latest
```

### Query the prediction endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

### Run monitoring

```bash
python src/monitor.py     # generates an Evidently drift report
```


## CI/CD

The GitHub Actions workflow automatically:

1. Installs dependencies and runs the test suite
2. Builds the Docker image
3. Pushes the image to the container registry

## Retraining Loop

Monitoring runs continuously against live traffic. When Evidently flags data or concept drift, the orchestrator triggers a new training run, which flows back through steps 1–6 and redeploys an updated model.


