# Predicción de falla cardíaca · Proyecto integrador de MLOps

Modelo de clasificación binaria que estima la presencia de enfermedad cardíaca
(`HeartDisease`) a partir de variables clínicas. Dataset: *Heart Failure Prediction*
(Kaggle, 918 pacientes). Incluye análisis sin *data leakage*, API con FastAPI, Docker,
Kubernetes (Minikube), integración continua y monitoreo de deriva con Evidently.

## Resultados

| Etapa | Resultado |
|---|---|
| Fuga de datos (SVC, AUC de prueba) | A (columna filtrada): 1.000 · B (escalado previo): 0.928 · C (Pipeline correcto): 0.914 |
| Mejor modelo (AUC en validación cruzada) | RandomForest, 0.931 |
| Evaluación final (prueba) | AUC 0.931 · accuracy 0.875 · recall de enfermo 0.92 |
| Matriz de confusión (prueba) | 67 VN · 15 FP · 8 FN · 94 VP |
| Deriva, entrenamiento vs prueba | 0 de 11 variables |
| Deriva, población simulada | Age, Cholesterol y MaxHR |

## Estructura

```
app/api.py              API REST (FastAPI)
app/model.joblib        Pipeline entrenado (preprocesamiento + RandomForest)
docker/                 Dockerfile y requirements.txt
k8s/                    Deployment y Service de Kubernetes
tests/test_api.py       Pruebas con pytest
.github/workflows/      Integración continua (flake8 + pytest)
```

## Cómo ejecutar

Requiere Python 3.13 (el modelo se entrenó con scikit-learn 1.6.1).

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r docker/requirements.txt pytest flake8 httpx
uvicorn app.api:app --reload      # documentación en http://127.0.0.1:8000/docs
pytest                            # pruebas
```

### Docker

```bash
docker build -t heart-api -f docker/Dockerfile .
docker run --rm -p 8000:8000 heart-api
```

### Kubernetes (Minikube)

```bash
minikube start --driver=docker --memory=3000 --cpus=2
minikube image load heart-api:latest
minikube kubectl -- apply -f k8s/
minikube service heart-service --url
```

## Prevención de data leakage

* La división entrenamiento/prueba se hace antes de cualquier transformación.
* Imputación, escalado y codificación viven dentro de un `Pipeline`, así que se ajustan
  solo con los pliegues de entrenamiento en cada paso de la validación cruzada.
* En `Cholesterol` y `RestingBP`, los ceros (valores faltantes disfrazados) se convierten
  en `NaN` y el Pipeline los imputa; la API aplica la misma regla.
