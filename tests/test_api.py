from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)

BAJO_RIESGO = {
    "Age": 40, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 120,
    "Cholesterol": 190, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 172,
    "ExerciseAngina": "N", "Oldpeak": 0, "ST_Slope": "Up",
}

ALTO_RIESGO = {
    "Age": 62, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140,
    "Cholesterol": 268, "FastingBS": 0, "RestingECG": "LVH", "MaxHR": 106,
    "ExerciseAngina": "Y", "Oldpeak": 3.6, "ST_Slope": "Down",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_formato_de_respuesta():
    r = client.post("/predict", json=BAJO_RIESGO)
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {"probabilidad_enfermedad", "prediccion"}
    assert 0 <= cuerpo["probabilidad_enfermedad"] <= 1
    assert cuerpo["prediccion"] in (0, 1)


def test_alto_riesgo_tiene_mayor_probabilidad_que_bajo():
    bajo = client.post("/predict", json=BAJO_RIESGO).json()
    alto = client.post("/predict", json=ALTO_RIESGO).json()
    assert alto["probabilidad_enfermedad"] > bajo["probabilidad_enfermedad"]


def test_ceros_se_tratan_como_faltantes():
    paciente = dict(BAJO_RIESGO, Cholesterol=0, RestingBP=0)
    r = client.post("/predict", json=paciente)
    assert r.status_code == 200


def test_colesterol_ausente_es_valido():
    paciente = dict(BAJO_RIESGO)
    del paciente["Cholesterol"]
    r = client.post("/predict", json=paciente)
    assert r.status_code == 200


def test_valor_invalido_devuelve_422():
    r = client.post("/predict", json=dict(BAJO_RIESGO, Sex="X"))
    assert r.status_code == 422


def test_campo_obligatorio_faltante_devuelve_422():
    paciente = dict(BAJO_RIESGO)
    del paciente["Age"]
    r = client.post("/predict", json=paciente)
    assert r.status_code == 422
