# Eccomas Sensor Pipeline

Pipeline ordenado para:

1. recortar la geometria en `y >= 12`
2. preparar features disponibles en inferencia
3. entrenar expertos de `Cp` por regimen de Mach
4. entrenar un encoder latente + gating
5. visualizar siempre el espacio latente
6. destilar el sensor de routing a ecuaciones simbolicas
7. ejecutar inferencia de `Cp` desde `X` crudo

## Estructura

- `main.py`: CLI principal
- `eccomas_sensor/config.py`: rutas y hiperparametros
- `eccomas_sensor/prepare_data.py`: recorte de geometria y plots
- `eccomas_sensor/features.py`: features para expertos y encoder
- `eccomas_sensor/feature_store.py`: normalizacion y guardado de features
- `eccomas_sensor/models.py`: expertos y modelo latente
- `eccomas_sensor/train_experts.py`: entrenamiento de expertos
- `eccomas_sensor/train_latent.py`: entrenamiento del encoder/gating y export del latente
- `eccomas_sensor/latent_viz.py`: plots del espacio latente
- `eccomas_sensor/sensor_distillation.py`: regresores simbolicos para aproximar el routing del sensor
- `eccomas_sensor/inference.py`: inferencia `X -> sensor -> mezcla de expertos -> Cp`

## Flujo recomendado

```bash
python eccomas_sensor_pipeline/main.py prepare-data
python eccomas_sensor_pipeline/main.py prepare-features
python eccomas_sensor_pipeline/main.py train-experts
python eccomas_sensor_pipeline/main.py train-latent
python eccomas_sensor_pipeline/main.py distill-sensor
```

Tambien existe:

```bash
python eccomas_sensor_pipeline/main.py run-all
```

## Rutas externas

Puedes pasar rutas fuera del repo directamente por comando:

```bash
python eccomas_sensor_pipeline/main.py prepare-data \
  --raw-data-dir /ruta/completa/a/raw_data \
  --cut-data-dir /ruta/completa/a/data_cut_y12 \
  --pipeline-root /ruta/completa/a/eccomas_sensor_pipeline
```

Si solo quieres pintar el espacio de diseno sin rehacer el recorte:

```bash
python eccomas_sensor_pipeline/main.py plot-design-space \
  --raw-data-dir /ruta/completa/a/raw_data \
  --cut-data-dir /ruta/completa/a/data_cut_y12
```

## Logica de features

El encoder y el sensor simbolico solo usan variables disponibles en inferencia. No usan `Cp`, ni `gradCp`, ni ninguna variable derivada de la solucion CFD.

- Features de expertos: mas ricas, para maximizar capacidad predictiva de `Cp`.
- Features de encoder: mas compactas y fisicas, para que luego gplearn pueda aproximar bien el latente.

El espacio latente se fuerza a 3D para poder visualizarlo siempre con proyecciones 2D y vistas 3D.

## Inferencia

Modo neuronal completo:

```bash
python eccomas_sensor_pipeline/main.py infer \
  --input-path /ruta/a/X.npy \
  --mode neural
```

Modo con sensor simbolico destilado:

```bash
python eccomas_sensor_pipeline/main.py infer \
  --input-path /ruta/a/X.npy \
  --mode symbolic
```

Por defecto, los resultados se guardan en `eccomas_sensor_pipeline/outputs/inference/` e incluyen:

- `cp_pred`: prediccion final de `Cp`
- `gates`: pesos del routing
- `expert_id`: experto dominante por fila
- `sensor_scores`: scores previos al softmax del sensor
- `z`: solo en modo `neural`
