# Eccomas Full Aircraft

Pipeline activa para trabajar con el dataset ONERA completo usando una representación reducida de superficie.

## Objetivo

Pasar de `260,774` puntos raw por condición a una superficie simplificada y reutilizable, y a partir de ahí montar una pipeline completa:

1. preparar la referencia de superficie
2. extraer arrays reducidos por condición
3. construir features
4. entrenar expertos
5. entrenar el teacher latent/MoE
6. destilar el sensor simbólico
7. hacer inferencia
8. opcionalmente, refinar el `Cp` con una rama residual de difusión

## Comandos

```bash
source .venv/bin/activate

python eccomas_full_aircrafts/main.py inspect-raw

python eccomas_full_aircrafts/main.py prepare-reference-surface \
  --reference-split train \
  --reference-condition-index 0 \
  --x-bins 1080 \
  --y-bins 540

python eccomas_full_aircrafts/main.py prepare-reduced-data \
  --surface upper

python eccomas_full_aircrafts/main.py prepare-features

python eccomas_full_aircrafts/main.py explore-dataset

python eccomas_full_aircrafts/main.py train-experts

python eccomas_full_aircrafts/main.py train-latent

python eccomas_full_aircrafts/main.py distill-sensor

python eccomas_full_aircrafts/main.py infer \
  --mode symbolic

python eccomas_full_aircrafts/main.py train-diffusion \
  --baseline-mode symbolic

python eccomas_full_aircrafts/main.py infer-diffusion \
  --split test \
  --baseline-mode symbolic
```

## Qué queda guardado

- `outputs/surfaces/`
  Referencia simplificada `upper/lower`
- `outputs/reduced_data/`
  Arrays reducidos `X_cut_*.npy`, `Y_cut_*.npy`
- `outputs/features/`
  Features estandarizadas para expertos y gate
- `outputs/models/`
  Expertos y `latent_sensor_moe.pth`
- `outputs/sensor/`
  Sensor simbólico híbrido
- `outputs/inference/`
  Predicciones `neural` o `symbolic`
- `outputs/diffusion/`
  Configuración y artefactos de la rama residual de difusión
- `results/`
  Figuras de referencia y `Cp fields` raw
- `exploration_data/`
  Caracterización del dataset, diseño experimental y rangos de `Cp`

## Notas

- La ruta de entrenamiento actual está pensada para una sola cara de referencia cada vez. El default es `upper`.
- `plot-raw-fields` ya soporta `points` y `surface`, pero la visualización principal recomendada sigue siendo `points`.
- `train-diffusion` usa como baseline una predicción `symbolic` o `neural`; si no existe el `.npz` base para `train/test`, lo genera automáticamente con `infer`.
