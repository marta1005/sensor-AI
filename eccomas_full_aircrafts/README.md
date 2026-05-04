# Eccomas Full Aircraft

Pipeline activa para el avión completo reducida a una sola cara (`upper` o `lower`) con una arquitectura espacial centrada en choque.

## Idea actual

La ruta activa ya no es el `MoE` por condición. Ahora trabajamos con:

1. una superficie reducida y reutilizable
2. features espaciales sobre esa superficie
3. un `teacher` global con:
   - backbone `U-Net`
   - cuello `latente`
   - dos cabezas locales: `smooth` y `shock`
4. un sensor simbólico local que destila el `routing` del teacher
5. una mezcla final:
   - `Cp = (1 - alpha) * Cp_smooth + alpha * Cp_shock`

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

python eccomas_full_aircrafts/main.py explore-dataset \
  --reduced-surface upper

python eccomas_full_aircrafts/main.py prepare-features \
  --reduced-surface upper

python eccomas_full_aircrafts/main.py train-shock-experts \
  --reduced-surface upper

python eccomas_full_aircrafts/main.py distill-shock-sensor \
  --reduced-surface upper

python eccomas_full_aircrafts/main.py infer-shock-symbolic \
  --reduced-surface upper
```

## Qué se guarda

- `outputs/surfaces/`
  Referencia simplificada `upper/lower`
- `outputs/<surface>/reduced_data/`
  Arrays reducidos `X_cut_*.npy`, `Y_cut_*.npy`
- `outputs/<surface>/features/`
  Features estandarizadas y targets auxiliares
- `outputs/<surface>/models/`
  Checkpoint del `teacher` latente `smooth/shock`
- `outputs/<surface>/sensor/`
  Sensor simbólico local destilado
- `outputs/<surface>/inference/`
  Predicciones `shock_symbolic`
- `outputs/<surface>/metrics/`
  Métricas del teacher y del sensor
- `results/`
  Figuras de `Cp`
- `exploration_data/`
  Caracterización del dataset

## Plots

```bash
python eccomas_full_aircrafts/main.py plot-inference-cp \
  --reduced-surface upper \
  --split test \
  --layout truth-pred-error \
  --view top \
  --condition-indices 79

python eccomas_full_aircrafts/main.py plot-inference-cp-grid \
  --reduced-surface upper \
  --split test \
  --view top \
  --condition-indices 2 65 79 39
```

Si no pasas `--prediction-path`, los plots buscan por defecto:

- `outputs/<surface>/inference/X_cut_<split>_shock_symbolic.npz`

## Nota práctica

La lectura importante del sensor simbólico ya no es “qué experto global elijo para una condición”, sino “dónde aparece una región de choque en la superficie y cuánto debe mandar la cabeza `shock` frente a la `smooth`”.
