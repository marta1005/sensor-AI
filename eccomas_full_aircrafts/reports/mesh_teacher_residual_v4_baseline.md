# Mesh Teacher Residual V4 Baseline

Snapshot date: 2026-06-12

## Model

- Surface: `upper`
- Architecture: `mesh_teacher_cp_shock_residual_v4`
- Projection: 2D `x-y`
- Edge projection: `xy`
- Shock residual: enabled
- Hidden dimension: `96`
- Latent dimension: `4`
- Message passing steps: `6`
- Points per condition: `54,294`
- Grid size: `426 x 1093`

## Test Metrics

From `eccomas_full_aircrafts/outputs/upper/metrics/mesh_teacher_training.json`:

| Metric | Value |
| --- | ---: |
| final_test_cp_mae | 0.052396 |
| cp_rmse | 0.128600 |
| cp_mae_shock_zone | 0.090608 |
| cp_mae_smooth_zone | 0.045693 |
| shock_mae | 0.041712 |
| shock_precision | 0.851419 |
| shock_recall | 0.931451 |
| shock_iou | 0.802650 |
| shock_f1 | 0.889616 |

From `eccomas_full_aircrafts/outputs/upper/metrics/mesh_pipeline_diagnostics_test.json`:

| Metric | Value |
| --- | ---: |
| physical_mae_mean | 0.018817 |
| physical_mae_median | 0.011441 |
| physical_mae_max | 0.107588 |
| physical_rmse_mean | 0.040815 |
| physical_shock_zone_mae_mean | 0.032541 |
| physical_smooth_zone_mae_mean | 0.016410 |

## Worst Test Conditions

| Condition | Mach | AoA | Pi | Regime | MAE | Shock-zone MAE | Smooth-zone MAE |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 111 | 0.50 | 12.5 | 4.0 | subsonic | 0.107588 | 0.211117 | 0.089671 |
| 7 | 0.50 | 12.5 | 1.0 | subsonic | 0.089831 | 0.193752 | 0.071334 |
| 55 | 0.30 | 12.5 | 2.0 | subsonic | 0.078459 | 0.218813 | 0.055239 |
| 39 | 0.88 | 8.0 | 1.0 | supersonic | 0.074337 | 0.154038 | 0.060804 |
| 135 | 0.85 | 9.0 | 4.0 | supersonic | 0.067217 | 0.144136 | 0.053515 |

## Local Artifacts

These artifacts are intentionally ignored by Git because they are generated outputs:

- `eccomas_full_aircrafts/outputs/upper/models/mesh_teacher.pth`
- `eccomas_full_aircrafts/outputs/upper/models/mesh_teacher_config.json`
- `eccomas_full_aircrafts/outputs/upper/inference/X_cut_test_mesh_teacher.npz`
- `eccomas_full_aircrafts/outputs/upper/metrics/mesh_teacher_training.json`
- `eccomas_full_aircrafts/outputs/upper/metrics/mesh_pipeline_diagnostics_test.json`
- `eccomas_full_aircrafts/results/upper/mesh_diagnostics_test_worst.png`
- `eccomas_full_aircrafts/results/upper/cp_full_aircraft_inference_tb_upper_test_truth_pred_error_grid_2_39_65_79.png`

## Interpretation

This baseline is presentable: the residual mesh teacher improves the average physical test MAE and keeps a strong auxiliary shock detector. The remaining bottleneck is concentrated in high-AoA conditions, especially subsonic cases with `AoA=12.5`, where shock-zone error is still much higher than smooth-zone error.
