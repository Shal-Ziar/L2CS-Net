# Collect only (face detector, no gaze model)
uv run python3 -m personal_trainer collect --windowed

# Fine-tune on collected session
uv run python3 -m personal_trainer finetune \
  --snapshot models/l2cs_gaze360_resnet50.safetensors \
  --data-dir training_data/<session-id>

# Full pipeline in one shot
uv run python3 -m personal_trainer run \
  --snapshot models/l2cs_gaze360_resnet50.safetensors \
  --epochs 10 --lr 1e-5
