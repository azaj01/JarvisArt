# Evaluation

Evaluate image editing quality using VIEScore and pixel-level metrics.

## Quick Start

### Global Metrics

```bash
cd src/evaluation

python test_global_metrics.py \
    --model_json_path <path_to_test_json> \
    --backbone google  # or qwen25vl
```

### Local Metrics

```bash
cd src/evaluation

python test_local_metrics.py \
    --test_json_path <path_to_test_json> \
    --mask_path <path_to_mask_dir>
```


## Metrics

| Metric | Description |
|--------|-------------|
| SC | Instruction satisfaction (0-10) |
| PQ | Content consistency (0-10) |
| Overall | √(SC × PQ) (0-10) |
| L1/L2 | Pixel-level loss |

## Configuration

- **Google Gemini**: Set `GOOGLE_API_KEY` env variable
- **Qwen2.5-VL**: Requires DashScope API Key
