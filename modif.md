# Modifications

## 2026-03-21 — Cleanup

| File | What was removed |
|------|-----------------|
| `utils/util_models.py` | 4 dead classes (`AutoencoderFlat_old`, `Autoencoder0`, `AutoencoderConv`, `AutoencoderConvV2`), duplicate/unused imports |
| `utils/util_data.py` | `get_system_new()`, old `introduce_overlap()` + its private helpers, `__main__` block |
| `utils/util_dev.py` | Dead `test_maps()` function |
| `utils/fl_client.py` | Fixed broken import (`utils.   util_models` → `utils.util_models`), removed unused imports (`argparse`, `pathlib`, `matplotlib`, `pebble`) |
| `utils/fl_test.py` | Deleted (broken file with wrong import paths) |
| `requirements-old.txt` | Deleted |

## 2026-03-21 — main.py refactor

- Removed `parse_arguments()` and `merge_settings()` (argparse removed)
- `main(config=None)` now takes an optional dict, merged on top of `my_config.settings`
- `my_config.py` is the sole source of defaults
