# Environments

No single environment can run every model: their reference code pins incompatible `transformers` versions.
Install the base package (`uv pip install -e .`) into each environment, plus one of the files below.

| File | Models |
|---|---|
| `qwen3vl_mimovl.txt` | Qwen3-VL-8B-Instruct, MiMo-VL-7B-RL |
| `internvl25.txt` | InternVL2.5-8B |
| `qwenvl.txt` | Qwen-VL-Chat |
| `minigptv2.txt` | MiniGPT-v2 |
| none: use each model's own repository | CogVLM, Shikra, Ferret |
