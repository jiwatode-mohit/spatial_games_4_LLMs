# Spatial Positioning Experiment

- Timestamp (UTC): `2026-03-16T00:47:58.391677+00:00`
- Model: `local-qwen3-8b-vllm`
- Thinking modes: `off, on`
- Translator mode: `on`
- Cases per level: `10`
- Games: `spatialgame1_v0, spatialgame2_v0, spatialgame3_v0`
- Output dir: `/home/jiwatode/projects/nobackup/gvgai_cog/scm/GVGAI_GYM_cog/positioning/20260315T203819Z__model_local-qwen3-8b-vllm`

## Overall

- Cases: `300`
- Raw strict contract rate: `0.5`
- Sanitized strict contract rate: `0.79`
- Position parse rate: `0.79`
- Exact position rate: `0.306667`
- Mean latency seconds: `49.737249`
- Total tokens: `860324`

## Files

- `per_case_results.csv`
- `per_case_results.json`
- `summary_overall.csv`
- `summary_by_thinking.csv`
- `summary_by_game.csv`
- `summary_by_level.csv`
- `summary_by_game_level.csv`
- `summary_by_thinking_game_level.csv`

## Best/Worst Level Buckets

- Best exact-position bucket: `spatialgame1_v0 lvl0 thinking=on` => `0.7`
- Worst exact-position bucket: `spatialgame2_v0 lvl0 thinking=off` => `0.0`

## Example Miss

- Game/level: `spatialgame1_v0 lvl0`
- Thinking: `off`
- True position: `(4, 1)`
- Predicted position: `(4, 0)`
- Prompt hash: `8a1e238cb8684e21`
