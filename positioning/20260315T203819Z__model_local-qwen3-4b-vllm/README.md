# Spatial Positioning Experiment

- Timestamp (UTC): `2026-03-15T23:35:35.236878+00:00`
- Model: `local-qwen3-4b-vllm`
- Thinking modes: `off, on`
- Translator mode: `on`
- Cases per level: `10`
- Games: `spatialgame1_v0, spatialgame2_v0, spatialgame3_v0`
- Output dir: `/home/jiwatode/projects/nobackup/gvgai_cog/scm/GVGAI_GYM_cog/positioning/20260315T203819Z__model_local-qwen3-4b-vllm`

## Overall

- Cases: `300`
- Raw strict contract rate: `0.5`
- Sanitized strict contract rate: `0.92`
- Position parse rate: `0.92`
- Exact position rate: `0.15`
- Mean latency seconds: `35.149847`
- Total tokens: `1021996`

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

- Best exact-position bucket: `spatialgame1_v0 lvl2 thinking=off` => `0.5`
- Worst exact-position bucket: `spatialgame2_v0 lvl0 thinking=off` => `0.0`

## Example Miss

- Game/level: `spatialgame1_v0 lvl0`
- Thinking: `on`
- True position: `(4, 1)`
- Predicted position: `(3, 1)`
- Prompt hash: `47b1fccc41bfc7ed`
