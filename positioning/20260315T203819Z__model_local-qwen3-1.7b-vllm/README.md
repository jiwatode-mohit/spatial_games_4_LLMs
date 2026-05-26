# Spatial Positioning Experiment

- Timestamp (UTC): `2026-03-15T21:53:30.170154+00:00`
- Model: `local-qwen3-1.7b-vllm`
- Thinking modes: `off, on`
- Translator mode: `on`
- Cases per level: `10`
- Games: `spatialgame1_v0, spatialgame2_v0, spatialgame3_v0`
- Output dir: `/home/jiwatode/projects/nobackup/gvgai_cog/scm/GVGAI_GYM_cog/positioning/20260315T203819Z__model_local-qwen3-1.7b-vllm`

## Overall

- Cases: `300`
- Raw strict contract rate: `0.5`
- Sanitized strict contract rate: `0.6`
- Position parse rate: `0.95`
- Exact position rate: `0.07`
- Mean latency seconds: `14.775434`
- Total tokens: `924868`

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

- Best exact-position bucket: `spatialgame3_v0 lvl1 thinking=off` => `0.3`
- Worst exact-position bucket: `spatialgame1_v0 lvl3 thinking=off` => `0.0`

## Example Miss

- Game/level: `spatialgame1_v0 lvl0`
- Thinking: `off`
- True position: `(4, 1)`
- Predicted position: `(3, 2)`
- Prompt hash: `eb7165ce10b5c420`
