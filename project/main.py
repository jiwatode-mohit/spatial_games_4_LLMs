import os
import json
import pandas as pd
import gym_gvgai as gvgai
from datetime import datetime
import argparse
import re
import csv
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import gc
import psutil
import signal
import sys
import traceback
import shutil

from llm.agent.llm_agent import LLMPlayer
from llm.agent.llm_translator import LLMTranslator
from llm.utils.agent_components import show_state_gif, generate_mapping_and_ascii, extract_avatar_position_from_state
from llm.utils.vgdl_utils import load_level_map, load_vgdl_rules
from dotenv import load_dotenv
from llm.utils.config import get_profile_config
from SCM import SCMManager, mapped_ascii_for_scm

_VERBOSITY_LEVELS = {"quiet": 0, "info": 1, "debug": 2}
_CURRENT_VERBOSITY = "info"

def set_logging_verbosity(level: str) -> None:
    global _CURRENT_VERBOSITY
    lvl = (level or "info").strip().lower()
    if lvl not in _VERBOSITY_LEVELS:
        lvl = "info"
    _CURRENT_VERBOSITY = lvl
    os.environ["GVGAI_LOG_VERBOSITY"] = lvl

def log_msg(level: str, message: str) -> None:
    if _VERBOSITY_LEVELS.get(_CURRENT_VERBOSITY, 1) >= _VERBOSITY_LEVELS.get(level, 1):
        print(message)

# --- Helper functions for managing run directories (inspired by llm_agent_loop.py) ---
def get_game_name_simple(env_name_full):
    """Extracts a simplified game name like 'zelda-lvl1' from 'gvgai-zelda-lvl1-v0'."""
    match = re.search(r'gvgai-(.*?)-v0', env_name_full)
    if match:
        return match.group(1)
    return env_name_full # Fallback

def get_model_name_simple(model_name_full):
    """Returns the model name, handling 'portkey-' prefix for directory naming."""
    return model_name_full

def resolve_effective_mode(
    model_name_full: str,
    mode: str,
    qwen_thinking: str,
    translator_mode: str = "on",
    causal_mode: str = "off",
    scm_bootstrap: str = "off",
    planning_mode: str = "off",
    planning_horizon_x: int = 1,
) -> str:
    """
    Build a mode label used for run-path identity.
    For local Qwen3-on-vLLM runs, include thinking mode to avoid collisions.
    """
    try:
        profile = get_profile_config(model_name_full)
        client_type = profile.get("client_type")
        resolved_model = str(profile.get("model", "")).lower()
        is_qwen3_vllm = (client_type == "vllm") and ("qwen3" in resolved_model)
    except Exception:
        is_qwen3_vllm = False

    suffix = (
        f"__translator-{translator_mode}__causal-{causal_mode}"
        f"__scm-bootstrap-{scm_bootstrap}"
        f"__planning-{planning_mode}__planx-{int(planning_horizon_x)}"
    )
    if is_qwen3_vllm:
        return f"{mode}__thinking-{qwen_thinking}{suffix}"
    return f"{mode}{suffix}"

def is_qwen3_vllm_profile(model_name_full: str) -> bool:
    try:
        profile = get_profile_config(model_name_full)
        client_type = profile.get("client_type")
        resolved_model = str(profile.get("model", "")).lower()
        return (client_type == "vllm") and ("qwen3" in resolved_model)
    except Exception:
        return False

def resolve_qwen_sampling_overrides(model_name_full: str, qwen_thinking: str) -> dict:
    """
    Apply recommended decoding presets for local Qwen3-vLLM.
    - thinking(on): temperature=0.6, top_p=0.95, top_k=20, min_p=0
    - non-thinking(off): temperature=0.7, top_p=0.8, top_k=20, min_p=0
    - auto: no forced override
    """
    if not is_qwen3_vllm_profile(model_name_full):
        return {}

    if qwen_thinking == "on":
        return {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        }
    if qwen_thinking == "off":
        return {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        }
    return {}

def get_run_dir_path(base_dir, model_name_full, env_name_full, mode, run_id):
    """Get the directory path for a specific run, including the mode."""
    model_simple = get_model_name_simple(model_name_full)
    game_simple = get_game_name_simple(env_name_full)
    return os.path.join(base_dir, model_simple, game_simple, mode, f"run_{run_id}")

def check_run_dir_is_taken(base_dir, model_name_full, env_name_full, mode, run_id):
    """Check if the specified run directory is taken (has a completed benchmark_analysis.json or is currently running)."""
    run_dir = get_run_dir_path(base_dir, model_name_full, env_name_full, mode, run_id)
    analysis_file_path = os.path.join(run_dir, "benchmark_analysis.json")
    temp_file_path = os.path.join(run_dir, ".running_temp")
    
    # A run is "taken" if:
    # 1. The benchmark_analysis.json directory exists (indicating successful completion)
    # 2. The .running_temp file exists (indicating another worker is currently running this task)
    return os.path.exists(analysis_file_path) or os.path.isfile(temp_file_path)

def find_next_available_run_id(base_dir, model_name_full, env_name_full, mode, initial_run_id):
    """Find the next available run ID where no successful completion exists (no benchmark_analysis.json)."""
    run_id = initial_run_id
    while check_run_dir_is_taken(base_dir, model_name_full, env_name_full, mode, run_id):
        print(f"Run_id {run_id} (Game: {env_name_full}, Model: {model_name_full}, Mode: {mode}) already has successful completion. Trying next run ID.")
        run_id += 1
    return run_id

def check_memory_usage(threshold_percent=85):
    """Check if memory usage is above threshold"""
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > threshold_percent:
        print(f"WARNING: High memory usage detected: {memory_percent}%")
        return True
    return False

def safe_cleanup(obj, obj_name="object"):
    """Safely cleanup an object with error handling"""
    try:
        if obj is not None:
            if hasattr(obj, 'close'):
                obj.close()
            elif hasattr(obj, 'cleanup'):
                obj.cleanup()
            del obj
    except Exception as e:
        print(f"Warning: Error cleaning up {obj_name}: {e}")

def sanitize_for_filename(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(value))

def backup_existing_run_artifacts(run_dir: str, backup_root: str, run_context: dict) -> str:
    """
    Move all existing artifacts in run_dir into a timestamped backup directory.
    Returns the backup directory path when a move occurred, otherwise an empty string.
    """
    if not os.path.isdir(run_dir):
        return ""
    existing_entries = [name for name in os.listdir(run_dir) if name not in {".", ".."}]
    if not existing_entries:
        return ""

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(
        backup_root,
        f"{ts}__{sanitize_for_filename(run_context.get('env_name_full', 'env'))}"
        f"__{sanitize_for_filename(run_context.get('model_name_full', 'model'))}"
        f"__{sanitize_for_filename(run_context.get('run_mode', 'mode'))}"
        f"__run_{int(run_context.get('run_id', 0))}",
    )
    os.makedirs(backup_dir, exist_ok=True)

    moved_any = False
    for name in existing_entries:
        src = os.path.join(run_dir, name)
        dst = os.path.join(backup_dir, name)
        shutil.move(src, dst)
        moved_any = True
    return backup_dir if moved_any else ""

def compute_token_aggregates_from_step_log(step_log):
    sum_input = 0
    sum_output = 0
    sum_total = 0
    num_with_usage = 0
    num_estimated = 0
    num_unavailable = 0
    sum_step_time_seconds = 0.0

    for entry in step_log or []:
        source = entry.get("token_usage_source", "unavailable")
        in_tok = int(entry.get("input_tokens", 0) or 0)
        out_tok = int(entry.get("output_tokens", 0) or 0)
        tot_tok = int(entry.get("total_tokens", in_tok + out_tok) or 0)
        step_time_seconds = float(entry.get("step_time_seconds", 0.0) or 0.0)

        if source != "unavailable":
            num_with_usage += 1
        if source == "estimated":
            num_estimated += 1
        if source == "unavailable":
            num_unavailable += 1

        sum_input += in_tok
        sum_output += out_tok
        sum_total += tot_tok
        sum_step_time_seconds += step_time_seconds

    steps = len(step_log or [])
    coverage = (num_with_usage / steps) if steps > 0 else 0.0
    mean_step_time_seconds = (sum_step_time_seconds / steps) if steps > 0 else 0.0
    return {
        "sum_input_tokens": sum_input,
        "sum_output_tokens": sum_output,
        "sum_total_tokens": sum_total,
        "sum_step_time_seconds": round(sum_step_time_seconds, 6),
        "mean_step_time_seconds": round(mean_step_time_seconds, 6),
        "num_steps_with_usage": num_with_usage,
        "num_steps_estimated": num_estimated,
        "num_steps_unavailable": num_unavailable,
        "token_coverage_ratio": round(coverage, 6),
    }

def write_results_index_record(record: dict, results_index_dir: str):
    index_dir = Path(results_index_dir)
    records_dir = index_dir / "records"
    manifests_dir = index_dir / "manifests"
    records_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    record_name = (
        f"{ts}__{sanitize_for_filename(record.get('model_profile', 'unknown'))}"
        f"__{sanitize_for_filename(record.get('game', 'unknown'))}"
        f"__{sanitize_for_filename(record.get('mode', 'unknown'))}"
        f"__run_{record.get('run_id', '0')}"
        f"__job_{sanitize_for_filename(record.get('slurm_job_id', 'none'))}"
        f"__arr_{sanitize_for_filename(record.get('slurm_array_task_id', 'none'))}.json"
    )
    record_path = records_dir / record_name
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    lock_path = manifests_dir / ".lock"
    jsonl_path = manifests_dir / "runs.jsonl"
    csv_path = manifests_dir / "runs.csv"
    key = f"{record.get('game_env_id')}|{record.get('model_profile')}|{record.get('mode')}|{record.get('run_id')}"
    record = dict(record)
    record["record_path"] = str(record_path)
    record["record_key"] = key
    fieldnames = list(record.keys())

    with open(lock_path, "w", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp, fcntl.LOCK_EX)

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

        rows = []
        if csv_path.exists():
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_fields = reader.fieldnames or []
                if existing_fields:
                    fieldnames = list(dict.fromkeys(existing_fields + fieldnames))
                rows = list(reader)

        updated = False
        for idx, row in enumerate(rows):
            row_key = f"{row.get('game_env_id')}|{row.get('model_profile')}|{row.get('mode')}|{row.get('run_id')}"
            if row_key == key:
                rows[idx] = {k: record.get(k, "") for k in fieldnames}
                updated = True
                break
        if not updated:
            rows.append({k: record.get(k, "") for k in fieldnames})

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

def run_single_game_task(env_name_full: str, mode: str, model_name_full: str, requested_run_id: int,
                         base_output_dir: str, max_steps: int, results_index_dir: str,
                         portkey_virtual_key: str = None,
                         force_rerun: bool = False,
                         qwen_thinking: str = "auto",
                         effective_mode: str = None,
                         translator_mode: str = "on",
                         causal_mode: str = "off",
                         scm_bootstrap: str = "off",
                         scm_blueprint_path: str = "scm_blueprint.py",
                         planning_mode: str = "off",
                         planning_horizon_x: int = 1,
                         shuffle_action_meanings: str = "off",
                         action_shuffle_seed: int = 42):
    """
    Worker function to run a single game instance with improved error handling and resource management.
    """
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"Process {os.getpid()}: Received signal {signum}, attempting graceful shutdown...")
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    actual_model_for_agent = model_name_full
    run_mode = effective_mode or resolve_effective_mode(
        model_name_full,
        mode,
        qwen_thinking,
        translator_mode=translator_mode,
        causal_mode=causal_mode,
        scm_bootstrap=scm_bootstrap,
        planning_mode=planning_mode,
        planning_horizon_x=planning_horizon_x,
    )
    qwen_sampling_overrides = resolve_qwen_sampling_overrides(model_name_full, qwen_thinking)
    
    # Log information about virtual key usage for this task
    if portkey_virtual_key:
        print(f"Process {os.getpid()}: Task for {model_name_full} received specific virtual key.")
    elif model_name_full.startswith("portkey-") or model_name_full == "gemini":
        print(f"Warning: Task for Portkey-type model {model_name_full} did not receive a specific virtual key.")
    
    actual_run_id_to_use: int = requested_run_id
    if force_rerun:
        print(f"Force rerun enabled. Using run_id: {actual_run_id_to_use} for {env_name_full}, {model_name_full}, {run_mode}.")
    else:
        print(f"Using requested run_id: {actual_run_id_to_use}")

    run_dir = get_run_dir_path(base_output_dir, model_name_full, env_name_full, run_mode, actual_run_id_to_use)
    temp_file_path = os.path.join(run_dir, ".running_temp")
    run_started_at = datetime.now()

    os.makedirs(run_dir, exist_ok=True)
    backup_root = os.path.join("backups", "run_artifacts")
    backup_path = backup_existing_run_artifacts(
        run_dir=run_dir,
        backup_root=backup_root,
        run_context={
            "env_name_full": env_name_full,
            "model_name_full": model_name_full,
            "run_mode": run_mode,
            "run_id": actual_run_id_to_use,
        },
    )
    if backup_path:
        print(f"Moved existing run artifacts to backup: {backup_path}")

    # Create temporary file to mark this task as running
    try:
        with open(temp_file_path, 'w') as f:
            f.write(f"Started at: {run_started_at.isoformat()}\n")
            f.write(f"Process ID: {os.getpid()}\n")
            f.write(f"Game: {env_name_full}\n")
            f.write(f"Model: {model_name_full}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Effective Mode: {run_mode}\n")
            f.write(f"Qwen Thinking: {qwen_thinking}\n")
            f.write(f"Qwen Sampling Overrides: {json.dumps(qwen_sampling_overrides, ensure_ascii=True)}\n")
            f.write(f"Run ID: {actual_run_id_to_use}\n")
            f.write(f"Translator Mode: {translator_mode}\n")
            f.write(f"Causal Mode: {causal_mode}\n")
            f.write(f"SCM Bootstrap: {scm_bootstrap}\n")
            f.write(f"SCM Blueprint Path: {scm_blueprint_path}\n")
            f.write(f"Planning Mode: {planning_mode}\n")
            f.write(f"Planning Horizon x: {planning_horizon_x}\n")
            f.write(f"Shuffle Action Meanings: {shuffle_action_meanings}\n")
            f.write(f"Action Shuffle Seed: {action_shuffle_seed}\n")
        print(f"Created temporary file: {temp_file_path}")
    except Exception as e:
        print(f"Warning: Could not create temporary file {temp_file_path}: {e}")
    
    print(
        f"\n==== Starting game: {env_name_full} | Mode: {mode} | Effective Mode: {run_mode} "
        f"| Model: {model_name_full} | Qwen Thinking: {qwen_thinking} | Run: {actual_run_id_to_use} "
        f"| translator={translator_mode} | causal={causal_mode} | scm_bootstrap={scm_bootstrap} "
        f"| planning={planning_mode} | planx={planning_horizon_x} ===="
    )
    if qwen_sampling_overrides:
        print(f"[QWEN3] Using sampling overrides: {qwen_sampling_overrides}")

    # Initialize variables for proper cleanup
    env = None
    gif_saver = None
    player = None
    translator = None
    scm_manager = None
    scm_summary = {}
    step_count = 0
    info = {}
    game_successful = False
    result_summary = ""
    error_type = None
    error_message = None
    
    try:
        # Check memory before starting
        if check_memory_usage(80):
            gc.collect()  # Force garbage collection
        
        # Create environment with timeout and error handling
        try:
            env = gvgai.make(env_name_full)
            if env is None:
                raise RuntimeError(f"Failed to create environment for {env_name_full}")
        except Exception as e:
            raise RuntimeError(f"Environment creation failed for {env_name_full}: {e}")
        
        # Load VGDL rules with error handling
        try:
            vgdl_rules = load_vgdl_rules(env_name_full)
            if not vgdl_rules:
                print(f"Warning: Empty VGDL rules for {env_name_full}")
        except Exception as e:
            print(f"Warning: Could not load VGDL rules for {env_name_full}: {e}")
            vgdl_rules = ""
        
        # Determine level and load level layout
        level_match_in_env_name = re.search(r'-lvl(\d+)-v\d+$', env_name_full)
        level_idx_0_based = 0
        if level_match_in_env_name:
            level_idx_0_based = int(level_match_in_env_name.group(1))
        
        try:
            level_layout = load_level_map(env_name_full, level_idx_0_based + 1)
        except Exception as e:
            print(f"Warning: Could not load level layout for {env_name_full}: {e}")
            level_layout = None

        if level_layout is None:
            print(f"Warning: No level layout found for {env_name_full} (level {level_idx_0_based})")

        # Translator mode controls action-prompt rules only.
        translated_rules = ""
        if translator_mode == "on":
            try:
                translator = LLMTranslator(model_name=actual_model_for_agent)
                translation_text = translator.translate(vgdl_rules=vgdl_rules, level_layout=level_layout)
                translated_rules = f"Game rules in natural language:\n{translation_text}"
            except Exception as e:
                print(f"Warning: Translation failed for {env_name_full}: {e}. Falling back to raw VGDL for translator_mode=on.")
                translated_rules = f"Game rules (raw VGDL):\n{vgdl_rules}"
        elif translator_mode == "vgdl":
            translated_rules = f"Game rules (raw VGDL):\n{vgdl_rules}"
        elif translator_mode == "off":
            translated_rules = ""
        else:
            print(f"Warning: Unknown translator_mode '{translator_mode}'. Defaulting to off.")
            translated_rules = ""

        if scm_bootstrap == "on":
            try:
                scm_manager = SCMManager(
                    model_name=actual_model_for_agent,
                    run_dir=run_dir,
                    run_id=actual_run_id_to_use,
                    game_env_id=env_name_full,
                    blueprint_path=scm_blueprint_path,
                )
                ok, err = scm_manager.bootstrap(vgdl_rules=vgdl_rules, level_layout=level_layout)
                if not ok:
                    print(f"Warning: SCM bootstrap failed: {err}")
            except Exception as e:
                print(f"Warning: SCM manager initialization/bootstrap failed: {e}")
                scm_manager = None
        
        # Create player with error handling
        try:
            player = LLMPlayer(
                model_name=actual_model_for_agent,
                env=env,
                vgdl_rules=translated_rules,
                initial_state=level_layout,
                mode=mode,
                rotate_state=False,
                expand_state=False,
                log_dir=run_dir,
                qwen_thinking=qwen_thinking,
                qwen_sampling_overrides=qwen_sampling_overrides,
                planning_mode=planning_mode,
                planning_horizon_x=planning_horizon_x,
                shuffle_action_meanings=(shuffle_action_meanings == "on"),
                action_shuffle_seed=action_shuffle_seed,
            )
        except Exception as e:
            raise RuntimeError(f"Player creation failed for {env_name_full}: {e}")

        # Initialize GIF saver
        try:
            gif_saver = show_state_gif()
        except Exception as e:
            print(f"Warning: GIF saver initialization failed: {e}")
            gif_saver = None
        
        # Reset environment for clean start.
        try:
            env.reset()
            if gif_saver:
                gif_saver(env)
        except Exception as e:
            print(f"Warning: Environment reset or GIF capture failed: {e}")
        
        # Get initial state
        try:
            _, _, _, info_init = env.step(0)
            ascii_state = info_init.get('ascii', '')
            if not ascii_state:
                print(f"Warning: Initial ASCII state is empty for {env_name_full}")
        except Exception as e:
            print(f"Warning: Initial step failed: {e}")
            ascii_state = ""

        done = False
        sprite_map = {}

        # Main game loop with improved error handling
        while not done and (max_steps is None or step_count < max_steps):
            try:
                # Check memory periodically
                if step_count % 50 == 0 and check_memory_usage(90):
                    log_msg("info", f"High memory usage at step {step_count}, forcing garbage collection")
                    gc.collect()
                
                log_msg("debug", (
                    f"== {env_name_full} | Mode: {mode} | Effective Mode: {run_mode} | Model: {model_name_full} "
                    f"| Thinking: {qwen_thinking} | Run: {actual_run_id_to_use} | Step {step_count+1}/{max_steps if max_steps is not None else 'inf'} =="
                ))

                # Generate current state representation
                try:
                    current_sprite_map, current_ascii_state_str, _ = generate_mapping_and_ascii(
                        state_str=ascii_state,
                        vgdl_text=vgdl_rules,
                        existing_mapping=sprite_map 
                    )
                    sprite_map.update(current_sprite_map)
                except Exception as e:
                    print(f"Warning: State generation failed at step {step_count}: {e}")
                    current_ascii_state_str = ascii_state or ""

                try:
                    current_position = extract_avatar_position_from_state(
                        ascii_lines=current_ascii_state_str,
                        sprite_to_char=sprite_map,
                        flip_vertical=False
                    )
                except Exception as e:
                    print(f"Warning: Position extraction failed at step {step_count}: {e}")
                    current_position = None

                # Select action with timeout and error handling
                next_ascii_mapped_for_scm = ""
                try:
                    action = player.select_action(
                        current_state=current_ascii_state_str,
                        current_position=current_position,
                        sprite_map=sprite_map,
                        scm_context=(scm_manager.render_prompt_context() if scm_manager and causal_mode == "on" else None),
                    )
                except Exception as e:
                    raise RuntimeError(f"Action selection failed at step {step_count}: {e}")

                # Execute action
                try:
                    _, reward, done, info = env.step(action)
                    winner_info = info.get('winner', None)
                    player.update(action=action, reward=reward, winner=winner_info)
                    ascii_state = info.get('ascii', '')
                    try:
                        _, mapped_next_ascii, _ = generate_mapping_and_ascii(
                            state_str=ascii_state,
                            vgdl_text=vgdl_rules,
                            existing_mapping=sprite_map,
                        )
                        next_ascii_mapped_for_scm = mapped_ascii_for_scm(mapped_next_ascii, sprite_map)
                    except Exception:
                        next_ascii_mapped_for_scm = mapped_ascii_for_scm(ascii_state or "", sprite_map)

                    if player and player.reflection_mgr.step_log:
                        player.reflection_mgr.step_log[-1]["state_before_mapped"] = mapped_ascii_for_scm(current_ascii_state_str, sprite_map)
                        player.reflection_mgr.step_log[-1]["state_after_mapped"] = next_ascii_mapped_for_scm
                    if gif_saver:
                        gif_saver(env)
                    step_count += 1
                except Exception as e:
                    print(f"Warning: Environment step failed at step {step_count}: {e}")
                    # Try to continue with next step
                    step_count += 1
                    if step_count >= (max_steps or 1000):  # Prevent infinite loops
                        break
                    
            except Exception as e:
                print(f"Error in game loop at step {step_count}: {e}")
                traceback.print_exc()
                break  # Exit game loop on serious errors
        
        # Determine game outcome
        win_status = info.get('winner', 'UNKNOWN') if info else 'UNKNOWN'
        total_reward_val = player.total_reward if player else 0
        game_successful = done or (max_steps is not None and step_count >= max_steps)
        
        result_summary = (
            f"Finished: {env_name_full}, {model_name_full}, {run_mode}, thinking={qwen_thinking}, "
            f"run {actual_run_id_to_use}, steps {step_count}, reward {total_reward_val}, "
            f"win {win_status}, proper_end: {game_successful}"
        )

    except Exception as e:
        error_msg = f"ERROR during game: {env_name_full}, Model: {model_name_full}, Mode: {run_mode}, Run: {actual_run_id_to_use}"
        print(f"!!! {error_msg} !!!")
        print(f"Error type: {type(e).__name__}, Message: {e}")
        traceback.print_exc()
        error_type = type(e).__name__
        error_message = str(e)
        result_summary = (
            f"Failed: {env_name_full}, {model_name_full}, {run_mode}, thinking={qwen_thinking}, "
            f"run {actual_run_id_to_use} - {type(e).__name__}: {str(e)}"
        )
        
    finally:
        # Comprehensive cleanup with error handling
        print(f"Starting cleanup for {env_name_full}, run {actual_run_id_to_use}")
        run_finished_at = datetime.now()
        runtime_seconds = max(0.0, (run_finished_at - run_started_at).total_seconds())
        runtime_info = {
            "run_started_at": run_started_at.isoformat(),
            "run_finished_at": run_finished_at.isoformat(),
            "runtime_seconds": round(runtime_seconds, 3),
            "runtime_minutes": round(runtime_seconds / 60.0, 3),
        }
        
        # Clean up environment
        safe_cleanup(env, "environment")
        
        # Save logs for all runs (including failed/time-limited runs) so llm_io is always available.
        meaningful_step_ratio = None

        if player:
            try:
                player.save_logs()
            except Exception as e:
                print(f"Warning: Could not save player logs: {e}")

        # Save benchmark analysis only when run reached a successful end-state.
        if game_successful and player:
            try:
                analysis_file_path = os.path.join(run_dir, "benchmark_analysis.json")
                scm_summary = scm_manager.get_summary() if scm_manager else {}
                player.export_analysis(
                    analysis_file_path,
                    runtime_info=runtime_info,
                    run_metadata={
                        "translator_mode": translator_mode,
                        "causal_mode": causal_mode,
                        "scm_bootstrap": scm_bootstrap,
                        "planning_mode": planning_mode,
                        "planning_horizon_x": int(planning_horizon_x),
                        **scm_summary,
                    },
                )
                print(f"Analysis saved to {analysis_file_path}")
                try:
                    step_metrics_path = os.path.join(analysis_file_path, "step_metrics.json")
                    with open(step_metrics_path, "r", encoding="utf-8") as f:
                        metrics_payload = json.load(f)
                    meaningful_step_ratio = metrics_payload.get("meaningful_step_ratio")
                except Exception:
                    meaningful_step_ratio = None
            except Exception as e:
                print(f"Warning: Could not save player analysis: {e}")
        elif player:
            print(f"Game was not successful. Skipping benchmark analysis export.")

        # Save run-level summary for all runs (successful or failed).
        token_aggregates = compute_token_aggregates_from_step_log(
            player.reflection_mgr.step_log if player else []
        )
        try:
            run_summary_path = os.path.join(run_dir, "run_summary.json")
            summary_payload = {
                "game": env_name_full,
                "model": model_name_full,
                "mode": run_mode,
                "base_mode": mode,
                "translator_mode": translator_mode,
                "causal_mode": causal_mode,
                "scm_bootstrap": scm_bootstrap,
                "scm_blueprint_path": str(Path(scm_blueprint_path).resolve()) if scm_blueprint_path else "",
                "qwen_thinking_mode": qwen_thinking,
                "planning_mode": planning_mode,
                "planning_horizon_x": int(planning_horizon_x),
                "qwen_sampling_overrides": qwen_sampling_overrides,
                "action_meanings": player.action_map if player and hasattr(player, "action_map") else {},
                "action_prompt_meanings": player.action_map_prompt if player and hasattr(player, "action_map_prompt") else {},
                "shuffle_action_meanings": (shuffle_action_meanings == "on"),
                "action_shuffle_seed": int(action_shuffle_seed),
                "run_id": actual_run_id_to_use,
                "process_id": os.getpid(),
                "steps": step_count,
                "total_reward": player.total_reward if player else None,
                "winner": info.get('winner', None) if info else None,
                "game_successful": game_successful,
                "runtime": runtime_info,
                "token_aggregates": token_aggregates,
                "meaningful_step_ratio": meaningful_step_ratio,
            }
            if scm_manager:
                summary_payload.update(scm_manager.get_summary())
            if player and hasattr(player, "get_planning_summary"):
                summary_payload["planning_summary"] = player.get_planning_summary()
            if player and hasattr(player, "get_last_mentioned_action_summary"):
                summary_payload["last_mentioned_action_summary"] = player.get_last_mentioned_action_summary()
            with open(run_summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)
            print(f"Run summary saved to {run_summary_path}")
        except Exception as e:
            print(f"Warning: Could not save run summary: {e}")

        # Save GIF if available
        if gif_saver and step_count > 0:
            try:
                gif_path = os.path.join(run_dir, "gameplay.gif")
                gif_saver.save(gif_path)
                print(f"GIF saved to {gif_path}")
            except Exception as e:
                print(f"Warning: Could not save GIF: {e}")

        # Clean up objects
        safe_cleanup(player, "player")
        safe_cleanup(translator, "translator")
        safe_cleanup(gif_saver, "gif_saver")
        
        # Final status print
        if player:
            print(f"[{run_mode.upper()}] Game: {env_name_full} Model: {model_name_full} Run: {actual_run_id_to_use} "
                  f"ended after {step_count} steps. Total reward: {player.total_reward}. Success: {game_successful}")
        
        # Clean up temporary file
        try:
            if os.path.isfile(temp_file_path):
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file_path}: {e}")
        
        # Force garbage collection
        gc.collect()

        # Write centralized machine-ingestible results index.
        try:
            game_match = re.search(r"gvgai-(.*)-lvl(\d+)-v(\d+)$", env_name_full)
            simple_game = game_match.group(1) if game_match else env_name_full
            level_num = int(game_match.group(2)) if game_match else -1
            index_record = {
                "record_version": 1,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "game_env_id": env_name_full,
                "game": simple_game,
                "level": level_num,
                "model_profile": model_name_full,
                "mode": run_mode,
                "base_mode": mode,
                "translator_mode": translator_mode,
                "causal_mode": causal_mode,
                "scm_bootstrap": scm_bootstrap,
                "scm_blueprint_path": str(Path(scm_blueprint_path).resolve()) if scm_blueprint_path else "",
                "qwen_thinking_mode": qwen_thinking,
                "planning_mode": planning_mode,
                "planning_horizon_x": int(planning_horizon_x),
                "qwen_sampling_overrides": qwen_sampling_overrides,
                "action_meanings": player.action_map if player and hasattr(player, "action_map") else {},
                "action_prompt_meanings": player.action_map_prompt if player and hasattr(player, "action_map_prompt") else {},
                "shuffle_action_meanings": (shuffle_action_meanings == "on"),
                "action_shuffle_seed": int(action_shuffle_seed),
                "run_id": actual_run_id_to_use,
                "status": "completed" if game_successful else "failed",
                "winner": info.get("winner", None) if info else None,
                "steps": step_count,
                "total_reward": player.total_reward if player else None,
                "run_started_at": runtime_info.get("run_started_at"),
                "run_finished_at": runtime_info.get("run_finished_at"),
                "runtime_seconds": runtime_info.get("runtime_seconds"),
                "runtime_minutes": runtime_info.get("runtime_minutes"),
                "run_dir": run_dir,
                "step_metrics_json_path": os.path.join(run_dir, "benchmark_analysis.json", "step_metrics.json") if game_successful else "",
                "step_metrics_csv_path": os.path.join(run_dir, "benchmark_analysis.json", "step_metrics.csv") if game_successful else "",
                "run_summary_path": os.path.join(run_dir, "run_summary.json"),
                "error_type": error_type,
                "error_message": error_message,
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
                "hostname": os.uname().nodename,
                "pid": os.getpid(),
                "meaningful_step_ratio": meaningful_step_ratio,
            }
            if scm_manager:
                index_record.update(scm_manager.get_paths())
            if player and hasattr(player, "get_planning_summary"):
                index_record.update(player.get_planning_summary())
            if player and hasattr(player, "get_last_mentioned_action_summary"):
                index_record.update(player.get_last_mentioned_action_summary())
            index_record.update(token_aggregates)
            write_results_index_record(index_record, results_index_dir=results_index_dir)
        except Exception as e:
            print(f"Warning: Could not write centralized results index: {e}")
        
        print(f"Cleanup completed for {env_name_full}, run {actual_run_id_to_use}")
    
    return result_summary


def generate_tasks_prioritized(
    game_list_to_process,
    models,
    modes,
    num_runs,
    base_output_dir,
    max_steps,
    results_index_dir,
    portkey_virtual_keys_loaded,
    force_rerun,
    specific_level=None,
    qwen_thinking: str = "auto",
    translator_mode: str = "on",
    causal_mode: str = "off",
    scm_bootstrap: str = "off",
    scm_blueprint_path: str = "scm_blueprint.py",
    planning_mode: str = "off",
    planning_horizon_x: int = 1,
    shuffle_action_meanings: str = "off",
    action_shuffle_seed: int = 42,
):
    """
    Generate tasks with prioritized ordering: complete all games for run 1, then run 2, etc.
    """
    tasks = []
    
    for current_run_num in range(1, num_runs + 1):
        print(f"Preparing tasks for run {current_run_num}...")
        
        for game_short_name_cli in game_list_to_process:
            game_base_name_for_level_iteration = game_short_name_cli
            game_version_for_level_iteration = "0"

            match_versioned_cli = re.match(r"(.+)_v(\d+)", game_short_name_cli)
            if match_versioned_cli:
                game_base_name_for_level_iteration = match_versioned_cli.group(1)
                game_version_for_level_iteration = match_versioned_cli.group(2)
            
            game_base_name_for_level_iteration = re.sub(r'-lvl\d+', '', game_base_name_for_level_iteration)

            levels_to_process = []
            game_dir_name = f"{game_base_name_for_level_iteration}_v{game_version_for_level_iteration}"

            script_dir = Path(__file__).parent
            possible_game_paths = [
                Path(f"../gym_gvgai/envs/games/{game_dir_name}"),
                Path(f"gym_gvgai/envs/games/{game_dir_name}"),
            ]
            
            game_levels_path = None
            for path in possible_game_paths:
                if path.is_dir():
                    game_levels_path = path
                    break
            
            if game_levels_path is None:
                print(f"Warning: Game directory not found for {game_dir_name}. Will attempt level 0.")
                levels_to_process.append(0)
                continue
                
            print(f"Discovering levels for {game_dir_name} in path: {game_levels_path.resolve()}")
            if game_levels_path.is_dir():
                found_level_indices = set()
                for f_path in sorted(game_levels_path.glob("*.txt")):
                    filename = f_path.name
                    if filename.lower() == f"{game_base_name_for_level_iteration}.txt".lower():
                        continue
                    
                    primary_pattern = rf"{re.escape(game_base_name_for_level_iteration)}_lvl(\d+)\.txt"
                    level_match_primary = re.match(primary_pattern, filename, re.IGNORECASE)
                    
                    if level_match_primary:
                        found_level_indices.add(int(level_match_primary.group(1)))
                    else:
                        fallback_pattern = r'lvl(\d+)\.txt'
                        level_match_fallback = re.match(fallback_pattern, filename, re.IGNORECASE)
                        if level_match_fallback:
                            found_level_indices.add(int(level_match_fallback.group(1)))
                
                if found_level_indices:
                    levels_to_process = sorted(list(found_level_indices))
                    print(f"Found levels for {game_dir_name}: {levels_to_process}")
                else:
                    print(f"Warning: No level files found for {game_dir_name}. Will attempt level 0.")
                    levels_to_process.append(0)
            else:
                print(f"Warning: Game directory {game_levels_path} not found. Will attempt level 0.")
                levels_to_process.append(0)

            if not levels_to_process:
                continue

            if specific_level is not None:
                if specific_level in levels_to_process:
                    levels_to_process = [specific_level]
                    print(f"Filtered levels for {game_dir_name} to specific level: {specific_level}")
                else:
                    print(
                        f"Warning: Requested --specific_level {specific_level} not found for "
                        f"{game_dir_name}. Skipping this game."
                    )
                    continue

            for level_num in levels_to_process:
                env_name_full = f'gvgai-{game_base_name_for_level_iteration}-lvl{level_num}-v{game_version_for_level_iteration}'

                for model_name_full in models:
                    for mode in modes:
                        effective_mode = resolve_effective_mode(
                            model_name_full,
                            mode,
                            qwen_thinking,
                            translator_mode=translator_mode,
                            causal_mode=causal_mode,
                            scm_bootstrap=scm_bootstrap,
                            planning_mode=planning_mode,
                            planning_horizon_x=planning_horizon_x,
                        )
                        task_args = {
                            "env_name_full": env_name_full,
                            "mode": mode,
                            "effective_mode": effective_mode,
                            "model_name_full": model_name_full,
                            "requested_run_id": current_run_num,
                            "base_output_dir": base_output_dir,
                            "max_steps": max_steps,
                            "results_index_dir": results_index_dir,
                            "force_rerun": force_rerun,
                            "qwen_thinking": qwen_thinking,
                            "translator_mode": translator_mode,
                            "causal_mode": causal_mode,
                            "scm_bootstrap": scm_bootstrap,
                            "scm_blueprint_path": scm_blueprint_path,
                            "planning_mode": planning_mode,
                            "planning_horizon_x": planning_horizon_x,
                            "shuffle_action_meanings": shuffle_action_meanings,
                            "action_shuffle_seed": int(action_shuffle_seed),
                        }
                        
                        if model_name_full in portkey_virtual_keys_loaded:
                            task_args["portkey_virtual_key"] = portkey_virtual_keys_loaded[model_name_full]
                        
                        tasks.append(task_args)
    
    return tasks


def main():
    parser = argparse.ArgumentParser(description='Run LLM Agent on GVGAI games with improved error handling')
    parser.add_argument('--games', nargs='*', default=None, help='List of game names')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['local-qwen3-8b-vllm'],
        help='List of models (default: local-qwen3-8b-vllm)'
    )
    parser.add_argument('--modes', nargs='+', default=['zero-shot', 'contextual'], help='List of modes')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs per game/model/mode')
    parser.add_argument('--base_output_dir', type=str, default='llm_agent_runs_output', help='Base output directory')
    parser.add_argument('--results_index_dir', type=str, default=None, help='Centralized results index directory (default: <base_output_dir>/results_index)')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum parallel workers (default: 4, reduced for stability)')
    parser.add_argument('--force_rerun', action='store_true', help='Force rerun existing tasks')
    parser.add_argument('--reverse', action='store_true', help='Process games in reverse order')
    parser.add_argument('--resume_game', type=str, default=None, help='Resume from specific game')
    parser.add_argument('--specific_level', type=int, default=None, help='Process only specific level')
    parser.add_argument(
        '--log_verbosity',
        type=str,
        choices=['quiet', 'info', 'debug'],
        default='info',
        help='Console logging verbosity (default: info)'
    )
    parser.add_argument(
        '--qwen_thinking',
        type=str,
        choices=['auto', 'on', 'off'],
        default='auto',
        help='Qwen3 thinking mode toggle for local vLLM profile (default: auto)'
    )
    parser.add_argument(
        '--translator_mode',
        type=str,
        choices=['on', 'off', 'vgdl'],
        default='on',
        help='Rule text mode for action prompts: translated NL (on), none (off), raw VGDL (vgdl).'
    )
    parser.add_argument(
        '--causal_mode',
        type=str,
        choices=['off', 'on'],
        default='off',
        help='Whether to inject SCM belief context into action prompts.'
    )
    parser.add_argument(
        '--scm_bootstrap',
        type=str,
        choices=['off', 'on'],
        default='off',
        help='Build initial SCM from VGDL + level layout.'
    )
    parser.add_argument(
        '--scm_blueprint_path',
        type=str,
        default='scm_blueprint.py',
        help='Path to SCM blueprint source.'
    )
    parser.add_argument(
        '--planning_mode',
        type=str,
        choices=['off', 'lookahead_actions'],
        default='off',
        help='Planning policy for action generation (default: off).'
    )
    parser.add_argument(
        '--planning_horizon_x',
        type=int,
        default=1,
        help='Lookahead horizon x for planning_mode=lookahead_actions (default: 1).'
    )
    parser.add_argument(
        '--shuffle_action_meanings',
        type=str,
        choices=['off', 'on'],
        default='off',
        help='Shuffle action legend/reminder order in prompts while keeping action IDs unchanged.'
    )
    parser.add_argument(
        '--action_shuffle_seed',
        type=int,
        default=42,
        help='Random seed used when --shuffle_action_meanings=on.'
    )

    args = parser.parse_args()
    set_logging_verbosity(args.log_verbosity)

    if args.causal_mode == "off":
        if args.scm_bootstrap != "off":
            print("Info: causal_mode=off, forcing scm_bootstrap=off.")
        args.scm_bootstrap = "off"

    if args.planning_mode == "off":
        args.planning_horizon_x = 1
    else:
        args.planning_horizon_x = max(1, int(args.planning_horizon_x))

    # Reduce max_workers if system has limited resources
    available_memory_gb = psutil.virtual_memory().total / (1024**3)
    if available_memory_gb < 16:  # Less than 16GB RAM
        recommended_workers = max(1, min(args.max_workers, 2))
        print(f"System has {available_memory_gb:.1f}GB RAM. Reducing max_workers to {recommended_workers} for stability.")
        args.max_workers = recommended_workers

    # Load Portkey virtual keys
    portkey_virtual_keys_loaded = {}
    
    try:
        script_dir = Path(__file__).parent
        dotenv_path = script_dir / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path=str(dotenv_path))
            print(f"Loaded .env file from: {dotenv_path.resolve()}")
        else:
            print(f"Warning: .env file not found at {dotenv_path.resolve()}")

        for model_cli_name in args.models:
            try:
                profile_data = get_profile_config(model_cli_name)
                virtual_key_env_var_name = profile_data.get("virtual_key_env_var")

                if virtual_key_env_var_name:
                    virtual_key_value = os.getenv(virtual_key_env_var_name)
                    if virtual_key_value:
                        portkey_virtual_keys_loaded[model_cli_name] = virtual_key_value
                        print(f"Loaded virtual key for '{model_cli_name}'")
                    else:
                        print(f"Warning: Virtual key env var '{virtual_key_env_var_name}' not found for '{model_cli_name}'")
                elif model_cli_name.startswith("portkey-"):
                    # Fallback for older naming
                    key_suffix = model_cli_name.replace('portkey-', '').upper().replace('-', '_')
                    env_var_name_fallback = f'PORTKEY_VIRTUAL_KEY_{key_suffix}'
                    key_fallback_val = os.getenv(env_var_name_fallback)
                    if key_fallback_val:
                        portkey_virtual_keys_loaded[model_cli_name] = key_fallback_val
                        print(f"Loaded virtual key for '{model_cli_name}' using fallback")
                    else:
                        print(f"Warning: No virtual key found for '{model_cli_name}'")

            except Exception as e:
                print(f"Error processing model '{model_cli_name}' for keys: {e}")
    
    except Exception as e:
        print(f"Error during Portkey key setup: {e}")

    # Determine games to process
    game_list_to_process = []
    if args.games:
        game_list_to_process = args.games
        print(f"Processing user-specified games: {game_list_to_process}")
    else:
        print("Scanning for all available games...")
        possible_paths = [
            Path("../gym_gvgai/envs/games/"),
            Path("gym_gvgai/envs/games/"),
        ]
        
        all_games_dir = None
        for path in possible_paths:
            if path.is_dir():
                all_games_dir = path
                break
        
        if all_games_dir is None:
            print(f"Error: Could not find games directory. Please specify games via --games.")
            return
            
        if all_games_dir.is_dir():
            for game_path in sorted(all_games_dir.iterdir()):
                if game_path.is_dir():
                    game_name = game_path.name
                    if "testgame" not in game_name.lower():
                        game_list_to_process.append(game_name)
            if game_list_to_process:
                print(f"Found {len(game_list_to_process)} games: {game_list_to_process}")
            else:
                print(f"Warning: No games found after excluding testgames.")
                return
        else:
            print(f"Error: Games directory {all_games_dir} not found. Please specify games via --games.")
            return
            
    # Apply resume_game and reverse filters
    if args.resume_game:
        try:
            start_index = game_list_to_process.index(args.resume_game)
            game_list_to_process = game_list_to_process[start_index:]
            print(f"Resuming from game: {args.resume_game}. Games to process: {game_list_to_process}")
        except ValueError:
            print(f"Warning: Resume game '{args.resume_game}' not found in the game list. Processing all games.")
    
    if args.reverse:
        game_list_to_process.reverse()
        print(f"Processing games in reverse order: {game_list_to_process}")

    # Generate tasks with prioritized ordering
    print(f"\n=== Generating tasks with prioritized execution ===")
    print(f"Strategy: Complete all games for run 1, then run 2, etc.")
    print(f"This ensures faster initial results across all games.")

    results_index_dir = args.results_index_dir or os.path.join(args.base_output_dir, "results_index")
    
    tasks = generate_tasks_prioritized(
        game_list_to_process=game_list_to_process,
        models=args.models,
        modes=args.modes,
        num_runs=args.num_runs,
        base_output_dir=args.base_output_dir,
        max_steps=args.max_steps,
        results_index_dir=results_index_dir,
        portkey_virtual_keys_loaded=portkey_virtual_keys_loaded,
        force_rerun=args.force_rerun,
        specific_level=args.specific_level,
        qwen_thinking=args.qwen_thinking,
        translator_mode=args.translator_mode,
        causal_mode=args.causal_mode,
        scm_bootstrap=args.scm_bootstrap,
        scm_blueprint_path=args.scm_blueprint_path,
        planning_mode=args.planning_mode,
        planning_horizon_x=args.planning_horizon_x,
        shuffle_action_meanings=args.shuffle_action_meanings,
        action_shuffle_seed=args.action_shuffle_seed,
    )

    if not tasks:
        print("No tasks to run.")
        return

    print(f"\nPrepared {len(tasks)} tasks to run with {args.max_workers} workers.")
    print(f"Task execution order: Run 1 (all games) -> Run 2 (all games) -> ... -> Run {args.num_runs} (all games)")
    
    # Using ProcessPoolExecutor with improved error handling.
    # In some environments (e.g., restricted /dev/shm), process-pool semaphore
    # creation fails with PermissionError; fallback to serial execution.
    completed_count = 0
    total_tasks = len(tasks)
    failed_tasks = []

    def record_failed_task(task, error):
        failed_task_info = (
            f"{task['env_name_full']}, {task['model_name_full']}, "
            f"{task.get('effective_mode', task['mode'])}, run_{task['requested_run_id']}"
        )
        failed_tasks.append(failed_task_info)
        print(
            f"[{completed_count}/{total_tasks}] !!! Task failed: {failed_task_info} - "
            f"{type(error).__name__}: {error} !!!"
        )
        traceback.print_exc()

    def run_tasks_serially(task_list):
        nonlocal completed_count
        for task in task_list:
            completed_count += 1
            try:
                result = run_single_game_task(**task)
                print(f"[{completed_count}/{total_tasks}] Task completed: {result}")
            except Exception as e:
                record_failed_task(task, e)

    future_to_task = {}
    try:
        try:
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(run_single_game_task, **task): task for task in tasks}

                # Process completed tasks
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    completed_count += 1

                    try:
                        result = future.result(timeout=180)  # 180 second timeout for getting results
                        print(f"[{completed_count}/{total_tasks}] Task completed: {result}")
                    except Exception as e:
                        record_failed_task(task, e)
        except PermissionError as e:
            print(
                "Process pool initialization failed with PermissionError; "
                "falling back to serial execution."
            )
            print(f"PermissionError details: {e}")
            run_tasks_serially(tasks)
    except KeyboardInterrupt:
        print("\n!!! Keyboard interrupt received (Ctrl+C) !!!")
        print("Attempting to cancel remaining tasks...")
        
        # Cancel all pending futures
        for future in future_to_task.keys():
            if not future.done():
                cancelled = future.cancel()
                if cancelled:
                    print(f"Cancelled pending task")
                else:
                    print(f"Could not cancel running task")
        
        # Wait for graceful shutdown
        import time
        time.sleep(2)
        
        print("Shutdown complete. Some tasks may have been interrupted.")
        return
    except Exception as e:
        print(f"!!! Unexpected error in main execution: {e} !!!")
        traceback.print_exc()
        return

    # Summary
    print(f"\n=== Execution Summary ===")
    print(f"Total tasks: {total_tasks}")
    print(f"Completed: {completed_count}")
    print(f"Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        print("\nFailed tasks:")
        for failed_task in failed_tasks:
            print(f"  - {failed_task}")
    
    print(f"=== All tasks processed ===")


if __name__ == "__main__":
    main()
