import subprocess
import itertools
import os
from datetime import datetime
import nanoid as nano
from pathlib import Path


# =============================================================================
# SWEEP CONFIGURATION - MODIFY THIS SECTION TO RUN DIFFERENT SWEEPS
# =============================================================================

# Auto-suspend configuration
AUTO_SUSPEND = True  # Set to True to automatically suspend after all sweeps complete

# Define multiple sweeps to run in sequence
# Each entry will run as a complete sweep with its own experiment name and video compilation
# Note: A 4-character nanoid will be automatically added to experiment names for uniqueness
#       (e.g., "g1-crawl-start-sweep" becomes "g1-crawl-start-sweep_a1b2")
SWEEP_QUEUE = [
    # {
    #     "task_name": "g1-shamble-start",
    #     "experiment_name": "g1-shamble-start-sweep",  # Nanoid will be auto-appended
    #     "start_from_run": 1,  # Set to 1 to start from beginning, or higher to resume
    #     "sweep_params": {
    #         #  "env.rewards.flat_orientation_l2.weight": [-3.0],
    #          "env.rewards.com_centered_over_feet.weight": [1.0, 1e-1, 0.0],
    #          "env.rewards.dof_acc_l2.weight": [-1e-1, -1e-2, 0.0],

    #          "agent.max_iterations": [2500],

    #     },
    #     "sweep_param_sets": [],
    #     "exclude_rules": [],
    # },


    # {
    #     "task_name": "g1-shamble",
    #     "experiment_name": "g1-shamble-sweep",  # Nanoid will be auto-appended
    #     "start_from_run": 1,  # Set to 1 to start from beginning, or higher to resume
    #     "sweep_params": {
    #         # "env.rewards.leg_joint_vel_l2.weight": [-1e-3, 0.0],
    #         "env.rewards.flat_orientation_l2.weight": [0.3, 0.0, -0.3],
    #         # "env.rewards.pose_deviation_hip.weight": [-0.1, -0.3],
    #         # "env.rewards.pose_deviation_knees.weight": [0.0, -0.3],
    #         # "env.rewards.feet_air_time.weight": [0.1, 1.0, 3.0],

    #         # "env.rewards.feet_slide.params.threshold": [1e-2, 0.0],
    #         # "env.rewards.leg_joint_vel_l2.weight": [ 1e-3],
    #         # "env.rewards.action_rate_l2.weight": [ 1e-3],
    #         # "env.rewards.feet_slide.params.threshold": [1e-3],
    #         # "env.rewards.pose_deviation_knees.weight": [0.0, -0.1],
    #         # "env.rewards.dof_acc_l2.weight": [0.0, -1e-2],
    #     },
    #     "sweep_param_sets": [],
    #     "exclude_rules": [],
    # },

    # {
    #     "task_name": "g1-quad",
    #     "experiment_name": "g1-quad-sweep",  # Nanoid will be auto-appended
    #     "start_from_run": 1,  # Set to 1 to start from beginning, or higher to resume
    #     "sweep_params": {
    #         #  "env.rewards.pose_deviation_all.weight": [ -0.3, -0.1],
    #         #  "env.rewards.flat_orientation_l2.weight": [ 0.1, 0.0, -.1],
    #          "agent.max_iterations": [1500],


    #     },
    #     "sweep_param_sets": [],
    #     "exclude_rules": [],
    # },

    # {
    #     "task_name": "g1-locomotion",
    #     "experiment_name": "g1-locomotion-sweep",  # Nanoid will be auto-appended
    #     "start_from_run": 1,  # Set to 1 to start from beginning, or higher to resume
    #     "sweep_params": {
    #         # "env.terminations.base_contact.params.sensor_cfg.body_names": ["torso_link","__OMIT__"],
    #          "env.rewards.track_lin_vel_xy_exp.weight": [ 1.0, 2.0 ],
    #         "env.rewards.track_ang_vel_z_exp.weight":  [  1.0, 2.0 ],

    #          "agent.max_iterations": [2500],
    #         #  "env.rewards.flat_orientation_l2.weight": [ 0.1, 0.0, -.1],

    #     },
    #     "sweep_param_sets": [
    #         # [
    #         #     {
    #         #         "env.rewards.joint_deviation_hip.weight": -0.1,
    #         #         "env.rewards.joint_deviation_arms.weight": -0.2,
    #         #         "env.rewards.joint_deviation_torso.weight": -0.1,
    #         #         "env.rewards.joint_deviation_all.weight": 0.0,
    #         #     },
    #         #     {
    #         #         "env.rewards.joint_deviation_hip.weight": 0.0,
    #         #         "env.rewards.joint_deviation_arms.weight": 0.0,
    #         #         "env.rewards.joint_deviation_torso.weight": 0.0,
    #         #         "env.rewards.joint_deviation_all.weight": -0.1,
    #         #     }
    #         #     ,
    #         #     {
    #         #         "env.rewards.joint_deviation_hip.weight": 0.0,
    #         #         "env.rewards.joint_deviation_arms.weight": 0.0,
    #         #         "env.rewards.joint_deviation_torso.weight": 0.0,
    #         #         "env.rewards.joint_deviation_all.weight": -0.3,
    #         #     }
    #         # ],
    #     ],
    #     "exclude_rules": [],
    # },
        {
        "task_name": "g1-crawl-start",
        "experiment_name": "g1-crawl-start-sweep",  # Nanoid will be auto-appended
        "start_from_run": 1,  # Set to 1 to start from beginning, or higher to resume
        "sweep_params": {

            # "env.rewards.acc_violation.weight": [ -0.1,0.0],

            # "env.rewards.either_foot_off_ground.weight":  [ -3.0, -1.0, -0.3],
            "env.rewards.pose_deviation.weight": [ -0.3, -0.1],
            "env.rewards.pose_deviation.params.delay_s": [ 1.0, 2.0 ],
            "env.rewards.pose_deviation.params.ramp_s": [ 0.0, 0.25 ],


            

            #  "env.rewards.pose_deviation_all.weight": [-1.0, 0.0],
            #  "env.rewards.flat_orientation_l2.weight": [1.0,0.3],
            #  "env.rewards.base_height_l2.weight": [-1.0,-0.3,-0.1],

             "agent.max_iterations": [1000],
            #  "env.rewards.flat_orientation_l2.weight": [ 0.1, 0.0, -.1],

        },
        "sweep_param_sets": [],
        "exclude_rules": [],
    },
        # {
    #     "task_name": "g1-crawl-start",
    #     "experiment_name": "g1-crawl-start-sweep",  # Nanoid will be auto-appended
    #     "start_from_run": 1,  # Set to 1 to start from beginning, or higher to resume
    #     "sweep_params": {
    #         "env.rewards.dof_acc_l2.weight": [-1e-3, -1e-4],
    #         # "env.rewards.pose_deviation_all.weight": [-1.0,-0.3,-0.1],
    #     },
    #     "sweep_param_sets": [
    #         # [
    #         #     {
    #         #         "env.rewards.pose_deviation_hip.weight": -0.3,
    #         #         "env.rewards.base_height_l2.weight": -0.1,
    #         #     },
    #         #     {
    #         #         "env.rewards.pose_deviation_hip.weight": -0.3,
    #         #         "env.rewards.base_height_l2.weight": -1.,
    #         #     }
    #         # ],
    #     ],
    #     "exclude_rules": [
    #         # Example: exclude when both params are 0.1
    #         # {
    #         #     "env.rewards.flat_orientation_l2.weight": 0.1,
    #         #     "env.rewards.joint_deviation_all.weight": 0.1,
    #         # },
    #     ],
    # },
]

# Notes on configuration:
# - SWEEP_PARAMS: Use lists for parameters you want to sweep (cartesian product will be generated)
# - Use "__OMIT__" to sweep between omitting a parameter vs including it
#   Example: "param": ["__OMIT__", "null"] sweeps between not setting it vs setting to null
#
# - SWEEP_PARAM_SETS: Groups of parameters that must be tested together
#   Each group is a list of dicts - one dict will be chosen from each group.
#   Cartesian product is taken ACROSS groups, but NOT within a group.
#   IMPORTANT: Parameters in SWEEP_PARAM_SETS must NOT also appear in SWEEP_PARAMS
#
# - EXCLUDE_RULES: List of rules to exclude specific parameter combinations
#   Each rule is a dict - if a combination matches ALL conditions in ANY rule, it's excluded.
#   Values can be: single value (exact match), list/tuple/set (membership), or callable (predicate)

# =============================================================================
# END OF CONFIGURATION - Don't modify below unless changing sweep logic
# =============================================================================

# Special sentinel value for omitting parameters
OMIT_PARAM = "__OMIT__"

def log_parameter_combination(combination, run_number, total_runs, log_file, status="STARTED", error=None, command_type=None, full_command=None, output_log_path=None):
    """Log parameter combination to a text file with status tracking."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        # Include UUID (agent.run_name) inline if available
        uuid_suffix = None
        if isinstance(combination, dict):
            uuid_suffix = combination.get("agent.run_name") or combination.get("run_name")
        header = f"\nRun #{run_number}/{total_runs}"
        if uuid_suffix:
            header += f" [{uuid_suffix}]"
        header += f" - {timestamp} - {status}"
        if command_type:
            header += f" ({command_type})"
        f.write(header + "\n")
        for param_name, param_value in combination.items():
            f.write(f"{param_name}: {param_value}\n")
        if full_command:
            f.write(f"Command: {' '.join(full_command)}\n")
        if output_log_path:
            f.write(f"Output Log: {output_log_path}\n")
        if error:
            f.write(f"ERROR: {error}\n")
        f.write("-" * 50 + "\n")

def _value_matches_rule(candidate_value, rule_value):
    """Return True if candidate_value satisfies rule_value condition."""
    # Callable predicate
    if callable(rule_value):
        return bool(rule_value(candidate_value))
    # Membership over an iterable of options
    if isinstance(rule_value, (list, tuple, set)):
        return candidate_value in rule_value
    # Exact equality
    return candidate_value == rule_value


def _rule_matches_combination(combination, rule):
    """Return True if combination satisfies all key conditions in rule."""
    for param_name, rule_value in rule.items():
        # If the param is not part of this sweep combination, rule cannot match
        if param_name not in combination:
            return False
        if not _value_matches_rule(combination[param_name], rule_value):
            return False
    return True


def is_excluded_combination(combination):
    """Check EXCLUDE_CONFIG to decide if this combination should be filtered out."""
    if not EXCLUDE_CONFIG:
        return False
    for rule in EXCLUDE_CONFIG:
        if _rule_matches_combination(combination, rule):
            return True
    return False


def _validate_and_prepare_group_sets(group_sets):
    """Validate group sets and return list of normalized groups.

    - Ensure each group is a non-empty list of dicts
    - Ensure all dicts within a group have identical keys
    - Return as-is after validation
    """
    if not group_sets:
        return []
    normalized = []
    for idx, group in enumerate(group_sets):
        if not isinstance(group, list) or len(group) == 0:
            raise ValueError(f"SWEEP_PARAM_SETS group #{idx} must be a non-empty list of dicts")
        keys_ref = None
        norm_group = []
        for choice_i, choice in enumerate(group):
            if not isinstance(choice, dict) or len(choice) == 0:
                raise ValueError(f"SWEEP_PARAM_SETS group #{idx} choice #{choice_i} must be a non-empty dict")
            keys = tuple(sorted(choice.keys()))
            if keys_ref is None:
                keys_ref = keys
            elif keys != keys_ref:
                raise ValueError(
                    f"SWEEP_PARAM_SETS group #{idx} has inconsistent keys across choices. "
                    f"Expected {keys_ref}, got {keys}"
                )
            norm_group.append(choice)
        normalized.append(norm_group)
    return normalized


def _assert_no_overlap_between_groups_and_params(group_sets, sweep_params):
    if not group_sets:
        return
    grouped_keys = set()
    for idx, group in enumerate(group_sets):
        for k in group[0].keys():
            if k in grouped_keys:
                raise ValueError(f"Parameter '{k}' appears in multiple SWEEP_PARAM_SETS groups (group #{idx})")
            grouped_keys.add(k)
    overlap = grouped_keys.intersection(set(sweep_params.keys()))
    if overlap:
        raise ValueError(
            "Parameters cannot appear in both SWEEP_PARAM_SETS and SWEEP_PARAMS: " + ", ".join(sorted(overlap))
        )


def _cartesian_product_of_dicts(dicts_list):
    """Cartesian product over a list of lists of dicts, merging dicts per pick.

    Input: [[{a:1},{a:2}], [{b:3},{b:4}]] -> [{a:1,b:3},{a:1,b:4},{a:2,b:3},{a:2,b:4}]
    """
    if not dicts_list:
        return [{}]
    result = [dict()]
    for choices in dicts_list:
        new_result = []
        for base in result:
            for choice in choices:
                merged = dict(base)
                # Fail loudly on key conflict
                for k, v in choice.items():
                    if k in merged and merged[k] != v:
                        raise ValueError(f"Conflicting assignments for '{k}': {merged[k]} vs {v}")
                    merged[k] = v
                new_result.append(merged)
        result = new_result
    return result


def generate_parameter_combinations():
    """Generate all combinations from SWEEP_PARAMS and SWEEP_PARAM_SETS with validation."""
    sweep_params = SWEEP_CONFIG.get("SWEEP_PARAMS", {})
    group_sets_raw = SWEEP_CONFIG.get("SWEEP_PARAM_SETS", [])

    # Validate grouped sets and overlaps
    group_sets = _validate_and_prepare_group_sets(group_sets_raw)
    _assert_no_overlap_between_groups_and_params(group_sets, sweep_params)

    # Build cartesian of independent params
    independent_param_combos = [{}]
    if sweep_params:
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        independent_param_combos = [dict(zip(param_names, values)) for values in itertools.product(*param_values)]

    # Build cartesian across groups (each group is pick-one-of-these dicts)
    grouped_combos = _cartesian_product_of_dicts(group_sets) if group_sets else [{}]

    # Merge independent with grouped; fail loudly on conflicts
    all_combos = []
    for a in independent_param_combos:
        for b in grouped_combos:
            merged = dict(a)
            conflict_keys = [k for k in b.keys() if k in merged and merged[k] != b[k]]
            if conflict_keys:
                raise ValueError(
                    "Conflicts between SWEEP_PARAMS and SWEEP_PARAM_SETS: " + ", ".join(conflict_keys)
                )
            merged.update(b)
            if not is_excluded_combination(merged):
                all_combos.append(merged)

    # If neither independent nor grouped provided, still return one empty combo
    if not all_combos:
        if not sweep_params and not group_sets:
            return [{}]
    return all_combos

def build_train_args(sweep_params):
    """Build training arguments from sweep and fixed parameters.
    
    Parameters with value OMIT_PARAM ("__OMIT__") will be skipped entirely,
    allowing you to sweep between including/omitting a parameter.
    """
    train_args = []
    
    # Add sweep parameters (skip those marked with OMIT_PARAM)
    RESERVED_KEYS = {"resume_checkpoint"}  # handled as CLI flag, not a Hydra override
    for param_name, param_value in sweep_params.items():
        # Skip reserved keys that map to CLI flags
        if param_name in RESERVED_KEYS:
            continue
        if param_value != OMIT_PARAM:
            train_args.append(f"{param_name}={param_value}")
    
   
    
    return train_args

def get_combination_description(sweep_params, combination_num, total_combinations):
    """Generate a description for the current parameter combination."""
    if not sweep_params:
        return f"Set {combination_num}/{total_combinations}: (fixed parameters only)"
    
    param_strs = []
    for param_name, param_value in sweep_params.items():
        short_name = param_name.split('.')[-1]
        # Treat resume_checkpoint set to "null"/None/empty as omitted for display purposes
        is_resume_and_null = (
            param_name == "resume_checkpoint" and (
                param_value is None or (isinstance(param_value, str) and param_value.strip().lower() in ("null", "", "none"))
            )
        )
        if param_value == OMIT_PARAM or is_resume_and_null:
            param_strs.append(f"{short_name}=<omitted>")
        else:
            param_strs.append(f"{short_name}={param_value}")
    param_description = ", ".join(param_strs)
    
    return f"Set {combination_num}/{total_combinations}: {param_description}"

def run_command(command_args, description="Running command", prefix=None, log_file=None, run_number=None, total_runs=None, combination=None, command_type=None):
    """Helper function to execute shell commands and stream output to CLI."""
    print(f"\n--- {description} ---")
    print(f"Command: {' '.join(command_args)}")
    try:
        # Prepare per-run output capture file if logging is enabled
        output_log_path = None
        output_log_file = None
        if log_file and run_number and total_runs and command_type:
            sweep_dir = os.path.dirname(log_file)
            uuid_suffix = None
            if isinstance(combination, dict):
                uuid_suffix = combination.get("agent.run_name") or combination.get("run_name")
            safe_uuid = uuid_suffix if uuid_suffix else "NA"
            output_log_filename = f"{command_type.lower()}_run_{run_number:02d}_{safe_uuid}.log"
            output_log_path = os.path.join(sweep_dir, output_log_filename)
            # Write header for the output log
            with open(output_log_path, 'a') as outf:
                outf.write(f"=== {description} ===\n")
                outf.write(f"Command: {' '.join(command_args)}\n")
        
        # Popen allows streaming output
        process = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Stream output line by line
        for line in process.stdout:
            if prefix:
                # Add prefix to each line, preserving original formatting
                formatted = f"[{prefix}] {line}"
                print(formatted, end='')
            else:
                formatted = line
                print(line, end='') # `end=''` prevents extra newlines
            # Append to output log if enabled
            if output_log_path:
                with open(output_log_path, 'a') as outf:
                    outf.write(formatted)

        process.wait() # Wait for the process to complete

        if process.returncode != 0:
            error_msg = f"Command exited with non-zero status {process.returncode}"
            print(f"\nError: {error_msg}")
            if log_file and run_number and total_runs and combination:
                log_parameter_combination(combination, run_number, total_runs, log_file, status="FAILED", error=error_msg, command_type=command_type, full_command=command_args, output_log_path=output_log_path)
            return False
        else:
            print("\nCommand completed successfully.")
            if log_file and run_number and total_runs and combination:
                log_parameter_combination(combination, run_number, total_runs, log_file, status="COMPLETED", command_type=command_type, full_command=command_args, output_log_path=output_log_path)
            return True

    except FileNotFoundError:
        error_msg = f"Command not found. Make sure '{command_args[0]}' is in your PATH or correctly specified."
        print(f"Error: {error_msg}")
        if log_file and run_number and total_runs and combination:
            log_parameter_combination(combination, run_number, total_runs, log_file, status="FAILED", error=error_msg, command_type=command_type, full_command=command_args, output_log_path=output_log_path if 'output_log_path' in locals() else None)
        return False
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(f"Error: {error_msg}")
        if log_file and run_number and total_runs and combination:
            log_parameter_combination(combination, run_number, total_runs, log_file, status="FAILED", error=error_msg, command_type=command_type, full_command=command_args, output_log_path=output_log_path if 'output_log_path' in locals() else None)
        return False

def run_single_sweep(sweep_config, sweep_number, total_sweeps):
    """Run a single sweep configuration."""
    # Extract config values
    TASK_NAME = sweep_config["task_name"]
    BASE_EXPERIMENT_NAME = sweep_config["experiment_name"]
    START_FROM_RUN = sweep_config.get("start_from_run", 1)
    
    # Add unique nanoid suffix to experiment name
    sweep_nanoid = nano.generate(size=4)
    EXPERIMENT_NAME = f"{BASE_EXPERIMENT_NAME}_{sweep_nanoid}"
    
    # Set up config for this sweep (for use by helper functions)
    global SWEEP_CONFIG, EXCLUDE_CONFIG
    SWEEP_CONFIG = {
        "SWEEP_PARAMS": sweep_config.get("sweep_params", {}),
        "SWEEP_PARAM_SETS": sweep_config.get("sweep_param_sets", []),
    }
    EXCLUDE_CONFIG = sweep_config.get("exclude_rules", [])
    
    print(f"\n{'='*80}")
    print(f"STARTING SWEEP {sweep_number}/{total_sweeps}: {EXPERIMENT_NAME}")
    print(f"Base name: {BASE_EXPERIMENT_NAME}")
    print(f"Sweep ID: {sweep_nanoid}")
    print(f"Task: {TASK_NAME}")
    print(f"{'='*80}\n")
    
    # Define base commands (resume handled exclusively via sweep parameters)
    TRAIN_BASE_CMD = ["python", "scripts/rsl_rl/train.py","--task", TASK_NAME, "--headless"]

    PLAY_BASE_CMD = ["python", "scripts/rsl_rl/play.py", "--task", TASK_NAME, "--headless", "--video", "--video_length", "200", "--enable_cameras"]

    # Create per-sweep output directory and log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_sweep_dir = "sweep-logs"
    os.makedirs(base_sweep_dir, exist_ok=True)
    sweep_output_dir = os.path.join(base_sweep_dir, f"{EXPERIMENT_NAME}_{timestamp}")
    os.makedirs(sweep_output_dir, exist_ok=True)
    log_file = os.path.join(sweep_output_dir, f"sweep_log_{EXPERIMENT_NAME}_{timestamp}.txt")
    
    # Write header to log file
    with open(log_file, 'w') as f:
        f.write(f"Parameter Sweep Log - {EXPERIMENT_NAME}\n")
        f.write(f"Task: {TASK_NAME}\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Resuming from run #{START_FROM_RUN}\n")
        f.write("\nSweep Configuration:\n")
        for param_name, param_values in SWEEP_CONFIG["SWEEP_PARAMS"].items():
            f.write(f"{param_name}: {param_values}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("\nRun Status Summary:\n")
        f.write("==================\n")

    # Generate parameter combinations
    parameter_combinations = generate_parameter_combinations()
    
    print(f"Starting parameter sweep: {EXPERIMENT_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Resuming from run #{START_FROM_RUN}")
    print(f"Generated {len(parameter_combinations)} parameter combinations to test.")
    print(f"Logging to: {log_file}")
    print(f"Sweep outputs directory: {sweep_output_dir}")
    
    # Print sweep configuration
    print(f"\nSweep parameters:")
    for param_name, param_values in SWEEP_CONFIG["SWEEP_PARAMS"].items():
        print(f"  {param_name}: {param_values}")
    if SWEEP_CONFIG.get("SWEEP_PARAM_SETS"):
        print(f"\nGrouped parameter sets:")
        for gi, group in enumerate(SWEEP_CONFIG["SWEEP_PARAM_SETS"]):
            print(f"  Group #{gi} (pick one of):")
            for choice in group:
                pretty = ", ".join(f"{k}={v}" for k, v in choice.items())
                print(f"    - {{ {pretty} }}")
    
    for i, combination in enumerate(parameter_combinations):
        # Skip runs before START_FROM_RUN
        if i + 1 < START_FROM_RUN:
            continue
            
        description = get_combination_description(combination, i+1, len(parameter_combinations))
        run_prefix = f"Sweep {sweep_number}/{total_sweeps}, Run {i+1}/{len(parameter_combinations)}"
        # Generate a short run identifier and augment combination for logging
        run_name = nano.generate(size=4)
        combo_with_names = dict(combination)
        combo_with_names["agent.experiment_name"] = EXPERIMENT_NAME
        combo_with_names["agent.run_name"] = run_name
        
        # Log the start of the parameter combination (includes UUID)
        log_parameter_combination(combo_with_names, i+1, len(parameter_combinations), log_file, status="STARTED")
        
        print(f"\n{'='*80}\nStarting {description}\n{'='*80}")

        # Construct parameter arguments
        train_args = build_train_args(combination)
        # Ensure experiment/run names are passed via Hydra overrides
        train_args += [
            f"agent.experiment_name={EXPERIMENT_NAME}",
            f"agent.run_name={run_name}",
        ]

        # Construct the full train command
        full_train_cmd = TRAIN_BASE_CMD + train_args
        # If resume_checkpoint is part of this combination, append as CLI flag when not null/omitted
        rc_value = combination.get("resume_checkpoint")
        if rc_value is not None:
            # Treat special omit/null values as skip
            if isinstance(rc_value, str):
                rc_lower = rc_value.strip().lower()
            else:
                rc_lower = None
            if not (
                rc_value == OMIT_PARAM or
                rc_value is None or
                (isinstance(rc_value, str) and rc_lower in ("", "null", "none"))
            ):
                full_train_cmd += ["--resume_checkpoint", str(rc_value)]
        train_ok = run_command(full_train_cmd, f"Training for {description}", prefix=run_prefix, 
                   log_file=log_file, run_number=i+1, total_runs=len(parameter_combinations), 
                   combination=combo_with_names, command_type="TRAIN")

        # Construct the full play command (no extra args needed for play usually)
        # Resolve the full run directory name (timestamp_UUID) for this run and pass to play
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", EXPERIMENT_NAME))
        matching_runs = []
        if os.path.isdir(log_root_path):
            for d in os.listdir(log_root_path):
                full_path = os.path.join(log_root_path, d)
                if os.path.isdir(full_path) and d.endswith(f"_{run_name}"):
                    matching_runs.append(d)
        if not matching_runs:
            # Fail loudly so it's clear what to fix
            raise RuntimeError(f"Could not find a run directory in '{log_root_path}' ending with '_{run_name}'")
        resolved_run_dir = sorted(matching_runs)[-1]
        combo_with_names["resolved_run_dir"] = resolved_run_dir

        # Write sweep overrides (UUID + params) into the run directory for later analysis/labeling
        try:
            run_dir_full = os.path.join(log_root_path, resolved_run_dir)
            overrides_path = os.path.join(run_dir_full, "sweep_overrides.txt")
            with open(overrides_path, 'w') as outf:
                outf.write(f"uuid: {run_name}\n")
                for k, v in combination.items():
                    outf.write(f"{k}={v}\n")
        except Exception as e:
            # Fail loudly
            raise RuntimeError(f"Failed to write sweep_overrides.txt for run '{resolved_run_dir}': {e}")

        play_args = [
            f"agent.experiment_name={EXPERIMENT_NAME}",
            f"agent.load_run={resolved_run_dir}",
        ]
        full_play_cmd = PLAY_BASE_CMD + play_args
        if not train_ok:
            # Skip play if training failed, but log it explicitly
            log_parameter_combination(combo_with_names, i+1, len(parameter_combinations), log_file, status="SKIPPED", command_type="PLAY", full_command=full_play_cmd)
            continue

        run_command(full_play_cmd, f"Playing for {description}", prefix=run_prefix,
                   log_file=log_file, run_number=i+1, total_runs=len(parameter_combinations),
                   combination=combo_with_names, command_type="PLAY")

    print(f"\n🎉 All {len(parameter_combinations)} parameter combinations finished!")
    
    # Automatically analyze results
    print(f"\n🔍 Starting automatic analysis of sweep results...")
    
    # Log analysis start
    with open(log_file, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("Analysis Results\n")
        f.write("================\n")
        f.write(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Analysis with explicit output directory for all generated files
        results = analyze_sweep_results(experiment_name=EXPERIMENT_NAME, output_dir=sweep_output_dir)
        
        # Log analysis results
        with open(log_file, 'a') as f:
            if results and results['success']:
                f.write("\n✅ Analysis completed successfully!\n")
                f.write("Generated files:\n")
                f.write(f"  - Experiment guide: {results['experiment_guide_file']}\n")
                f.write(f"  - Basic concatenated video: {results['video_file']}\n")
                if results.get('labeled_video_file'):
                    f.write(f"  - Labeled concatenated video: {results['labeled_video_file']}\n")
                f.write(f"  - Video mapping: {results['video_mapping_file']}\n")
            else:
                f.write("\n⚠️ Analysis completed with issues\n")
                if results:
                    f.write(f"Error details: {results.get('error', 'Unknown error')}\n")
        
        if results and results['success']:
            print(f"\n🎊 SWEEP COMPLETED SUCCESSFULLY!")
            print(f"📁 Results saved:")
            print(f"   📄 Experiment guide: {results['experiment_guide_file']}")
            print(f"   🎬 Basic concatenated video: {results['video_file']}")
            if results.get('labeled_video_file'):
                print(f"   🏷️  Labeled concatenated video: {results['labeled_video_file']}")
            print(f"   🗺️  Video mapping: {results['video_mapping_file']}")
        else:
            print(f"\n⚠️  Sweep completed but analysis had issues. Check the output above.")
            
    except Exception as e:
        error_msg = f"Error during analysis: {e}"
        print(f"\n❌ {error_msg}")
        # Log analysis error
        with open(log_file, 'a') as f:
            f.write(f"\n❌ {error_msg}\n")
            f.write("You can manually run analysis later with:\n")
            f.write("python sweep_analyzer.py\n")
        print(f"   You can manually run analysis later with:")
        print(f"   python sweep_analyzer.py")
    # Always write and print the absolute path to the sweep output directory when done
    abs_sweep_output_dir = os.path.abspath(sweep_output_dir)
    with open(log_file, 'a') as f:
        f.write(f"\nFull sweep output directory: {abs_sweep_output_dir}\n")
    print(f"\n📁 Full sweep output directory: {abs_sweep_output_dir}")
    
    return {
        "experiment_name": EXPERIMENT_NAME,
        "sweep_output_dir": abs_sweep_output_dir,
        "results": results if 'results' in locals() else None
    }


def main():
    """Run all sweeps in the SWEEP_QUEUE sequentially."""
    if not SWEEP_QUEUE:
        print("❌ No sweeps configured in SWEEP_QUEUE. Please add at least one sweep configuration.")
        return
    
    print(f"\n{'='*80}")
    print(f"🚀 STARTING SWEEP QUEUE")
    print(f"Total sweeps to run: {len(SWEEP_QUEUE)}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for sweep_idx, sweep_config in enumerate(SWEEP_QUEUE, 1):
        try:
            result = run_single_sweep(sweep_config, sweep_idx, len(SWEEP_QUEUE))
            all_results.append(result)
            
            print(f"\n✅ Completed sweep {sweep_idx}/{len(SWEEP_QUEUE)}: {sweep_config['experiment_name']}")
            
        except Exception as e:
            print(f"\n❌ Error in sweep {sweep_idx}/{len(SWEEP_QUEUE)}: {e}")
            all_results.append({
                "experiment_name": sweep_config.get('experiment_name', 'unknown'),
                "error": str(e),
                "sweep_output_dir": None,
                "results": None
            })
            # Continue with next sweep even if this one failed
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🏁 ALL SWEEPS COMPLETED")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(all_results, 1):
        status = "✅" if result.get("results") or (not result.get("error")) else "❌"
        print(f"{status} Sweep {i}: {result['experiment_name']}")
        if result.get('sweep_output_dir'):
            print(f"   📁 {result['sweep_output_dir']}")
        if result.get('error'):
            print(f"   ❌ Error: {result['error']}")
    
    print(f"\n{'='*80}\n")


def extract_parameters(experiment_dir):
    """Extract parameters and sweep overrides from an experiment directory.

    Returns a dict that includes flags for agent/env configs, the experiment
    folder name, optional UUID, and any sweep overrides captured in
    `sweep_overrides.txt`.
    """
    agent_file = os.path.join(experiment_dir, "params", "agent.yaml")
    env_file = os.path.join(experiment_dir, "params", "env.yaml")

    params = {}

    if os.path.exists(agent_file):
        params['has_agent_config'] = True
    if os.path.exists(env_file):
        params['has_env_config'] = True

    params['experiment_dir'] = os.path.basename(experiment_dir)

    # Try to read sweep overrides file created by the sweep script
    overrides_path = os.path.join(experiment_dir, "sweep_overrides.txt")
    overrides = {}
    if os.path.exists(overrides_path):
        try:
            with open(overrides_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.startswith('uuid:'):
                        params['uuid'] = line.split(':', 1)[1].strip()
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        overrides[k.strip()] = v.strip()
        except Exception:
            # Fail quietly here since labeling can still proceed with folder info
            pass
    if overrides:
        params['overrides'] = overrides

    # Derive UUID from directory suffix if not present
    if 'uuid' not in params:
        parts = params['experiment_dir'].rsplit("_", 1)
        if len(parts) == 2 and len(parts[1]) == 4 and parts[1].isalnum():
            params['uuid'] = parts[1]

    return params

def check_ffmpeg_capabilities():
    """Check what ffmpeg encoders and filters are available."""
    # Skip capability checks - user has validated ffmpeg command works
    return {
        'has_libx264': True,
        'has_drawtext': True
    }

def create_labeled_videos(experiments, timestamp, capabilities=None, base_output_dir="."):
    """Create individual labeled videos for each experiment with UUID and key params."""
    labeled_videos = []
    temp_dir_name = f"temp_labeled_{timestamp}"
    temp_dir = os.path.join(base_output_dir, temp_dir_name)
    os.makedirs(temp_dir, exist_ok=True)
    
    print("🏷️  Creating labeled videos...")
    
    # Use the font file from assets - no checks, just use it
    font_file = "assets/RobotoFlex-Variable.ttf"
    print(f"   Using font: {font_file}")
    
    for i, exp in enumerate(experiments, 1):
        # Build label text: include UUID and a compact set of params if available
        folder_name = exp['directory']
        run_uuid = exp['params'].get('uuid')
        # Choose a small subset of overrides to display to keep overlay readable
        overrides = exp['params'].get('overrides', {})
        # Include up to 3 key-value pairs
        override_items = list(overrides.items())[:3]
        kv_text = ", ".join(f"{k.split('.')[-2]}={v}" for k, v in override_items)
        if run_uuid and kv_text:
            label_text = f"{run_uuid} | {kv_text}"
        elif run_uuid:
            label_text = f"{run_uuid}"
        else:
            label_text = f"{folder_name}"
        
        output_video = os.path.join(temp_dir, f"labeled_{i:02d}.mp4")
        
        # Use the user's validated ffmpeg format
        cmd = [
            "/usr/bin/ffmpeg", "-i", exp['video_path'],
            "-vf", f"drawtext=fontfile={font_file}:text='{label_text}':fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=10:y=10",
            "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-c:a", "copy",
            "-y", output_video
        ]
        
        print(f"   Processing video {i}/{len(experiments)}: {folder_name}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            labeled_videos.append(output_video)
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to process video {i}: {e.stderr[:100] if e.stderr else 'Unknown error'}")
            return None, temp_dir
    
    return labeled_videos, temp_dir

def analyze_sweep_results(experiment_name, base_logs_dir="logs/rsl_rl", create_overlays=True, output_dir=None):
    """
    Analyze sweep results and create concatenated video with experiment guide.
    
    Args:
        experiment_name (str): Name of the experiment (e.g., "g1_23dof_sweep_v4")
        base_logs_dir (str): Base directory for logs
        create_overlays (bool): Whether to attempt creating text overlays on videos
    
    Returns:
        dict: Results summary with file paths and experiment count
    """
    
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing sweep results for: {experiment_name}")
    print(f"{'='*60}")
    
    # Construct the sweep directory path
    sweep_dir = os.path.join(base_logs_dir, experiment_name)
    
    if not os.path.exists(sweep_dir):
        print(f"❌ Error: Sweep directory not found: {sweep_dir}")
        return None
    
    # Find all experiment directories
    experiment_dirs = sorted([d for d in os.listdir(sweep_dir) if os.path.isdir(os.path.join(sweep_dir, d))])
    
    # Analyze all experiments
    experiments = []
    for exp_dir in experiment_dirs:
        full_path = os.path.join(sweep_dir, exp_dir)
        video_path = os.path.join(full_path, "videos", "play", "rl-video-step-0.mp4")
        
        if os.path.exists(video_path):
            params = extract_parameters(full_path)
            experiments.append({
                'directory': exp_dir,
                'video_path': video_path,
                'params': params
            })
    
    if not experiments:
        print(f"❌ No experiments with videos found in {sweep_dir}")
        return None
    
    print(f"📊 Found {len(experiments)} experiments with videos")
    
    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    experiment_guide_file = os.path.join(output_dir, f"{experiment_name}_experiment_guide_{timestamp}.txt")
    video_mapping_file = os.path.join(output_dir, f"{experiment_name}_video_mapping_{timestamp}.txt")
    concat_video_file = os.path.join(output_dir, f"{experiment_name}_concatenated_{timestamp}.mp4")
    concat_labeled_video_file = os.path.join(output_dir, f"{experiment_name}_concatenated_labeled_{timestamp}.mp4")
    
    # Create experiment guide
    with open(experiment_guide_file, "w") as f:
        f.write(f"Experiment Sweep Results: {experiment_name}\n")
        f.write("=" * (26 + len(experiment_name)) + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Found {len(experiments)} experiments with videos\n\n")
        
        for i, exp in enumerate(experiments, 1):
            # Include the 4-char run UUID if present in directory name
            folder_name = exp['directory']
            run_uuid = None
            parts = folder_name.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) == 4 and parts[1].isalnum():
                run_uuid = parts[1]
            header = f"Run {i:2d}: [{run_uuid}] {folder_name}" if run_uuid else f"Run {i:2d}: {folder_name}"
            f.write(header + "\n")
            f.write(f"  Video: {exp['video_path']}\n")
            f.write(f"  Config: Agent={'✓' if exp['params'].get('has_agent_config') else '✗'}, "
                   f"Env={'✓' if exp['params'].get('has_env_config') else '✗'}\n")
            f.write("\n")
    
    print(f"✅ Experiment guide created: {experiment_guide_file}")
    
    # Always create overlays since user validated the command works
    capabilities = check_ffmpeg_capabilities()
    print(f"🔧 FFmpeg ready for overlays")
    
    # Try to create videos
    video_created = False
    labeled_video_created = False
    
    try:
        # Create basic concatenated video (always works)
        concat_file = os.path.join(output_dir, f"temp_concat_{timestamp}.txt")
        with open(concat_file, "w") as f:
            for exp in experiments:
                f.write(f"file '{os.path.abspath(exp['video_path'])}'\n")
        
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy", "-y", concat_video_file
        ]
        
        print("🎬 Creating basic concatenated video...")
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(concat_file)
        print(f"✅ Basic concatenated video created: {concat_video_file}")
        video_created = True
        
        # Create labeled video - user validated the ffmpeg command works
        labeled_videos, temp_dir = create_labeled_videos(experiments, timestamp, capabilities, base_output_dir=output_dir)
        
        if labeled_videos:
            # Create concat file for labeled videos
            labeled_concat_file = os.path.join(output_dir, f"temp_labeled_concat_{timestamp}.txt")
            with open(labeled_concat_file, "w") as f:
                for video in labeled_videos:
                    f.write(f"file '{os.path.abspath(video)}'\n")
            
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", labeled_concat_file,
                "-c", "copy", "-y", concat_labeled_video_file
            ]
            
            print("🏷️  Creating labeled concatenated video...")
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Cleanup
            os.remove(labeled_concat_file)
            import shutil
            shutil.rmtree(temp_dir)
            
            print(f"✅ Labeled concatenated video created: {concat_labeled_video_file}")
            labeled_video_created = True
        else:
            print(f"⚠️  Could not create labeled videos, only basic concatenation available")
        
        # Create video mapping
        with open(video_mapping_file, "w") as f:
            f.write(f"Video Segment Mapping\n")
            f.write("=" * 21 + "\n\n")
            if labeled_video_created:
                f.write(f"Labeled video: {concat_labeled_video_file}\n")
            f.write(f"Basic video: {concat_video_file}\n\n")
            f.write("Each video segment corresponds to:\n")
            f.write(f"(Each segment is approximately 4 seconds long)\n\n")
            
            for i, exp in enumerate(experiments, 1):
                folder_name = exp['directory']
                run_uuid = None
                parts = folder_name.rsplit("_", 1)
                if len(parts) == 2 and len(parts[1]) == 4 and parts[1].isalnum():
                    run_uuid = parts[1]
                segment_label = f"{folder_name} [{run_uuid}]" if run_uuid else folder_name
                f.write(f"Segment {i:2d} ({(i-1)*4:2.0f}s-{i*4:2.0f}s): {segment_label}\n")
                f.write(f"    Video source: {exp['video_path']}\n")
                f.write("\n")
        
        print(f"✅ Video mapping created: {video_mapping_file}")
        
    except Exception as e:
        print(f"⚠️  Video creation failed: {e}")
        print(f"   Experiment guide still available: {experiment_guide_file}")
    
    # Summary
    print(f"\n📋 SWEEP ANALYSIS COMPLETE:")
    print(f"   📊 Experiments analyzed: {len(experiments)}")
    print(f"   📄 Experiment guide: {experiment_guide_file}")
    if video_created:
        print(f"   🎬 Basic concatenated video: {concat_video_file}")
    if labeled_video_created:
        print(f"   🏷️  Labeled concatenated video: {concat_labeled_video_file}")
    if video_created or labeled_video_created:
        print(f"   🗺️  Video mapping: {video_mapping_file}")
    print(f"{'='*60}\n")
    
    return {
        'experiment_count': len(experiments),
        'experiment_guide_file': experiment_guide_file,
        'video_file': concat_video_file if video_created else None,
        'labeled_video_file': concat_labeled_video_file if labeled_video_created else None,
        'video_mapping_file': video_mapping_file if (video_created or labeled_video_created) else None,
        'success': video_created or labeled_video_created
    }


if __name__ == "__main__":
    main()
    
    if AUTO_SUSPEND:
        import time
        print("\n⏳ Waiting 5 seconds before suspend...")
        time.sleep(5)
        print("💤 Suspending computer...")
        subprocess.run(["systemctl", "suspend"], check=True)