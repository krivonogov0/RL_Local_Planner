import isaacsim.core.utils.prims as prim_utils
import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaacsim.core.utils.semantics import add_update_semantics
from prettytable import PrettyTable

TERM_COUNTS = {}
experiment_count = 0


def benchmark(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Collects and logs termination statistics during RL policy training.

    This event callback tracks how often each termination condition is triggered
    and periodically displays statistics about the most common failure modes.

    Features:
    - Tracks termination counts across all environments
    - Periodically prints formatted statistics (every 5 experiments)
    - Computes fractional occurrence of each termination condition
    - Maintains global counters across multiple calls

    Args:
        env (ManagerBasedRLEnv): The RL training environment instance
        env_ids (torch.Tensor): Tensor of environment IDs being evaluated

    Global State:
        TERM_COUNTS (dict): Maintains cumulative count of each termination condition
        experiment_count (int): Total number of experiments processed

    Output Format:
        Prints PrettyTable with columns:
        - Termination Reason: Name of the termination condition
        - Count: Absolute number of occurrences
        - Fraction: Relative frequency (count/total)

    Example Output:

        Total Experiments: 30
        +---------------------+-------+---------+
        |   Termination Conditions Statistics   |
        +---------------------+-------+---------+
        | Termination Reason  | Count | Fraction|
        +---------------------+-------+---------+
        | time_out            | 142   |  0.71   |
        | base_contact        | 38    |  0.19   |
        | is_success          | 20    |  0.10   |
        +---------------------+-------+---------+

    Notes:
        - Call this function after each environment step during training
        - Statistics are printed every 5th experiment (modifiable in code)
        - Reset global variables TERM_COUNTS and experiment_count between training runs
        - Only tracks active termination terms from the termination manager
    """
    global TERM_COUNTS, experiment_count

    list_terms = env.termination_manager.active_terms
    experiment_count += 1

    if not TERM_COUNTS:
        TERM_COUNTS = {term: 0 for term in list_terms}
        experiment_count = 0

    for term in list_terms:
        term_value = env.termination_manager.get_term(term)
        if term_value.item():
            TERM_COUNTS[term] += 1

    if experiment_count % 5 == 0:
        total_terminations = sum(TERM_COUNTS.values())

        print(f"\nTotal Experiments: {experiment_count}")

        table = PrettyTable()
        table.title = "Termination Conditions Statistics"
        table.field_names = ["Termination Reason", "Count", "Fraction"]

        for term, count in TERM_COUNTS.items():
            fraction = count / total_terminations if total_terminations > 0 else 0.0
            table.add_row([term, count, f"{fraction:.2f}"])

        print(table)


def add_semantic(env: ManagerBasedRLEnv, env_ids: torch.Tensor, prim_path: str, semantic_label: str):
    """Adds semantic label to a prim at given path.

    Args:
        env: RL environment manager.
        env_ids: Tensor of environment IDs to update.
        prim_path: Path to the prim in the scene.
        semantic_label: Semantic label to assign to the prim.
    """
    prim = prim_utils.get_prim_at_path(prim_path)

    add_update_semantics(prim=prim, semantic_label=semantic_label)
