import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from prettytable import PrettyTable

TERM_COUNTS = {}
experiment_count = 0


def benchmark(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
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
