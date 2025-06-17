# RL LOCAL PLANNER (Isaac Lab)

## Overview

This repository contains a project focused on training local navigation policies for the robots in the `Isaac Lab` simulator.  
It includes simulation scenarios, environment configurations, and training scripts using the `PPO` algorithm.

It allows you to develop in an isolated environment, outside of the core `Isaac Lab` repository.

**Who is this repository for?**

- Researchers and developers working on mobile robotics and reinforcement learning.
- Rapid prototyping of new approaches to local navigation in simulation.

**Key Features:**

- `Isolation` Work outside the core `Isaac Lab` repository, ensuring that your development efforts remain self-contained.
- `Flexibility` Easily adjust the Markov Decision Process (`MDP`) settings and neural network architectures through configuration files.

**Keywords:** `Reinforcement Learning`, `PPO`, `Isaac Lab`, `Local Navigation`

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    python -m pip install -e source/RL_Local_Planner
    ```

- Verify that the extension is correctly installed by:

  - Listing the available tasks:

    Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
    (in the `scripts/list_envs.py` file) so that it can be listed.

    ```bash
    python scripts/list_envs.py
    ```

  - Running a task:

    ```bash
    python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
    ```

    **AVAILABLE TASKS:**

    ```bash
    python scripts/skrl/train.py --task=Template-Rl-Local-Planner-v0 --num_envs=128 --max_iterations=4000
    ```

    ```bash
    python scripts/skrl/train.py --task=Template-Rl-Local-Planner-Privileged-Info-v0 --num_envs=128 --max_iterations=4000 --enable_cameras
    ```

    ```bash
    python scripts/skrl/train.py --task=Template-Rl-Local-Planner-Jetbot-v0 --num_envs=128 --max_iterations=4000
    ```

## Policy inference

To launch a trained policy:

```bash
python scripts/skrl/play.py --task=<TASK_NAME> --checkpoint=<CHECKPOINT_PATH> --num_envs=1 --enable_cameras
```

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
