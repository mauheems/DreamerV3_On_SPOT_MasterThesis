# Analysis Notebooks

All notebooks have outputs stripped — run them by setting `checkpoint_dir` and `data_dir` at the top of each notebook.

Notebooks ending in `_noobs` use the **NoObs** variant (no camera input). See the main README for the distinction.

---

## 1. `exploration_data_collected.ipynb`

**Purpose**: Inspect raw episode data before training.

Shows trajectory shapes, sensor value distributions, normalisation sanity checks, and episode statistics. Useful for verifying data quality before kicking off training.

---

## 2. `reward_computation.ipynb`

**Purpose**: Visualise how the reward signal is computed and distributed.

Plots per-timestep rewards alongside sensor observations (velocity, distance to goal). Used to verify the reward function is sensible before training.

---

## 3. `training_results_analysis.ipynb`

**Purpose**: Compare training runs across hyperparameter configurations.

- Config diff table across all experiment runs
- Side-by-side loss curves and metric grids
- Useful for ablation analysis and picking the best checkpoint

Set `run_dirs` in cell 2 to point to your experiment log directories.

---

## 4. `world_model_reconstruction.ipynb`

**Purpose** *(Visual variant)*: Evaluate posterior reconstruction — does the world model correctly reconstruct what it observed?

Feeds real observations through the encoder + RSSM and compares reconstructed observations to the originals. Tests whether the world model has learned a meaningful representation.

---

## 5. `results_latentspace.ipynb`

**Purpose**: Analyse the learned latent space.

- t-SNE of posterior latent states coloured by velocity, distance-to-goal, orientation, and reward
- Cross-checkpoint interpretability: do different checkpoints learn similar structure?

---

## 6. `agent_critic_evaluation.ipynb` *(Visual)*

**Purpose**: Evaluate whether the trained policy produces meaningful behaviour in imagination.

Loads a checkpoint, rolls out the policy in the world model's latent space, and visualises imagined trajectories. Shows whether the actor has learned to navigate toward the goal.

---

## 7. `agent_critic_evaluation_noobs.ipynb` *(NoObs)*

**Purpose**: Same as above but for the **NoObs** variant — no camera or terrain map.

This notebook is the primary evaluation tool for the main thesis result. The policy operates entirely on proprioceptive state.

---

## 8. `deployment_evaluation.ipynb`

**Purpose**: Evaluate the deployed policy on the physical SPOT robot.

Loads `.npz` recording files written by `spot_live.py` during real-robot trials and computes:
- Distance-to-goal over time
- Near-goal rate (fraction of time within threshold)
- 2D trajectory plots
- Cross-checkpoint comparisons

---

## Notes

- **Data paths**: Each notebook has a `checkpoint_dir` / `data_dir` variable near the top. Point these to your trained checkpoints and HDF5 episode files.
- **Figures**: Generated plots are saved to `figures/` for use in the thesis paper.
- **Outputs stripped**: All cell outputs have been removed to keep the repo lightweight. Re-run cells to reproduce figures.
