import matplotlib
from rl2_eval.py import evaluate_model
# env_name, eval_model, tasks, num_actions, num_states, num_traj, traj_len
def plot_models(models: list, algo: str, env_name: str, eval_model: str, tasks: int, num_actions: int, num_states: int, num_traj: int, traj_len: int):
    avg_rews = []

    # run rl2_eval on each model in the list, collect all average reward
    for mod in models:
        all_rews = evaluate_model(mod, algo, env_name, eval_model, tasks, num_actions, num_states, num_traj, traj_len)
        avg = all_rews/len(all_rews)
        avg_rews.append(avg)

    plt.plot(avg_rews, range(len(models)))
    plt.xlabel("Number of Training Iterations")
    plt.ylabel("Average Reward")

    return

def main():
    

if __name__ = "__main__":
    main()