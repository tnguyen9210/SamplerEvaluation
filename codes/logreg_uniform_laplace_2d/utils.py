
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import seaborn as sns
sns.set()
sns.set_palette("tab10")

# color_list = ["Green", "Blue"]
# color_map = mcolors.ListedColormap(["Green", "Blue"])


def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in config.items():
        if k == "data_x_marginal_params" and config["data_x_type"] != "mixgauss":
            continue
        info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")
    return info


def plot_prior_posterior_samples(
        samples_a_weights_prior, samples_b_weights_prior,
        samples_a_weights_posterior, num_feats, args):
    
    # Visualize the generated prior and posterior samples, individual features 
    fig, axes = plt.subplots(
        nrows=num_feats, ncols=1, sharex=True, sharey=True, figsize=(10,10))
    axes = axes.flatten()

    for i in range(num_feats):
        sns.kdeplot(samples_a_weights_prior[:,i], color="blue",
                    label="sample_a_prior", fill=False, ax=axes[i])
        sns.kdeplot(samples_b_weights_prior[:,i], color="green",
                    label="sample_b_prior", fill=False, ax=axes[i])
        sns.kdeplot(samples_a_weights_posterior[:,i], color="orange",
                    label="sample_a_posterior", fill=False, ax=axes[i])
        axes[i].legend()
    plt.savefig(f"./figures/{args['test_name']}_indi")
    plt.show()
    
    # # Visualize the generated prior and posterior samples, pair of features
    # fig, axes = plt.subplots(
    #     nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10,10))
    # axes = axes.flatten()

    # sns.kdeplot(x=samples_a_weights_prior[:,0], y=samples_a_weights_prior[:,1],
    #             n_levels=20, cmap="inferno", fill=False, cbar=True, ax=axes[0])

    # sns.kdeplot(x=samples_b_weights_prior[:,0], y=samples_b_weights_prior[:,1],
    #             n_levels=20, cmap="inferno", fill=False, cbar=True, ax=axes[1])

    # sns.kdeplot(x=samples_a_weights_posterior[:,0], y=samples_a_weights_posterior[:,1],
    #             n_levels=20, cmap="inferno", fill=False, cbar=True, ax=axes[2])
    # # axes[0].set_aspect(aspect="equal")
    # # axes[1].set_aspect(aspect="equal")
    # # axes[2].set_aspect(aspect="equal")
    # axes[0].set_title("sample_a_prior")
    # axes[1].set_title("sample_b_prior")
    # axes[2].set_title("sample_a_posterior")
    # plt.savefig(f"./figures/{args['test_name']}_pair")
    # plt.show()
