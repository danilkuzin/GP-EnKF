import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_rseeds = 10
    with open('results_multiple/0/parameters.pkl', "rb") as f:
        parameters = pkl.load(f)

    results = []
    for rseed in range(num_rseeds):
        with open('results_multiple/{}/results.pkl'.format(rseed), "rb") as f:
            results.append(pkl.load(f))

    # plot likelihood history
    fig, ax = plt.subplots(1)
    for result_key in results[0].keys():
        likelihood_history = np.zeros((num_rseeds, parameters.T))
        for rseed in range(num_rseeds):
            cur_res = results[rseed]
            likelihood_history[rseed] = cur_res[result_key].likelihood_history
        ax.plot(np.mean(likelihood_history, axis=0), label=result_key)
    ax.legend()
    plt.xlabel('iteration')
    plt.ylabel('log likelihood history')
    plt.savefig('graphics_new_multiple/compare_loglikelihood.eps', format='eps')

    #plot nmse history
    fig, ax = plt.subplots(1)
    for result_key in results[0].keys():
        nmse_history = np.zeros((num_rseeds, parameters.T))
        for rseed in range(num_rseeds):
            cur_res = results[rseed]
            nmse_history[rseed] = cur_res[result_key].nmse_history
        if result_key == 'gpenkf_learn_gp':
            ax.plot(np.mean(nmse_history, axis=0), label='Dual GP-EnKF')
        elif result_key == 'gpenkf_learn_liuwest_gp':
            ax.plot(np.mean(nmse_history, axis=0), label='Liu-West Dual GP-EnKF')
        elif result_key == 'gpenkf_augmented_gp':
            ax.plot(np.mean(nmse_history, axis=0), label='Joint GP-EnKF')
        elif result_key == 'normal_gp':
            ax.plot(np.mean(nmse_history, axis=0), label='Classic GP')

    ax.legend()
    plt.xlabel('iteration')
    plt.ylabel('NMSE history')
    plt.savefig('graphics_new_multiple/compare_nmse.eps', format='eps')