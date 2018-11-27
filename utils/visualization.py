import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
import torch
import numpy as np

class Visualizer:

    def __init__(self, config):
        if "title" in config.keys():
            self.title = config['title']
        else:
            self.title = True

        if "font-size" in config.keys():
            self.font_size = config['font-size']
        else:
            self.font_size = 16

        if "marker_width" in config.keys():
            self.marker_width = config['marker_width']
        else:
            self.marker_width = 1

        matplotlib.rcParams.update({'font.size': self.font_size})
        #matplotlib.rcParams.update()

    def plot_mle(self, mle_mu, mle_sigma, list_psi_t, thetas, plot_dir, name="MLE", nb_theta=12):
        height = nb_theta
        colormap = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        for theta_ax in range(thetas.shape[1]):
            plt.rcParams['figure.figsize'] = (22, height*5)
            for i in range(nb_theta):
                # plot of the convergence curve
                plt.subplot(height, 2, 2*i+1)
                x = list(range(list_psi_t.shape[1]))
                if list_psi_t.shape[2] == 2:
                    plt.errorbar(x, list_psi_t[i, :, 0, theta_ax], 3*list_psi_t[i, :, 1, theta_ax], label='ALFINet')
                else:
                    plt.plot(x, list_psi_t[i, :, 0, theta_ax], label='ALFINet')
                mu = mle_mu[i, theta_ax]*np.ones(list_psi_t.shape[1])
                sigma = mle_sigma[i, theta_ax]
                plt.plot(x, mu, label='MLE: mu')
                plt.plot(x, mu + 3*sigma, '--r', label='MLE: mu +- 3*sigma')
                plt.plot(x, mu - 3*sigma, '--r')
                if self.title:
                    plt.title("Convergence curve")
                plt.legend()

                # Plot of the density
                plt.subplot(height, 2, 2 * i + 2)
                mu = mu[0]

                # Plot MLE distribution
                if sigma == 0:
                    plt.axvline(x=mu, color=colormap[0], label='MLE: %f' % mu)
                else:
                    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
                    plt.plot(x, mlab.normpdf(x, mu, sigma), color=colormap[0], label='MLE: %f' % mu)

                if list_psi_t.shape[2] == 2:
                    mu = list_psi_t[i, -1, 0, theta_ax]
                    sigma = list_psi_t[i, -1, 1, theta_ax]
                    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
                    plt.plot(x, mlab.normpdf(x, mu, sigma), color=colormap[1], label='Last psi: %f' % mu)
                else:
                    plt.axvline(x=list_psi_t[i, -1, 0, theta_ax], color=colormap[1], label='Last psi: %f' % list_psi_t[i, -1, 0, theta_ax])
                plt.axvline(x=thetas[i, theta_ax], color=colormap[2],label='True theta: %f' % thetas[i, theta_ax])
                if self.title:
                    plt.title("Density estimator")
                plt.legend()
            plt.savefig(os.path.join(plot_dir, name + str(theta_ax).zfill(2)))
            plt.close()

    def plot_thetas_line(self, thetas, list_psi_t, MLE, plot_dir, name="Thetas"):
        for theta_ax in range(thetas.shape[1]):
            plt.rcParams['figure.figsize'] = (20, 10)
            colormap = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            plt.scatter(list_psi_t[:, -1, 0, theta_ax], thetas[:, theta_ax], label='ALFINet',
                        facecolors='none', edgecolors=colormap[0])
            plt.scatter(MLE[:, theta_ax], thetas[:, theta_ax], label='MLE', marker='x', facecolors=colormap[1])
            plt.plot(thetas[:, theta_ax], thetas[:, theta_ax], color=colormap[2])
            if self.title:
                plt.title("Thetas vs Thetas_est")
            plt.xlabel('Thetas estimated')
            plt.ylabel('Thetas')
            plt.legend()
            plt.savefig(os.path.join(plot_dir, name + str(theta_ax).zfill(2)))
            plt.close()


    def plot_loss(self, train_loss, val_loss, val_epochs, plot_dir, name="loss", start=0):
        plt.rcParams['figure.figsize'] = (10,10)

        train_epochs = range(len(train_loss))
        val_start = np.argwhere(np.array(val_epochs) >= start)[0,0]

        if self.title:
            plt.title("Learning curve")
        plt.plot(train_epochs[start:], train_loss[start:], label="Train loss")
        plt.plot(val_epochs[val_start:], val_loss[val_start:], label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, name), dpi=80)
        plt.close()

    def plot_rmse_t(self, rmse_t, config, plot_dir, name="rmse-t"):
        plt.rcParams['figure.figsize'] = (20,6)

        if self.title:
            plt.title("Evolution of the RMSE with time, for different thetas")
        for i_theta in range(config["nb_theta"]):
            plt.plot(rmse_t[i_theta,:], label=str(i_theta))
            plt.yscale('log')
        plt.xlabel("Iteration")
        plt.ylabel("RMSE")
        plt.legend()

        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, name))
        plt.close()

    def rmse_evol(self, rmse, plot_dir, name="rmse-evol"):
        plt.rcParams['figure.figsize'] = (6,6)
        colormap = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        if self.title:
            plt.title("Evolution of the average RMSE with time")
        avg = np.mean(rmse, axis=0)
        min = avg - np.std(rmse, axis=0)
        max = avg + np.std(rmse, axis=0)

        it = np.arange(rmse.shape[1]) + 1
        plt.plot(it, avg, color=colormap[0])
        plt.fill_between(it, min, max,
                        alpha=0.3, edgecolor=colormap[0], facecolor=colormap[0])
        plt.xlabel("Iteration")
        plt.ylabel("Average RMSE for different theta")
        #plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, name) + '.pdf', dpi=80)
        plt.close()

    def box_plot(self, rmse_t, plot_dir, name="box-plot-train"):
        plt.rcParams['figure.figsize'] = (10,10)

        plt.boxplot(rmse_t[:, -1])
        if self.title:
            plt.title("Distribution of the last MSE for all the thetas")
        plt.ylabel("$||\\theta^* - \\hat{\\theta}||^2_2$")
        plt.savefig(os.path.join(plot_dir, name), dpi=80)
        plt.close()

    def direction_correlation(self, alfi_direction, best_direction, plot_dir):
        plt.rcParams['figure.figsize'] = (6, 6)
        colormap = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        corrs = np.zeros(alfi_direction.shape[1:])
        for i in range(alfi_direction.shape[2]):
            for j in range(alfi_direction.shape[1]):
                corrs[j, i] = np.corrcoef(alfi_direction[:, j, i], best_direction[:, j, i])[0, 1]
            plt.plot(np.arange(alfi_direction.shape[1]) + 1, corrs[:, i], color=colormap[i])
        plt.xlabel("Iteration")
        plt.ylabel("Correlation of ideal and ALFI steps")
        plt.legend(["Theta %d " % i for i in range(alfi_direction.shape[2])])
        x_s = np.arange(0, alfi_direction.shape[1] + 1, 5)
        plt.xticks(x_s, x_s)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "correlation") + '.pdf', dpi=80)
        plt.close()


    def rmse_evol_box_plot(self, rmse_alfi, rmse_mle, rmse, plot_dir):
        plt.rcParams['figure.figsize'] = (12, 6)
        colormap = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        if self.title:
            plt.title("Evolution of the average RMSE with time")
        avg = np.mean(rmse, axis=0)
        min = avg - np.std(rmse, axis=0)
        max = avg + np.std(rmse, axis=0)

        it = np.arange(rmse.shape[1]) + 1
        # RMSE Evol
        plt.subplot(121)
        plt.plot(it, avg, color=colormap[0])
        plt.fill_between(it, min, max,
                         alpha=0.3, edgecolor=colormap[0], facecolor=colormap[0])
        plt.xlabel("Iteration")
        plt.ylabel("Average RMSE for different theta")
        x_s = np.arange(0, avg.shape[0]+1, 5)
        #x_s[0] = 1
        plt.xticks(x_s, x_s)
        # plt.yscale("log")
        plt.tight_layout()

        plt.subplot(122)
        box = plt.boxplot([rmse_alfi, rmse_mle], notch=True, patch_artist=True)
        colormap = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        box['boxes'][0].set_facecolor(colormap[0])
        box['boxes'][1].set_facecolor(colormap[1])
        plt.xticks(np.arange(2)+1, ['ALFI', 'MLE'])
        if self.title:
            plt.title("Distribution of the last RMSE for all the thetas")
        plt.tight_layout()

        plt.savefig(os.path.join(plot_dir, "rmse_evol_box_plot") + '.pdf', dpi=80)
        plt.close()

    def box_plot_mle(self, rmse_alfi, rmse_mle, plot_dir, name="box-plot-mle"):
        plt.rcParams['figure.figsize'] = (5, 5)

        box = plt.boxplot([rmse_alfi, rmse_mle], notch=True, patch_artist=True)
        colormap = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        box['boxes'][0].set_facecolor(colormap[0])
        box['boxes'][1].set_facecolor(colormap[1])
        plt.legend(box['boxes'], ['ALFI', 'MLE'], loc=9)
        if self.title:
            plt.title("Distribution of the last RMSE for all the thetas")
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, name)+ '.pdf', dpi=80)
        plt.close()

    def plot_hist_x(self, sim, proposal, list_psi_t, theta_real, config, plot_dir, name="hist-x"):
        plt.rcParams['figure.figsize'] = (18,12)

        nb_samples_gen = config["nb_samples_gen"]
        nb_samples_real = config["nb_samples_real"]
        nb_bins = config["nb_bins"]
        x_dim = sim.x_dim

        for dim in range(x_dim):
            for i_theta in range(config["nb_theta"]):
                proposal.update_psi(list_psi_t[[i_theta], -1])
                theta_star = theta_real[i_theta]
                theta_gen = proposal.sample(nb_samples_gen)
                X_gen = sim.forward(theta_gen.view(nb_samples_gen, -1), 1).view(-1, x_dim)
                X_real = sim.forward(theta_star.unsqueeze(0), nb_samples_real).view(-1, x_dim)
                plt.subplot(3, 3, i_theta+1)
                if self.title:
                    plt.title(r'Real: {} −− Inferred: {}'.format(
                        theta_star.cpu().detach().numpy(), 
                        proposal.psi[0,0,:].cpu().detach().numpy()))
                max_x_real = torch.mean(X_real) + 3 * torch.std(X_real)
                max_x_gen = torch.mean(X_gen) + 3 * torch.std(X_gen)
                min_x_real = torch.mean(X_real) - 3 * torch.std(X_real)
                min_x_gen = torch.mean(X_gen) - 3 * torch.std(X_gen)
                min_x = torch.min(min_x_real, min_x_gen).item()
                max_x = torch.max(min_x_real, min_x_gen).item()
                # hist1 = plt.hist(X_gen, label="Generated distribution", bins=nb_bins, range=[min_x,max_x], density=True, alpha=0.8)
                # hist2 = plt.hist(X_real, label="Real distribution", bins=nb_bins, range=[min_x,max_x], density=True, alpha=0.8)

                hist1 = plt.hist(X_gen[:,dim].cpu().detach().numpy(), 
                                 label="Generated distribution", 
                                 bins=nb_bins, 
                                 density=True, 
                                 alpha=0.8, 
                                 histtype=u'step')
                hist2 = plt.hist(X_real[:,dim].cpu().detach().numpy(), 
                                 label="Real distribution", 
                                 bins=nb_bins, 
                                 density=True, 
                                 alpha=0.8,
                                 histtype=u'step')

                plt.xlim(-1,1)
                plt.legend()
                plt.tight_layout()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, name + str(dim).zfill(2)) + ".pdf", dpi=80)
            plt.close()

    def init_box_plot(self, last_theta, theta_real, plot_dir, config, name="init-box-plot"):
        """Last theta: (nb_init, nb_theta*, theta_dim)"""

        nb_theta = config['nb_theta']
        nb_plots_per_row = 3
        nb_plots_per_column = np.ceil(nb_theta/nb_plots_per_row)

        plt.rcParams['figure.figsize'] = (nb_plots_per_row*8, nb_plots_per_column*5)
        for i_theta_real in range(nb_theta):
            plt.subplot(nb_plots_per_row, nb_plots_per_column, i_theta_real+1)
            plt.boxplot(last_theta[:,i_theta_real])

            if self.title:
                plt.title(r"$\theta^* = {:.2f}$ −− $\mathrm{{\mathbb{{E}}}}[\hat{{\theta}}]={:.2f}$"
                          .format(theta_real[i_theta_real], np.mean(last_theta[:,i_theta_real])))

        plt.savefig(os.path.join(plot_dir, name))
        plt.close()

    def init_rmse_t(self, rmse_t, theta_real, plot_dir, config, name="init-rmse-t"):
        """rmse-t: (nb_inits, nb_theta*, time)"""
        nb_inits = rmse_t.shape[0]
        nb_theta = config['nb_theta']
        nb_plots_per_row = 2
        nb_plots_per_column = nb_theta // nb_plots_per_row

        plt.rcParams['figure.figsize'] = (nb_plots_per_row*14, nb_plots_per_column*6)
        for i_theta_real in range(nb_theta):
            plt.subplot(nb_plots_per_column, nb_plots_per_row, i_theta_real+1)
            for i_init in range(nb_inits):
                plt.plot(rmse_t[i_init,i_theta_real,:])
            plt.xlabel("time")
            plt.ylabel("RMSE")
            if self.title:
                plt.title(r"$\theta^* = {}$".format(theta_real[i_theta_real]))

        plt.yscale("log")
        plt.savefig(os.path.join(plot_dir, name))
        plt.close()
