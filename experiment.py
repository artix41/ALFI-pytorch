import sys
import os
import shutil
import torch
import numpy as np

from models import ALFINet
from utils import Visualizer
import utils.losses as losses
import utils.weights as weights
from simulators import *
from proposals import *
from collections import Mapping


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def experiment(config, output_dir, load=False, verbose=2, device="cpu"):
    torch.manual_seed(42)

    device = None
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = device

    if "nb_threads" in config.keys():
        torch.set_num_threads(config["nb_threads"])

    sim_config = config["simulator"]
    prop_config = config["proposal"]
    net_config = config["networks"]
    train_config = config["train"]
    test_config = config["test"]
    plot_config = config["plot"]
    
    Model = str_to_class(config["model"])

    plot_dir = os.path.join(output_dir, "plots")
    model_dir = os.path.join(output_dir, "model")
    values_dir = os.path.join(output_dir, "values")
    
    nb_epochs = train_config["nb_epochs"]
    test_every = train_config["test_every"]
    save_every = train_config["save_every"]
    loss_func = Model.dict_loss_func[train_config["loss"]]
    weight_func = Model.dict_weight_func[train_config["weight"]]

    if isinstance(sim_config["parameters"], Mapping):
        simulator = str_to_class(sim_config["name"])(**sim_config["parameters"], device=device)
    else:
        simulator = str_to_class(sim_config["name"])(*sim_config["parameters"], device=device)
    proposal = str_to_class(prop_config["name"])(prop_config, device=device)
    
    alfi = Model(net_config, train_config, simulator, proposal, verbose=verbose, device=device)
    
    # =========== Load and make directories =========
    
    if load:
        if verbose >= 1:
            print("[*] Loading existing model...")
        alfi.load(model_dir, values_dir)
    else:
        os.makedirs(plot_dir)
        os.makedirs(model_dir)
        os.makedirs(values_dir)
    
    # ============= Training ==============
    
    if verbose >= 1:
        print("---------------------- Training ----------------------\n")
    
    while alfi.epoch < nb_epochs:
        alfi.train(train_config)
        #alfi.epoch += 1
        #continue
        if alfi.epoch % test_every == 0 or alfi.epoch == nb_epochs:            
            if verbose >= 2:
                print("\nValidation...")
            with torch.no_grad():
                list_psi_t, theta, val_loss = alfi.test(test_config)
                val_rmse = torch.mean(torch.norm(theta - list_psi_t[:, -1, 0,:], dim=1), 0)
                alfi.list_val_loss.append(val_loss)
                alfi.list_val_rmse.append(val_rmse)
                alfi.list_epochs_val.append(alfi.epoch)
            if verbose >= 2:
                print(" Loss: {:0.5f}\n RMSE: {:0.5f}\n".format(val_loss, val_rmse))
            
        if alfi.epoch % save_every == 0 or alfi.epoch == nb_epochs:
            if verbose >= 2:
                print("\nSaving...")
            name = 'epoch-{:04d}.model'.format(alfi.epoch)
            alfi.save(model_dir, values_dir)
            
        alfi.epoch += 1
                
    # ============= Testing ==============
        
    if verbose >= 1:
        print("---------------------- Testing ----------------------\n")
    
    list_psi_t, theta_real, test_loss = alfi.test(test_config, regenerate=True)
    
    rmse_t = torch.norm(theta_real.unsqueeze(1) - list_psi_t[:, :, 0,:], dim=2)
    print()
    
    if verbose >= 1:
        print("Test loss: {:0.5f}\n".format(test_loss))
    
    # ============ Saving ==============
    
    if verbose >= 1:
        print("---------------------- Creating plots ----------------------\n")
    
    epochs_train = [i for i in range(nb_epochs)]
    epochs_val = [i for i in range(0, nb_epochs, test_every)]
    
    np.set_printoptions(precision=2)
    visu = Visualizer(plot_config)
    
    # Losses
    visu.plot_loss(alfi.list_loss_epoch, alfi.list_val_loss, alfi.list_epochs_val, plot_dir, name="loss-start-0", start=0)
    if len(epochs_train) > 5:
        visu.plot_loss(alfi.list_loss_epoch, alfi.list_val_loss, alfi.list_epochs_val, plot_dir, name="loss-start-5", start=5)
        visu.plot_loss(alfi.list_loss_epoch, alfi.list_val_loss, alfi.list_epochs_val, plot_dir, name="loss-start-half", start=nb_epochs // 2)

    # RMSE
    visu.plot_loss(alfi.list_rmse_epoch, alfi.list_val_rmse, alfi.list_epochs_val, plot_dir, name="rmse-start-0", start=0)
    if len(epochs_train) > 5:
        visu.plot_loss(alfi.list_rmse_epoch, alfi.list_val_rmse, alfi.list_epochs_val, plot_dir, name="rmse-start-5", start=5)
        visu.plot_loss(alfi.list_rmse_epoch, alfi.list_val_rmse, alfi.list_epochs_val, plot_dir, name="rmse-start-half", start=nb_epochs // 2)
    
    # Temporal RMSE
    visu.plot_rmse_t(rmse_t.cpu().detach().numpy(), plot_config['rmse_t'], plot_dir, name="rmse-t")
    
    # Box plot MSE
    visu.box_plot(rmse_t.cpu().detach().numpy(), plot_dir, name="box-plot-MSE-test")
    
    # Histograms
    theta_real = alfi.unnormalize(theta_real, alfi.mu_theta, alfi.sigma_theta)
    list_psi_t[:,:,0] = alfi.unnormalize(list_psi_t[:,:,0], alfi.mu_theta, alfi.sigma_theta)
    if list_psi_t.shape[2] >= 2:    
        list_psi_t[:, :, 1] += torch.log(alfi.sigma_theta)
    
    visu.plot_hist_x(simulator, proposal, list_psi_t, theta_real, plot_config["hist-x"], plot_dir, name="hist-x")
    
    # MLE
    if config["simulator"]["name"] in ["PoissonSimulator", "LinearRegressionSimulator", "MultiDistriSimulator", "NewMultiDistriSimulator"]:
        mle, mle_sigma = simulator.get_mle(alfi.unnormalize(alfi.X_test, alfi.mu_X, alfi.sigma_X))
        if list_psi_t.shape[2] >= 2:
            list_psi_t[:, :, 1, :] = torch.exp(list_psi_t[:, :, 1, :])
        visu.plot_mle(mle.cpu().numpy(), mle_sigma.cpu().numpy(), list_psi_t.cpu().numpy(),
                      thetas=theta_real.cpu().numpy(), plot_dir=plot_dir, name="MLE")
        visu.plot_thetas_line(theta_real.cpu().numpy(), list_psi_t.cpu().numpy(),
                              mle.cpu().numpy(), plot_dir)
    
    # Test the influence of initialization
    nb_inits = plot_config["init_comparison"]["nb_inits"]
    nb_theta = plot_config["init_comparison"]["nb_theta"]
    rmse_t = []
    psi_t = []
    for i_init in range(nb_inits):
        list_psi_t, theta_real, test_loss = alfi.test(test_config, nb_theta=nb_theta, regenerate=False)
        rmse_t.append(torch.norm(theta_real.unsqueeze(1) - list_psi_t[:, :, 0, :], dim=2).cpu().detach().numpy())
        psi_t.append(list_psi_t[:, :, :, :].cpu().detach().numpy())
    theta_real = alfi.unnormalize(theta_real[:nb_theta], alfi.mu_theta, alfi.sigma_theta).cpu().detach().numpy()
    psi_t = np.array(psi_t)
    last_theta = alfi.unnormalize(psi_t[:, :, -1, 0, :], alfi.mu_theta.cpu().detach().numpy(), alfi.sigma_theta.cpu().detach().numpy())
    rmse_t = np.array(rmse_t)
    
    # MLE marginalized with respect to psi_0
    if psi_t.shape[3] >= 2:
        psi_t[:, :, :, 1, :] = np.exp(psi_t[:, :, :, 1, :]) * np.sqrt(nb_inits)
    avg_psi_t = psi_t.mean(0)
    avg_psi_t[:, :, 0] = alfi.unnormalize(avg_psi_t[:, :, 0], alfi.mu_theta.cpu().detach().numpy(), alfi.sigma_theta.cpu().detach().numpy())
    if avg_psi_t.shape[2] >= 2:
        avg_psi_t[:, :, 1] *= alfi.sigma_theta
    
    theta_mul = np.repeat(np.expand_dims(theta_real, 1), avg_psi_t.shape[1], 1)
    rmse_alfi_t = np.linalg.norm(theta_mul - avg_psi_t[:, :, 0, :], axis=2)
    rmse_alfi = np.linalg.norm(theta_real - avg_psi_t[:, -1, 0, :], axis=1)
    
    print("RMSE marginalized", np.mean(rmse_alfi_t[:,-1]))
    print("MSE marginalized", np.mean(rmse_alfi_t[:,-1]**2))
    # Box plot RMSE marginalized with respect to psi_0
    visu.box_plot(rmse_alfi_t, plot_dir, name="box-plot-MSE-test-marginalized")
    
    visu.rmse_evol(rmse_alfi_t, plot_dir=plot_dir)
    visu.rmse_evol(avg_psi_t[:, :, 0, 0], plot_dir=plot_dir, name="psi-avg-0-evolv")
    visu.rmse_evol(avg_psi_t[:, :, 0, 1], plot_dir=plot_dir, name="psi-avg-1-evolv")
    # if config["simulator"]["name"] in ["PoissonSimulator", "LinearRegressionSimulator", "MultiDistriSimulator", "NewMultiDistriSimulator"]:
    #     mle, mle_sigma = simulator.get_mle(alfi.unnormalize(alfi.X_test[:nb_theta], alfi.mu_X, alfi.sigma_X))
    # 
    #     visu.plot_mle(mle.cpu().numpy(), mle_sigma.cpu().numpy(), avg_psi_t,
    #                   thetas=theta_real, plot_dir=plot_dir, name="MLE-averaged")
    #     visu.plot_thetas_line(theta_real, avg_psi_t,
    #                           mle.cpu().numpy(), plot_dir, name="Thetas-averaged")
    # 
    #     rmse_mle = np.linalg.norm(theta_real - mle.cpu().numpy(), axis=1)
    #     visu.box_plot_mle(rmse_alfi, rmse_mle, plot_dir=plot_dir)
    # 
    #     visu.rmse_evol_box_plot(rmse_alfi, rmse_mle, np.linalg.norm(theta_mul - avg_psi_t[:, :, 0, :], axis=2), plot_dir=plot_dir)
    # 
    # # Compare the ideal direction with the direction taken by ALFI:
    # alfi_direction = avg_psi_t[:, :-1, 0, :] - avg_psi_t[:, 1:, 0, :]
    # best_direction = avg_psi_t[:, :-1, 0, :] - theta_mul[:, :-1, :]
    # visu.direction_correlation(alfi_direction, best_direction, plot_dir)
    # #for i in range(alfi_direction.shape[1]):
    # #    for j in range(alfi_direction.shape[2]):
    # #        print((i, j))
    # #        print(np.corrcoef(alfi_direction[:, i, j], best_direction[:, i, j]))
    # 
    # for i_theta_dim in range(last_theta.shape[2]):
    #     visu.init_box_plot(last_theta[:,:nb_theta,i_theta_dim], theta_real[:,i_theta_dim], plot_dir,
    #                        plot_config['init_box'], name="init-box-plot-{:02d}".format(i_theta_dim))
    # 
    # visu.init_rmse_t(rmse_t, theta_real, plot_dir, plot_config['init_rmse_t'], name="init-rmse-t")
    # 
