import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Local
from .live_loss import PlotLosses

def mnist_unnormalize(x):
    return x.clone().mul_(.3081).add_(.1307)

def mnist_normalize(x):
    return x.clone().sub_(.1307).div_(.3081)


def find_completion(model, ae_model, x, subset, classes,
                    lambda_rec = 1, lambda_norm=0, lambda_dist = 1,
                    max_iters = 1000, sort_hist = False, plot_freq = 1000, plotting = True):
    """
        Given an example input x and pixel range S, finds assigment of values to the
        remaining pixels which minmizes
                || p_f(y | [x_0^S, x*]) - y ||_2 + lambda*Reconstruction([x_0^S, x*])
        where p_f is output of probabilistic prediction model f and y is a uniform
        histogram over desired classes C
    """
    # Infer useful properties
    d, _ = x.squeeze().shape
    #nclasses = model.net(torch.zeros(x.shape)).shape[-1] # FIXME: revert to below once using models trained with the __call__ attrivb
    print(torch.zeros(x.shape).shape)
    nclasses = model(torch.zeros(x.shape)).shape[-1]

    # Encode subset in np slice format
    #pix_range = np.s_[subset[0], subset[1]]

    # Target Histrogram
    target_hist = torch.zeros(1, nclasses).detach()
    target_hist[:,classes] = 1
    target_hist /= target_hist.sum()

    # Instantiate optimization variable
    z = mnist_normalize(torch.zeros(x.shape).detach())   ## FIXME: Must make this general. Maybe add to my trainer classes.
    z.view(d,d)[subset].data.copy_(x.view(d,d)[subset].data)
    z.requires_grad_()

    if plotting:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(x.view(d,d).detach().numpy().squeeze())
        ax[1].imshow(x.view(d,d)[subset].detach().numpy().squeeze())
        ax[0].set_title('Input')
        ax[1].set_title('Fixed Attribute')
        ax[0].axis('off')
        ax[1].axis('off')
        plt.show()

    time.sleep(2)
    optimizer = torch.optim.Adam([z], lr=0.0005)#, weight_decay=1e-7)

    history = []
    plot_history = []
    liveloss = PlotLosses(cell_size=(2, 2), max_cols = 7)

    for niter in range(1, max_iters + 1):
        optimizer.zero_grad()
        pred = model(z).exp() # Current model has log soft max at end
        #loss = z.norm()
        #loss, loss_dict = CFP_loss(prob_pred, target_hist, sorting, use_l2_loss, use_ae_loss)
        if sort_hist:
            vals, inds = pred.sort(descending=True)
        else:
            vals = pred
        #pdb.set_trace()
        dist_loss  = (z - x.detach()).norm()
        hist_loss  = (vals - target_hist).norm(p=1)
        reconst_loss = (z - ae_model(z)).norm()
        loss =  lambda_dist*dist_loss + hist_loss + lambda_rec*reconst_loss
        loss.backward()
        z.grad.view(d,d)[subset] = 0 # These pixels are fixed!
        optimizer.step()

        losses = [dist_loss.detach().numpy(), hist_loss.detach().numpy(), reconst_loss.detach().numpy()]
        history.append(losses)


        if niter % 100 == 0:
            prt_str = "Iter: {:5} [{}/{} ({:.0f}%)]\tHist Loss: {:.6f}".format(
                niter,niter,max_iters,100*niter/max_iters,hist_loss)
            if lambda_dist > 0:
                prt_str += "\tDist to x0: {:.6f}".format(dist_loss)
            if lambda_norm > 0:
                prt_str += "\tL2 Loss: {:.6f}".format(loss_dict['norm'])
            if lambda_rec > 0:
                prt_str += "\tAE Loss: {:.6f}".format(reconst_loss)
            #print(prt_str)#,, recon_loss.detach().numpy()))

        z.data.clamp_(x.min(), x.max())

        if plotting and (niter % plot_freq == 0 or (niter == 1)):
            liveloss.update({
                'loss_distance': losses[0],
                'loss_histogram': losses[1],
                'loss_reconstruction': losses[2],
                'plt_imshow_x': mnist_unnormalize(x.clone().detach().squeeze()),
                'plt_imshow_z': mnist_unnormalize(z.clone().detach().squeeze()),
                'plt_bar_z': pred.detach().t().numpy().squeeze(),
                'plt_bar_trg': target_hist.numpy().squeeze()
            })
            liveloss.draw()
            # fig, ax = plt.subplots(1,4, figsize = (15,3))
            # x_plot = mnist_unnormalize(x.clone().detach().squeeze())
            # z_plot = mnist_unnormalize(z.clone().detach().squeeze())
            # ax[0].imshow(x_plot, vmin=0, vmax = 1, cmap = 'Greys')
            # ax[1].imshow(z_plot, vmin=0, vmax = 1, cmap = 'Greys')
            # pred_plot = pred.detach().t().numpy().squeeze()
            # target_hist_plot =  target_hist.numpy().squeeze()
            # ax[2].bar(range(nclasses),pred_plot)
            # ax[3].bar(range(nclasses),target_hist_plot)
            # for k in [2,3]:
            #     ax[k].set_xticks(range(nclasses))
            #     ax[k].set_xticklabels(range(nclasses))
            #     ax[k].set_ylim(0,1)
            # plt.show()
        else:
            liveloss.update({
                'loss_distance': losses[0],
                'loss_histogram': losses[1],
                'loss_reconstruction': losses[2]
            })

            #plt.show()

            #plot_history.append([niter, x_plot, z_plot, pred_plot, target_hist_plot])


        if loss.detach().numpy() < 1e-10:
            break


    return z, history, plot_history



def projected_gradient(model, P, x, S, tol = 1e-5, maxiter = 100, plotting = True):
    d, _ = x.squeeze().shape
    #nclasses = model.net(torch.zeros(x.shape)).shape[-1] # FIXME: revert to below once using models trained with the __call__ attrivb
    print(torch.zeros(x.shape).shape)
    nclasses = model(torch.zeros(x.shape)).shape[-1]


    if plotting:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(x.view(d,d).detach().numpy().squeeze())
        ax[1].imshow(x.view(d,d)[S].detach().numpy().squeeze())
        ax[0].set_title('Input')
        ax[1].set_title('Fixed Attribute')
        ax[0].axis('off')
        ax[1].axis('off')
        plt.show()

    # Instantiate optimization variable
    z = mnist_normalize(torch.zeros(x.shape).detach())   ## FIXME: Must make this general. Maybe add to my trainer classes.
    z.view(d,d)[S].data.copy_(x.view(d,d)[S].data)
    #z.requires_grad_()

    delta = torch.ones(5,5)
    h = torch.zeros(1,1,d,d)
    alpha = 0.001
    t = 0
    while delta.norm() > tol and t < maxiter:
        grad = torch.zeros(d,d)
        grad[S] = (z.view(d,d)[S] - x.view(d,d)[S])
        h = z - alpha*grad
        z_new = P(h)
        #z_new = h
        delta = (z_new - z).detach()
        z = z_new
        print(delta.norm().item())
        loss =  (z.view(d,d)[S] - x.view(d,d)[S]).norm()**2
        print('Loss: ', loss.item())
        if plotting:
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(x.view(d,d).detach().numpy().squeeze())
            ax[1].imshow(h.view(d,d).detach().numpy().squeeze())
            ax[2].imshow(z.view(d,d).detach().numpy().squeeze())

            ax[0].set_title('Input')
            ax[1].set_title('h')
            ax[2].set_title('z')

            for i in range(3):
                ax[i].axis('off')
            plt.show()
        t += 1

    print('Projected gradient converged!')
    if plotting:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(x.view(d,d).detach().numpy().squeeze())
        ax[1].imshow(z.view(d,d).detach().numpy().squeeze())
        ax[0].set_title('Input')
        ax[1].set_title('Output')
        ax[0].axis('off')
        ax[1].axis('off')
        plt.show()
    return z
