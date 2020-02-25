"""
    Miscelanous or deprecated things. Check that none of these are used by actual methods.
"""

####################################################################
################ CONTRACTION / FIXED POINT TOOLS ###################
####################################################################

def eval_contractiveness(f, data_loader):
    total = 0
    f.eval()
    for batch_idx, (X,_) in enumerate(data_loader):
        #data, target = data.to(self.device), data.to(self.device)
        X_r = f(X)
        b = X.shape[0]
        X1, X2 = torch.split(X, [b/2, b/2], dim = 0)
        X1_r, X2_r = torch.split(X_r, [b/2, b/2], dim = 0)
        ratios = (X1_r - X2_r).view(int(b/2),-1).norm(dim=1) / \
                 (X1 - X2).view(int(b/2),-1).norm(dim=1)
        #print(ratios.sum())
        total += ratios.sum()

    return total/len(data_loader.dataset)


def find_fixed_point(f, x0, maxiters = 20, tol = 1e-04, plot = True):
    f.eval()
    xp = x0.clone()

    # if plot:
    #     fig, ax = plt.subplots(1, 2, figsize = (8,4))
    #     ax[0].imshow(x0.squeeze())
    #     im = ax[1].imshow(xp.squeeze())
    #     ax[0].axis('off')
    #     ax[1].axis('off')
    #     plt.show()

    for i in range(maxiters):
        xp_new = f(xp).reshape(1,28,28)
        delta = (xp_new - xp).norm().detach().numpy()
        if plot:
            clear_output(wait=True)
            #im.set_array(xp_new.detach().squeeze())
            fig, ax = plt.subplots(1, 2, figsize = (8,4))
            ax[0].imshow(x0.squeeze())
            im = ax[1].imshow(xp_new.detach().squeeze())
            ax[0].axis('off')
            ax[1].axis('off')
            ax[1].set_title('Iter {}, ||f(x(t)) - x(t)|| = {:.2e}'.format(i, delta))
            plt.show()
            #ax[1].draw()
            #plt.draw()
            #plt.imshow()
            #plt.axis('off')
            #plt.show()
        else:
            print(delta)
        xp = xp_new
        if delta < tol:
            break
    return xp


def contraction_plot(f, X, Y, maxit=20):
    """
            - examples: list of tuples [(k, x)]
    """
    #ncol = len(examples)
    ncol = X.shape[0]
    fig, ax = plt.subplots(1,ncol,figsize = (2*ncol, 2))
    for i in range(X.shape[0]):
        ax[i].imshow(X[i].squeeze())
        ax[i].axis('off')
    plt.show()

    for it in range(1, maxit+1):
        fig, ax = plt.subplots(1, ncol, figsize = (2*ncol,2))
        X_new = f(X)#.reshape(4,1,28,28)
        change = (X_new - X).reshape(ncol,-1).norm(dim=1)
        for i in range(ncol):
            ax[i].imshow(X_new[i].detach().squeeze())
            ax[i].set_title('{:4.2f}'.format(change[i].detach().numpy()))
            ax[i].axis('off')
        plt.show()

        printstr = ["Iter: {:d}".format(it)]
        for i in range(ncol):
            for j in range(i+1, ncol):
                lip  = ((X_new[j] - X_new[i]).norm()/(X[j] - X[i]).norm()).detach().numpy()
                printstr.append('Lip({},{}): {:4.2f}'.format(Y[i], Y[j], lip))

        print('\t'.join(printstr))
        #print("Iter: {0:d}\t Lip({1},{1}): {3:4.2f}\t Lip({2},{2}): {4:4.2f}"
        #"\t Lip({1},{2}): {5:4.2f}".format(it, classes[i], classes[j], lips[], lips[], lips))
        X = X_new

            #
        # lip_c1  = ((X_new[1] - X_new[0]).norm()/(X[1] - X[0]).norm()).detach().numpy()
        # lip_c2  = ((X_new[3] - X_new[2]).norm()/(X[3] - X[2]).norm()).detach().numpy()
        # lip_c12 = ((X_new[2] - X_new[0]).norm()/(X[2] - X[0]).norm()).detach().numpy()



    return X



class MCExplainer(object):
    """
        Multi-step Contrastive Explainer.


        - alpha, p are parameters for the explanation scoring function - see their description there
    """
    def __init__(self, classifier, mask_model, classes, loss_type = 'norm_delta',
                reg_type = 'decay', alpha=1, p=1, plot_type = 'bar'):
        self.classifier = classifier
        self.mask_model = mask_model
        self.classes    = classes
        self.plot_type  = plot_type
        self.loss_type  = loss_type
        self.reg_type   = reg_type
        self.alpha      = alpha
        self.p          = p
        self.task       = mask_model.task
        self.input_size = mask_model.input_size
        #self.input_size = mask_model.image_size # Legacy. Replace by above once new models trainerd.
        #self.pykernel   = detect_kernel()

        if self.task == 'mnist':
            self.masker = mnist_masker#, h = model.mask_size, w = model.mask_size)
            self.input_type = 'image'
        elif self.task in ['hasy', 'leafsnap']:
            H, W = self.input_size
            self.masker = partial(image_masker_unnormalized, H=H, W=W)# #partial(hasy_masker)#,  h = model.mask_size, w = model.mask_size)    )
            self.input_type = 'image'
        elif self.task == 'ets':
            self.masker = ets_masker
            self.input_type = 'text'
        else:
            raise ValueError('Unrecognized task.')
        self.criterion = self._init_criterion()


    def _init_criterion(self):

        loss = self.loss_type

        if loss == 'norm_delta':
            crit = partial(normalized_deltas, reg_type = self.reg_type,
                    alpha=self.alpha, p=self.p, argmin_reg = None,
                    include_pred= True)
        else:
            raise ValueError('Not implemented yet')
        #nclasses = len(V)

        # if loss == 'euclidean':
        #     # Target Histrogram
        #     target_hist = torch.zeros(1, nclasses).detach()
        #     target_hist[:,tgt_classes] = 1
        #     target_hist /= target_hist.sum()
        #     hist_loss  = (vals - target_hist).norm(p=p)
        # elif loss == 'delta':
        #     #hist_loss  = torch.abs(delta_fun(vals) - delta_fun(target_hist))
        #     hist_loss = -delta_fun(vals) # Minus because want to maximize delta
        # elif loss == 'cumul_delta':
        #     hist_loss = -cumul_delta_fun(vals, verbose = verbose) # Minus because want to maximize delta
        # elif loss == 'norm_power':
        #     obj, C = normalized_power(vals)
        #     hist_loss = -obj
        # elif loss == 'norm_delta':
        #     obj, C = normalized_deltas(vals)#, all_classes)
        #     hist_loss = -obj

        return crit

    #@torch.no_grad() # FIXME - how to do this without global torch import
    def explain(self, x, y = None, verbose = False, show_plot = 1, save_path = None):
        # if y not provided, call classif model
        if y is None:
            out = self.classifier(x)
            p, y = out.max(1)
            y = y.item()

        if show_plot > 1 and self.input_type == 'image':
            plt.imshow(x.squeeze(), cmap='Greys')
            plt.title('Prediction: ' + self.classes[y] + ' (class no.{})'.format(y), fontsize = 20)
            plt.xticks([])
            plt.yticks([])
            plt.show()
        else:
            print(y)
            print(self.classes)
            print('Prediction: ' + self.classes[y] + ' (class no.{})'.format(y))
            #print('hasdasd')

        V = list(range(len(self.classes)))
        E = Explanation(x, y, self.classes, masker = self.masker, task = self.task)
        step = 0
        while len(V) > 1:
            print('Explanation step: ', step)

            if verbose > 1:
                print(V)

            #max_S, max_C, hist_V = # Once attrib_optimizer is created at init with partial: self.attrib_optimizer(x, V, show_plot = show_plot, verbose = max(0, verbose-1))
            if self.task in ['mnist', 'hasy', 'leafsnap']:
                max_S, max_C, hist_V = self.optimize_over_rectangle_attributes(
                                        self.mask_model, self.criterion, self.masker,
                                        x, y, V=V, tgt_classes = [2,3,4], sort_hist = False,
                                        show_plot = show_plot, plot_type = self.plot_type,
                                        loss = self.loss_type, class_names = self.classes,
                                        force_include_class = y, verbose = max(0, verbose -1)
                                        )
            elif self.task == 'ets':
                E.vocab = self.mask_model.vocab
                max_S, max_C, hist_V = self.optimize_over_ngram_attributes(
                                        self.mask_model, self.criterion, self.masker,
                                        x, y, V=V, tgt_classes = [2,3,4], sort_hist = False,
                                        show_plot = show_plot, plot_type = self.plot_type,
                                        loss = self.loss_type, class_names = self.classes,
                                        force_include_class = y, verbose = max(0, verbose -1)
                                        )
            else:
                raise ValueError("Wrong task")
            if max_S is None:
                print('Explanation failed')
                break
            if verbose > 1:
                print('Entailed classes: ')
                print([self.classes[c] for c in max_C])

            if len(max_C) == len(V) or len(max_C) == 0:
                print('Here')
                break

            # Remove classes from reference set
            V_prev = V.copy()
            if y in max_C:
                # predicted class in entailed set - keep that
                V = [k for k in V if k in max_C]  # np version: np.isin(V, max_C)
            else:
                # predicted class not in entailed set, remove entailed set
                V = [k for k in V if k not in max_C]
            if verbose > 0:
                print('Shrinking ground set: {} -> {}'.format(V_prev, V))

            predicate = AttrDict({'S': max_S, 'hyp': C, 'null_hyp': list(set(V).difference(set(C))),
                                  'hist': hist, 'V_prev': V_prev, 'V': V})


            E.predicates.append(predicate)
            step += 1

        return E

    #TODO: do we need static? Maybe make just normal with self, to avoid having to pass crit function
    @staticmethod
    def optimize_over_rectangle_attributes(model, criterion, masker, x, y, V = None, tgt_classes = [1,2],
                                 sort_hist = False, show_plot = False, plot_type = 'treemap', class_names = None,
                                 loss = 'euclidean', force_include_class = None, verbose = False):
        """
            Given masking model, input x and target histogram, finds subset of x (i.e., attribute)
            that minimizes loss with respect to target historgram.

             - V: ground set of possible classes
            plot: 0/1/2 (no plotting, plot only final, plot every iter)

        """
        model.net.eval()
        H, W = model.input_size
        h, w = model.mask_size
        max_obj, max_S = float("-inf"), None
        if V is None:
            V = np.array(list(range(10)))
        elif type(V) is list:
            V = np.array(V)
        if force_include_class:
            assert force_include_class in V, "Error: class to be force-included not in ground set"

        if KERNEL == 'terminal':
            #print("PLT ION")
            fig, ax = plt.subplots(1, 3, figsize = (4*3,4))
            plt.ion()
            plt.show()
            ims = None
        else:
            fig, ax, ims = None, None, None


        def handler(*args):
            print('Ctrl-c detected: will stop iterative plotting')
            nonlocal show_plot
            show_plot = min(show_plot, 1)

        #signal.signal(signal.SIGINT, handler)
        #print(KERNEL)
        if KERNEL == 'terminal': pbar = tqdm(total=(W-w)*(H-h))
        for (ii,jj) in itertools.product(range(0, W-w),range(0,H-h)):
                #print(ii,jj)
                if KERNEL == 'terminal': pbar.update(1)
                S = np.s_[ii:min(ii+h,H),jj:min(jj+w,W)]
                X_s, X_s_plot = masker(x, S)
                output = model(X_s)
                hist = output.detach().numpy().squeeze()
                hist_V = hist[V]
                obj, C_rel = criterion(hist_V, pred = y, V = V)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)

                # Convert from relative indices to true classes
                 #TODO: Maybe move this to criterion?
                C = V[C_rel]

                labstr = 'Current objective: {:8.4e} (i = {:2}, j = {:2})'.format(obj.item(), ii, jj)
                if show_plot > 1:
                    if KERNEL == 'ipython':
                        clear_output(wait=True)
                        ax = None

                    ax, rects, ims = plot_2d_attrib_entailment(x, X_s_plot, S, C, V, hist_V,
                                        plot_type = plot_type, class_names = class_names,
                                        title = labstr, ax = ax, ims =ims, return_elements = True
                                        )

                    if KERNEL == 'terminal':
                        plt.draw()
                        plt.pause(.005)
                        #rects[0].remove()
                        ax[0].clear() # Alternatively, do rects[0].remove(), though its slower
                        ax[1].clear()
                        ax[2].clear()
                    else:
                        plt.show()
                elif verbose:
                    print(labstr)

                if force_include_class is not None and (not force_include_class in C):
                    # Target class not in entailed set
                    #print(force_include_class, C)
                    #print(obj, C_rel)
                    #global DEBUG
                    #DEBUG = True
                    #obj, C_rel = criterion(hist_V, pred = y, V = V)#, verbose = True)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
                    obj = float("-inf")
                    continue
                if obj > max_obj:
                    max_obj = obj
                    max_S   = S

        if KERNEL == 'terminal': pbar.close()
        if max_S is None:
            print('Warning: could not find any attribute that causes the predicted class be entailed')
            return None, None, None

        if KERNEL == 'terminal':
            plt.ioff()
            plt.close('all')

        # Reconstruct vars for optimal
        X_s, X_s_plot = masker(x, max_S)
        output = model(X_s)
        hist = output.detach().numpy().squeeze()
        hist_V = hist[V]
        obj, C_rel = criterion(hist_V, pred = y, V=V)#, sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
        max_C = V[C_rel]

        ii, jj =  list(range(H)[max_S[0]])[0], list(range(W)[max_S[1]])[0]

        assert obj == max_obj
        labstr = 'Best objective: {:8.4f} (i = {:2}, j = {:2})'.format(max_obj.item(), ii, jj)
        #clear_output(wait=True)
        if show_plot:
            plot_2d_attrib_entailment(x, X_s_plot, max_S, max_C, V, hist_V, plot_type=plot_type,
            class_names = class_names,
            topk = 10, title = labstr, sort_bars = True)
            plt.show()
        else:
            print(labstr)

        return max_S, max_C, hist_V

    @staticmethod
    def optimize_over_ngram_attributes(model, criterion, masker, x, y, V = None,
                                 tgt_classes = [1,2], sort_hist = False,
                                 show_plot = False, plot_type = 'treemap',
                                 class_names = None, loss = 'euclidean',
                                 force_include_class = None, verbose = False):
        """
            Given masking model, input x and target histogram, finds subset of x
            (i.e., attribute) that minimizes loss with respect to target historgram.

             - V: ground set of possible classes
            plot: 0/1/2 (no plotting, plot only final, plot every iter)

        """
        model.net.eval()
        N = x.shape[1]
        n = model.mask_size
        max_obj, max_S = 100*torch.ones(1), None
        if V is None:
            V = np.array(list(range(10)))
        elif type(V) is list:
            V = np.array(V)
        if force_include_class:
            assert force_include_class in V, "Error: class to be force-included not in ground set"

        if KERNEL == 'terminal':
            fig, ax = plt.subplots(1, 2, figsize = (10,4),
                                gridspec_kw = {'width_ratios':[2, 1]})
            plt.ion()
            plt.show()
            ims = None
        else:
            fig, ax, ims = None, None, None

        def handler(*args):
            print('Ctrl-c detected: will stop iterative plotting')
            nonlocal show_plot
            show_plot = min(show_plot, 1)

        signal.signal(signal.SIGINT, handler)

        for ii in range(0, N-n):
            #print(ii)
            S = np.s_[range(ii,ii+n)]
            X_s    = masker(x, S)
            output = model(X_s)
            hist = output.detach().numpy().squeeze()
            hist_V = hist[V]
            obj, C_rel = criterion(hist_V, pred = y, V = V)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
            #obj, C_rel = criterion(hist_V, V=V)#sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
            # Convert from relative indices to true classes
             #TODO: Maybe move this to criterion?
            C = V[C_rel]
            labstr = 'Ngram: [{}:{}], Objective: {:8.4e}'.format(ii, ii+n,obj.item())
            ngram_str = ' '.join([model.vocab.itos[w.item()] for w in X_s[0]])
            if show_plot > 1:
                if KERNEL == 'ipython':
                    clear_output(wait=True)
                    ax = None

                ax = plot_text_attrib_entailment(x, X_s, S, C, V, hist_V,
                                    plot_type = plot_type,
                                    class_names = class_names,
                                    vocab = model.vocab, ax = ax,
                                    title = labstr, sort_bars = False)
                if KERNEL == 'terminal':
                    plt.draw()
                    plt.pause(.005)
                    ax[0].clear()
                    ax[1].clear()
                else:
                    plt.show()
            elif verbose:
                #ngram_str = ' '.join([model.vocab.itos[w.item()] for w in X_s[0]])
                #labstr = 'Ngram: {} ([{}:{}]), Objective: {:8.4e}'.format(ngram_str, ii, ii+n,obj.item())
                print(labstr)

            if force_include_class is not None and (not force_include_class in C):
                # Target class not in entailed set
                #print(force_include_class, C)
                obj = float("-inf")
                continue
            if obj < max_obj:
                max_obj = obj
                max_S   = S
                #clear_output(wait=True)

        if max_S is None:
            print('Warning: could not find any attribute that causes to predicted class be entailed')
            return None, None, None

        if KERNEL == 'terminal':
            plt.ioff()
            plt.close('all')

        # Reconstruct vars for optimal
        X_s = masker(x, max_S)
        output = model(X_s)
        hist = output.detach().numpy().squeeze()
        hist_V = hist[V]
        obj, C_rel = criterion(hist_V, pred = y, V = V)#, sort_hist = sort_hist, tgt_classes = tgt_classes, p = 1, loss = loss)
        max_C = V[C_rel]
        #print(asd.asd)
        ii = max_S[0]

        assert obj == max_obj
        ngram_str = ' '.join([model.vocab.itos[w.item()] for w in X_s[0]])
        labstr = 'Best objective: {:8.4f}, (i = {:2})'.format(max_obj.item(),ii)
        #clear_output(wait=True)
        if show_plot:
                plot_text_attrib_entailment(x, X_s, max_S, max_C, V, hist_V,
                                    plot_type = plot_type,
                                    class_names = class_names,
                                    vocab = model.vocab,
                                    title = labstr, sort_bars = False)
                plt.tight_layout()
                plt.show()
        else:
            print(labstr)
        return max_S, max_C, hist_V

    # def criterion(self, pred, V, tgt_classes = [7,9], sort_hist = False, p = 1, loss = 'delta', verbose = False):
    #     nclasses = len(V)
    #     if sort_hist:
    #         vals, inds = pred.sort(descending=True)
    #     else:
    #         vals = pred
    #     if loss == 'euclidean':
    #         # Target Histrogram
    #         target_hist = torch.zeros(1, nclasses).detach()
    #         target_hist[:,tgt_classes] = 1
    #         target_hist /= target_hist.sum()
    #         hist_loss  = (vals - target_hist).norm(p=p)
    #     elif loss == 'delta':
    #         #hist_loss  = torch.abs(delta_fun(vals) - delta_fun(target_hist))
    #         hist_loss = -delta_fun(vals) # Minus because want to maximize delta
    #     elif loss == 'cumul_delta':
    #         hist_loss = -cumul_delta_fun(vals, verbose = verbose) # Minus because want to maximize delta
    #     elif loss == 'norm_power':
    #         obj, C = normalized_power(vals)
    #         hist_loss = -obj
    #     elif loss == 'norm_delta':
    #         obj, C = normalized_deltas(vals)#, all_classes)
    #         hist_loss = -obj
    #     return hist_loss, C
