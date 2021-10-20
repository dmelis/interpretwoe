import pdb
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
import sklearn.naive_bayes


DEBUG = False

from .woe_utils import invlogit

class woe_wrapper():
    """
        Child methods should specify the following methods:
            - log_prob (computes loglikelhood of X given Y)
            - log_prob_partial (computes loglikelhood of X_S given Y)
        and should instantiate the following attributes:
            - priors (nparray of class prior probabilities)
    """
    def __init__(self, model, task, classes, input_size, complex_hyp_method='mixture'):
        self.model = model
        self.classes = classes
        self.task  = task
        self.input_size = input_size
        self.data_transform = None
        self.cache = None
        self.caching = False
        self.priors = None
        self.debug = False
        self.complex_hyp_method = complex_hyp_method

    def _start_caching(self):
        self.caching = True
        self.cache = {}

    def _stop_caching(self):
        self.caching = False
        self.cache = None

    def _process_inputs(self,x=None,y1=None, y2=None, subset=None):
        """ All other functions will assume:
                - x is (n,d) 2-dim array
                - y1, y2 are np.arrays with indices of classes
        """
        #pdb.set_trace()
        def _process_hypothesis(y):
            if type(y) is int:
                y = np.array([y])
            elif type(y) is list:
                y = np.array(y)
            elif type(y) is set:
                y = np.array(list(y))
            return y

        if y1 is not None:
            y1 = _process_hypothesis(y1)

        if y2 is None and y1 is not None:
            # Default to complement as alternate hypothesis
            y2 = np.array(list(set(self.classes).difference(set(y1))))
        elif y2 is not None:
            y2 = _process_hypothesis(y2)

        if x is not None:
            if x.ndim == 1:
                # Most functions expect more than one sample, so need to reshape.
                #assert x.shape[0] == self.d, (x.shape, self.d)
                #print(x.shape)
                x = x.reshape(1,-1)

        if subset is not None:
            if type(subset) is list:
                subset = np.array(subset)
            elif type(subset) is set:
                subset = np.array(list(subset))

        return x, y1, y2, subset

    def woe(self, x, y1, y2=None, subset=None, **kwargs):
        x,y1,y2,subset = self._process_inputs(x,y1,y2,subset)
        ll_num   = self.generalized_conditional_ll(x, y1, subset, **kwargs)
        ll_denom = self.generalized_conditional_ll(x, y2, subset, **kwargs)
        woe = ll_num - ll_denom
        # print('Num', ll_num)
        # print('Denom', ll_denom)
        return woe

    def _model_woe(self, x, y1, y2=None, **kwargs):
        """ Compute WoE as difference of posterior and prior log odds"""
        postodds = self.posterior_lodds(x, y1, y2)
        priodds  = self.prior_lodds(y1, y2)
        return postodds - priodds

    def prior_lodds(self, y1, y2=None):
        _,y1,y2,_ = self._process_inputs(None,y1,y2)
        odds_num = self.priors[y1]
        if len(y1) > 1:
            odds_num = odds_num.sum()
        else:
            odds_num = odds_num.item()
        odds_den = self.priors[y2]
        if len(y2) >1:
            odds_den = odds_den.sum()
        else:
            odds_den = odds_den.item()
        return np.log(odds_num/odds_den)

    def prior(self, ys):
        return self.priors[ys]

    def log_prior(self, ys):
        return np.log(self.priors[ys])

    def posterior_lodds(self, x, y1, y2=None, eps = 1e-12):
        x,y1,y2,_ = self._process_inputs(x,y1,y2)

        probs = self.model.predict_proba(x.reshape(1,-1))[0]

        odds_num = probs[y1]
        if len(y1) > 1: odds_num = odds_num.sum()

        odds_den = probs[y2]
        if len(y2) > 1: odds_den = odds_den.sum()

        odds_num = np.clip(odds_num, eps, 1-eps)
        odds_den = np.clip(odds_den, eps, 1-eps)

        return np.log(odds_num/odds_den)

    def log_posterior(self, x, y):
        # Mostly for debugging
        # log p( y | x) = log (x | y) + log p(y) - log Σ_y exp(log p(x|y))
        logpart = logsumexp(np.array([self.log_prob(x, c)+self.log_prior(y) for c in self.classes]))
        logpost = self.log_prob(x, y) + self.log_prior(y) - logpart
        return logpost

    def log_prob(self, x, y):
        """Compute log density of X conditioned on Y.

        Parameters
        ----------
        x : array-like, shape = [n_subset]
            Input
        y : int

        Returns
        -------
        log probability : float
        """
        pass

    def generalized_conditional_ll(self, x, y, subset=None, verbose = False, **kwargs):
        """A wrapper to handle different cases for conditional LL computation:
        simple or complex hypothesis.

        Parameters
        ----------
        x : array-like, shape = [n_subset]
            Input
        y : int or list of ints
            If just an int, this is simpl hypthesis ratio, if list it's composite
        Returns
        -------
        log probability : float
        """
        k = len(self.classes)

        # We memoize woe computation for efficiency. For this, need determinsitc hash
        # This has to be done before doing view or reshape on x.
        hash_x = id(x) #hash(x)  - hash is non determinsitc!
        hash_S = -1 if subset is None else subset[0]#.item()
        if DEBUG and hash_S != -1: print('here', hash_S)

        if (not self.debug) and y.shape[0] == 1:
            if verbose: print('H is simple hypotesis')
            if subset is None:
                loglike = self.log_prob(x, y[0])
            else:
                loglike = self.log_prob_partial(x, y[0], subset)
            loglike = loglike.squeeze() # to get scalar, to be consistent by logsumexp below
        else:
            if verbose: print('H is composite hypotesis')

            if self.complex_hyp_method == 'average':
                #### Averaging Approach - Does satisfy sum desideratum
                if subset is None:
                    loglike = self.log_prob_complex(x, y)
                else:
                    loglike = self.log_prob_complex_partial(x, y, subset)
            elif self.complex_hyp_method == 'max':
                #### Argmax Approach - Doesnt satisfy sum desideratum
                if subset is None:
                    loglikes = np.array([self.log_prob(x, yi) for yi in y])
                else:
                    loglikes = np.array([self.log_prob_partial(x, yi, subset) for yi in y])
                loglike = np.max(loglikes, axis=0)
            elif self.complex_hyp_method == 'mixture':
                #### Mixture Model Approach - Doesnt satisfy sum desideratum
                # Need to compute log prob with respect to each class in y, then do:
                # log p(x|Y∈C) = log (1/P(Y∈C) * ∑_y p(x|Y=y)p(Y=y) )
                #              = log ∑_y exp{ log p(x|Y=y) + log p(Y=y)} - log P(Y∈C)
                priors       = self.prior(np.array(range(k)))
                logpriors    = np.log(priors[y])
                logprior_set = np.log(priors[y].sum())
                loglike      = np.zeros((x.shape[0], y.shape[0])) # size npoints x hyp size
                lognormprob  = np.log(priors[y]/priors[y].sum())

                for i, yi in enumerate(y):
                    if self.caching and ((hash_x,hash_S)in self.cache) and (yi in self.cache[(hash_x,hash_S)]):
                        if DEBUG and hash_S != -1: print('using cached: ({},{})'.format(hash_x, hash_S))
                        loglike[:,i] = self.cache[(hash_x,hash_S)][yi]
                    else:
                        # yi = int2hot(torch.tensor([yi]*x.shape[0]), k) -> for pytorch
                        if subset is None:
                            loglike[:,i] = self.log_prob(x, yi) + lognormprob[i]#logpriors[i] #- logprior_set
                        else:
                            loglike[:,i] = self.log_prob_partial(x, yi, subset) + lognormprob[i]#+ logpriors[i] #- logprior_set
                        if self.caching:
                            if not (hash_x,hash_S) in self.cache:
                                self.cache[(hash_x,hash_S)] = {}

                loglike = logsumexp(loglike, axis = 1) #- logprior_set

        return loglike

    def decomposition_woe(self, x, y1, y2, exid = None, plot = True, rgba= True, order = 'fwd', figsize = (10,10)):
        """
            Note that regardless or display and computation order, will return partial
            woes in "natural" order (L->R, T->D)
        """
        # TODO get these number from model attribs
        k = 7
        d = int(np.sqrt(self.input_size))#[0]
        input_order_idxs, block_order_idxs = get_block_order(7, d, order = order)

        total_woe = self.woe(x, y1, y2)#,  args)

        nblocks = d**2 // k**2
        nrowb = ncolb = int(np.sqrt(nblocks))

        partial_woes = torch.zeros(x.shape[0], nblocks)

        if plot:
            # If no specific index provided We'll display the examlpe with the max total woe
            if x.shape[0] == 1:
                exid = 0
            elif exid is None:
                max_woe, exid = total_woe.max(0)
            print(exid)
            if rgba:
                plt.imshow(invlogit(x[exid,:]).reshape(28,28))
            else:
                plt.imshow(invlogit(x[exid,:]).reshape(28,28), cmap = 'Greys')
            plt.title('Total Woe: {:8.2f}'.format(total_woe[exid]))
            plt.show()


        ll_num_prev = ll_denom_prev = torch.zeros_like(x[:,0]) - 10000000000

        attrib_idxs = []
        for nb in range(nblocks):
            # nb is sequential order, bidx is "true" block index, in LR TD order
            # these are equal for order == 'fwd', but different in other cases
            bidx = block_order_idxs[nb]
            I  = nb // (28//k)
            J  = nb % (28//k)
            init_idx = nb*(k**2)
            block_idxs = input_order_idxs[init_idx:init_idx+k**2] # Input indices for this block

            cumulative = True
            #print(block_idxs)
            ### DEBUG
            #cumul_block_idxs = input_order_idxs[:init_idx+k**2]
            if cumulative:
                ll_num   = self.generalized_conditional_ll(x, y1, block_idxs)
                ll_denom = self.generalized_conditional_ll(x, y2, block_idxs)
            else:
                ll_num   = (self.generalized_conditional_ll(x, y1, block_idxs).exp() - ll_num_prev.exp()).log()
                ll_denom = (self.generalized_conditional_ll(x, y2, block_idxs).exp() - ll_denom_prev.exp()).log()
                ll_num_prev = ll_num
                ll_denom_prev = ll_denom

            woe = ll_num - ll_denom
            partial_woes[:, bidx] = woe
            attrib_idxs.append(block_idxs)

        partial_woes = partial_woes.detach().numpy()
        total_woe = total_woe.detach().numpy()

        if plot:
            self.plot_partial_woe(x[exid,:].squeeze(), partial_woes[exid,:].squeeze(), k, d, order)


        return total_woe, partial_woes, attrib_idxs #print(partial_woes.sum(axis=1).shape)

    @staticmethod
    def plot_partial_woe(x, partial_woes, block_width, input_width, order, rgba=False, figsize = (10,10)):
        if x.ndimension() > 1:
            x = x.view(-1) #.to(args.device)
        if rgba:
            cmap = plt.get_cmap()
            full_img = cmap(x)
            partial_img = 0*full_img
        else:
            full_img = x
            partial_img = full_img*0 + .1

        input_order_idxs, block_order_idxs = get_block_order(block_width, input_width, order = order)
        nblocks = input_width**2 // block_width**2
        nrowb = ncolb = int(np.sqrt(nblocks))

        plt.figure(figsize = figsize)
        gs1 = gridspec.GridSpec(nrowb, ncolb)
        gs1.update(wspace=0.00, hspace=0.2) # set the spacing between axes.

        #pdb.set_trace()

        woe_ranks = sp.stats.rankdata(-partial_woes, method = 'ordinal')
        for nb in range(nblocks):
            bidx = block_order_idxs[nb]
            I  = nb // (28//block_width)
            J  = nb % (28//block_width)
            init_idx = nb*(block_width**2)
            block_idxs = input_order_idxs[init_idx:init_idx+block_width**2]
            #ax = plt.subplot(gs1[nb]) if not reverse else plt.subplot(gs1[nblocks - nb - 1])
            ax = plt.subplot(gs1[bidx]) # Plot panes will have natural order hence bidx not nb
            partial_img[block_idxs] = full_img[block_idxs]
            if rgba:
                ax.imshow(partial_img.reshape(28,28,4))
            else:
                ax.imshow(partial_img.reshape(28,28), cmap = 'Greys', vmin = 0, vmax = 1)
            plt.axis('on')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

            ii, jj = np.unravel_index(min(block_idxs), (28,28))
            w, h   = block_width, block_width
            rects = patches.Rectangle((jj-.5,ii-.5),w,h,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rects)
            title = '{:2.2f}'.format(partial_woes[bidx])
            if partial_woes[bidx] > 10:
                title = r"$\bf{" + title + "}$"
            ax.set_title('Woe ' + title + ' ({}/{})'.format(woe_ranks[bidx], nblocks))

            # Grey out these pixels.
            if rgba:
                partial_img[block_idxs,3] *= 0.5
            else:
                partial_img[block_idxs] *= 0.05

        plt.show()

class woe_gaussian(woe_wrapper):
    def __init__(self, clf, X, task=None, classes=None, input_size=None, cond_type='full'):
        super().__init__(clf, task, classes, input_size)
        self.cond_type = cond_type # 'nb' or 'full'
        self.fit(X,clf.predict(X))

    def _process_inputs_depr(self, x = None, y = None, subset = None):
        """
        """
        if x.ndim == 1:
            # Most functions expect more than one sample, so need to reshape.
            assert x.shape[0] == self.d
            #print(x.shape)
            x = x.reshape(1,-1)
        if type(y) in [np.ndarray, list]:
            assert len(y) == 1
            y = y[0]
        if subset is None:
            return x,y
        else:
            return x,y,subset

    def fit(self, X, Y, eps=1e-8, shift_cov = 0.01):
        self.n, self.d = X.shape
        means = []
        covs  = []
        inv_covs = []
        log_det_covs = []
        distribs = []
        class_prior = []
        for c in self.classes:
            class_prior.append((Y==c).sum()/len(Y))
            μ = X[Y==c].mean(axis=0)
            means.append(μ)
            Σ = np.cov(X[Y==c].T)
            delta = max(eps - np.linalg.eigvalsh(Σ).min(), 0)
            Σ += np.eye(self.d)*1.1*delta # add to ensure PSD'ness
            Σ += shift_cov*np.eye(self.d) # TODO: find more principled smoothing
            try:
                assert np.linalg.eigvalsh(Σ).min() >= eps
            except:
                pdb.set_trace()
            if self.cond_type == 'nb':
                # Store digonals only
                Σ = np.diag(Σ)
                Σ_inv = 1/Σ
                logdet = np.log(Σ).sum()
                try:
                    distribs.append(multivariate_normal(μ,Σ))
                except:
                    pdb.set_trace(header='Failed at creating normal distrib')
            else:
                # Approach 1: Direct
                #Σ_inv = np.linalg.inv(Σ) # Change to pseudoinverse alla scipy
                #s, logdet = np.linalg.slogdet(Σ)
                #assert s >= 0
                # Approach 2: Alla scipy - with (more robust) psuedoinverse
                U,logdet =  _psd_pinv_decomposed_log_pdet(Σ, rcond = 1e-12)
                # If we use this - maybe move to storting U instead?
                Σ_inv = np.dot(U,U.T)

                try:
                    distribs.append(multivariate_normal(μ,Σ))
                except:
                    pdb.set_trace()


            covs.append(Σ)
            inv_covs.append(Σ_inv)
            log_det_covs.append(logdet)
            assert not np.isnan(Σ).any()

        self.means    = np.stack(means)
        self.covs     = np.stack(covs)
        self.inv_covs = np.stack(inv_covs)
        self.log_det_covs = np.stack(log_det_covs)
        self.distribs = distribs
        self.priors = np.array(class_prior)

        self.dl2π = self.d * np.log(2.*np.pi)


    def log_prob(self, x, y):
        """
            TODO: use stored log_det_covs, inv_covs for efficiency
        """
        x = self._process_inputs(x)[0]
        assert isinstance(y, (int, np.integer))

        μ = self.means[y,:]
        Σ = self.covs[y,:]

        if self.cond_type  == 'nb':
            logp = gaussian_logdensity(x, μ, Σ, independent=True)
        else:
            logp = gaussian_logdensity(x, μ, Σ, independent=False)

        return logp

    def log_prob_partial(self, x, y, subset):
        x,_,_,subset = self._process_inputs(x,subset=subset)
        assert isinstance(y, (int, np.integer)), y

        if self.cond_type == 'nb':
            # With Naive Bayes assumption, P(X_i | X_j, Y ) = P(X_i | Y),
            # so it suffices to marginalize over subset of features.
            μ = self.means[y,:]
            Σ = self.covs[y,:]
            logp = gaussian_logdensity(x, μ, Σ, marginal=subset, independent=True)

        elif self.cond_type == 'full':
            S = subset                            # S, the unconditioned variables
            T = list(set(range(self.d)) - set(S)) # Complement of S, conditioning variables

            ### Get Covariance
            # Might want to memoize/cache some of these? Sigma is indep of x.
            # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
            # Approach 1: Direct
            Σs  = self.covs[y,S][:,S]
            Σst = self.covs[y,S][:,T]          #  (|S|,|T|)
            Σt_inv = np.linalg.inv(self.covs[y,T][:,T])     #  (|T|,|T|)
            Σ_stinv = np.dot(Σst, Σt_inv)       # (|S|,|T|)
            Σ = Σs - np.dot(Σ_stinv, Σst.T)

            # Approach 2: Uses Schur Complement Identity, but we would still need to compute Σ_stinv to compute mean
            #Σb = np.linalg.inv(self.inv_covs[y,S][:,S])

            assert Σ.shape == (len(S), len(S))
            assert (np.linalg.eig(Σ)[0] >= 0).all(), 'Marginalized cov not psd'

            ### Get Mean
            diff_T = x[:,T] - self.means[y,T]   # (n,|T|) (broadcasting)
            μ = self.means[y,S] - np.dot(diff_T, Σ_stinv.T)   # (n, |S|)
            assert μ.shape == (n, len(S))

            Σ_inv = self.inv_covs[y,S][:,S] # =  np.linalg.inv(Σ) by schur complement
            Σ_logdet = np.linalg.slogdet(Σ)[1]

            logp = gaussian_logdensity(x, μ, Σ, independent=False)


        else:
            raise(ValueError)


        return logp


    def log_prob_old(self, x, y):
        """
        """
        x = self._process_inputs(x)[0]
        assert isinstance(y, (int, np.integer))

        ct   = self.d * np.log(2. * np.pi) + self.log_det_covs[y]
        diff = x - self.means[y, :]   # (x - μ_j) , size (n,d)

        if self.cond_type == 'nb': # Covariance is diagonal
            expterm = np.sum((diff ** 2) / (self.covs[y,:]), 1)
        else:
            expterm = np.diag(np.dot(np.dot(diff, self.inv_covs[y,:]),diff.T))
            # TODO: Faster and cleaner way to do this by storing sqrt of precision matrix. See
            # https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/_multivariate.py line 330

        # Debug
        logp = -0.5 * (ct + expterm)
        #print(self.distribs[y].pdf(x), self.distribs[y].logpdf(x))

        assert np.allclose(self.distribs[y].logpdf(x), logp), (self.distribs[y].logpdf(x), logp)

        return logp

    def log_prob_partial_old(self, x, y, subset):
        x,_,_,subset = self._process_inputs(x,subset=subset)
        assert isinstance(y, (int, np.integer))

        S = subset                            # S, the unconditioned variables
        T = list(set(range(self.d)) - set(S)) # Complement of S, conditioning variables

        n = x.shape[0]

        diff = x[:,subset] - self.means[y, subset]   # (n,|S|)

        if self.cond_type == 'nb':
            # Using naive Bayes, we do P(X_s | X_s1,...X_s, Y) = P (X_s, Y)
            # for GNB, this is easy: just drop non-included vars
            # en.wikipedia.org/wiki/Multivariate_normal_distribution#Marginal_distributions
            μ        = self.means[y,S]
            Σ_ss     = self.covs[y,S]

            #pdb.set_trace()
            #Σ_inv    = np.linalg.inv(Σ_ss)
            #if Σ_ss.ndim == 2:
                # We have non-indep covariance, but are making NB assumption
            Σ_logdet = np.log(Σ_ss).sum()

            diff = x[:,S] - μ
            expterm = np.sum((diff ** 2) / Σ_ss, 1)

        elif self.cond_type == 'full':

            ### Get Covariance
            # Might want to memoize/cache some of these? Sigma is indep of x.
            # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
            # Approach 1: Direct
            Σs  = self.covs[y,S][:,S]
            Σst = self.covs[y,S][:,T]          #  (|S|,|T|)
            Σt_inv = np.linalg.inv(self.covs[y,T][:,T])     #  (|T|,|T|)
            Σ_stinv = np.dot(Σst, Σt_inv)       # (|S|,|T|)
            Σ = Σs - np.dot(Σ_stinv, Σst.T)

            # Approach 2: Uses Schur Complement Identity, but we would still need to compute Σ_stinv to compute mean
            #Σb = np.linalg.inv(self.inv_covs[y,S][:,S])

            assert Σ.shape == (len(S), len(S))
            assert (np.linalg.eig(Σ)[0] >= 0).all(), 'Marginalized cov not psd'

            ### Get Mean
            diff_T = x[:,T] - self.means[y,T]   # (n,|T|) (broadcasting)
            μ = self.means[y,S] - np.dot(diff_T, Σ_stinv.T)   # (n, |S|)

            assert μ.shape == (n, len(S))

            Σ_inv = self.inv_covs[y,S][:,S] # =  np.linalg.inv(Σ) by schur complement
            Σ_logdet = np.linalg.slogdet(Σ)[1]

            diff = x[:,S] - μ
            expterm = np.diag(np.dot(np.dot(diff, Σ_inv),diff.T))

        else:
            raise(ValueError)

        ct   = len(S) * np.log(2. * np.pi) +  Σ_logdet
        logp = -0.5 * (ct + expterm)

        ### For continuous vars, pdf can be > 1 !!
        # try:
        #     assert (logp <= 0).all(), logp
        # except:
        #     pdb.set_trace()

        return logp


    def covariance_average(self, covs, means=None, weights=None, method = 'am'):
        if method == 'am':
            # Method 1: Arithmetic Mean
            Σ = np.average(covs, axis = 0, weights=weights)
            #print('Σ Method 1', Σ)
        elif method == 'empirical':
            #Σ = np.average(covs, axis = 0, weights=weights)
            #print('Σ Method 1', Σ[:3])
            # Method 2: Recreate empirical covarance matrix
            μ = np.average(means, axis = 0, weights=weights)
            sum = 0
            for i in range(covs.shape[0]):
                Σi = covs[i,:] if covs[i,:].ndim ==2 else np.diag(covs[i,:])
                sum += weights[i]*(Σi + np.dot(means[i,:].T, means[i,:]))
            Σ = sum - np.dot(μ,μ.T)
            if self.cond_type == 'nb':
                assert np.all(Σ > 0), Σ
                Σ = np.diag(Σ)
            #print('Σ Method 2', Σ[:3])
            #pdb.set_trace()
        else:
            raise ValueError()
        return Σ


    def log_prob_complex(self, x, y):
        x, y = self._process_inputs(x, y)[:2]
        assert len(y) > 1

        weights = self.priors[y]/self.priors[y].sum()
        μ = np.average(self.means[y,:], axis = 0, weights=weights)
        Σ = self.covariance_average(self.covs[y,:], means= self.means[y,:], weights=weights)

        if self.cond_type  == 'nb':
            logp = gaussian_logdensity(x, μ, Σ, independent=True)
        else:
            logp = gaussian_logdensity(x, μ, Σ, independent=False)

        return logp

    def log_prob_complex_partial(self, x, y, subset):
        x, y,_,subset = self._process_inputs(x, y, subset=subset)
        assert len(y) > 1

        if self.cond_type  == 'nb':
            weights = self.priors[y]/self.priors[y].sum()
            μ = np.average(self.means[y,:], axis = 0, weights=weights)
            Σ = self.covariance_average(self.covs[y,:], means= self.means[y,:], weights=weights)
            logp = gaussian_logdensity(x, μ, Σ, marginal=subset, independent=True)
        elif self.cond_type == 'full':
            raise NotImplementedError()
            #logp = gaussian_logdensity(x, μ, Σ, independent=False)

        return logp


class woe_scikit_gnb(woe_wrapper):
    """
        WOE Wrapper around scikit GNB classifier
    """
    def __init__(self, clf, task=None, classes=None, input_size=None, cond_type='nb'):
        super().__init__(clf, task, classes, input_size)
        self.cond_type = cond_type # 'nb' or 'full'
        self.priors = clf.class_prior_

    def log_prob(self, x, y):
        """
        """
        n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.model.sigma_[y, :]))
        n_ij -= 0.5 * np.sum(((x - self.model.theta_[y, :]) ** 2) /
                         (self.model.sigma_[y,:]), 1)
        return n_ij

    def log_prob_partial(self, x, y, subset):
        x,_,_,subset = self._process_inputs(x,subset=subset)
        assert isinstance(y, (int, np.integer))
        if self.cond_type == 'nb':
            # Using naive Bayes, we do P(X_s | X_s1,...X_s, Y) = P (X_s | Y)
            # for GNB, this is easy: just drop non-included vars
            # en.wikipedia.org/wiki/Multivariate_normal_distribution#Marginal_distributions
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.model.sigma_[y, subset]))
            n_ij -= 0.5 * np.sum(((x[:,subset] - self.model.theta_[y, subset]) ** 2) /
                             (self.model.sigma_[y, subset]), 1)
        elif self.cond_type == 'full':
            # Decided to put this modality in agnostic_kde instead.
            #μ = self.model.sigma_[y, subset] -
            raise NotImplementedError()
        else:
            raise(ValueError)
        return n_ij

    def log_prob_complex(self, x, y):
        x, y = self._process_inputs(x, y)[:2]
        assert len(y) > 1

        μ = np.average(self.model.theta_[y,:], axis = 0, weights=self.priors[y])
        Σ = np.average(self.model.sigma_[y,:], axis = 0, weights=self.priors[y])

        if self.cond_type  == 'nb':
            logp = gaussian_logdensity(x, μ, Σ, independent=True)
        else:
            logp = gaussian_logdensity(x, μ, Σ, independent=False)
        return logp

    def log_prob_complex_partial(self, x, y, subset):
        x, y,_,subset = self._process_inputs(x, y, subset=subset)
        assert len(y) > 1

        if self.cond_type  == 'nb':
            μ = np.average(self.model.theta_[y,:], axis = 0, weights=self.priors[y])
            Σ = np.average(self.model.sigma_[y,:], axis = 0, weights=self.priors[y])
            logp = gaussian_logdensity(x, μ, Σ, marginal=subset, independent=True)
        elif self.cond_type == 'full':
            raise NotImplementedError()
        return logp


class woe_scikit_agnostic_gauss(woe_scikit_gnb):
    """
        Fit a Gaussian DE model on the full conditional P(Y|X). Gaussian form allows
        for easy marginalzation.
    """
    def __init__(self, X, clf, task=None, classes=None):
        y = clf.predict(X)
        print((y==1).sum()/len(y))
        #fitted = sklearn.naive_bayes.GaussianNB().fit(X, y)
        super().__init__(fitted, task, classes, input_size=X.shape[1])


class woe_scikit_agnostic_kde(woe_wrapper):
    def __init__(self, X, clf, task=None, classes=None, cond_type='nb', bandwidth=1):
        super().__init__(clf, task, classes, input_size=X.shape[1])
        self.cond_type = cond_type
        self.class_prior = []
        self.class_means = []
        self.class_covas = []
        self.class_llfns = {}
        self.class_partial_llfns = {}
        self.fit(X)

    def fit(self, X):
        y = self.model.predict(X)

        # Fit global (full-input) class-conditional density estimators P(X | Y)
        for c in self.classes:
            #print(X[y==c].shape)
            self.class_prior.append((y==c).sum()/len(y))
            fitted_de = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(X[y==c])
            self.class_llfns[c] = fitted_de

        if self.cond_type == 'nb':
            # In addition we need per feature class-conditional marginalized density estimators P(X_i | Y)
            for c in self.classes:
                for f in range(self.input_size):
                    fitted_de = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(X[y==c, f].reshape(-1,1))
                    self.class_partial_llfns[c].append(fitted_de)


    def log_prob(self, x, y):
        assert type(y) is int or len(y) == 1
        y = y[0]
        if x.ndim == 1: x = x.reshape(1, -1)
        #print(x.shape)
        return self.class_llfns[y].score_samples(x)

    def log_prob_partial(self, subset, x, y):
        assert type(y) is int or len(y) == 1
        y = y[0]
        assert type(subset) is int or len(subset) == 1, "Only size one subsets for now"

        #print(subset, y)
        if self.cond_type == 'nb':
            #return self.class_llfns[y].score_samples(x[subset])
            if x.ndim == 1:
                x = x.reshape(1, -1)
            x_s = x[:,subset].reshape(-1, max(len(subset), 1))
            lp = self.class_partial_llfns[y][subset[0]].score_samples(x_s)
        elif self.cond_type == 'full':
            # μ =
            # Σ =
            raise NotImplementedError()


    def prior_lodds(self, y_num, y_den):
        return np.log(self.class_prior[y_num]/self.class_prior[y_den])


### TAKEN FROM SCIPY:
def _pinv_1d(v, eps=1e-5):
    """
    A helper function for computing the pseudoinverse.
    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Elements of v smaller than eps are considered negligible.
    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.
    """
    return np.array([0 if abs(x) < eps else 1/x for x in v], dtype=float)

def _psd_pinv_decomposed_log_pdet(mat, cond=None, rcond=None,
                                  lower=True, check_finite=True):
    """
    Compute a decomposition of the pseudo-inverse and the logarithm of
    the pseudo-determinant of a symmetric positive semi-definite
    matrix.
    The pseudo-determinant of a matrix is defined as the product of
    the non-zero eigenvalues, and coincides with the usual determinant
    for a full matrix.
    Parameters
    ----------
    mat : array_like
        Input array of shape (`m`, `n`)
    cond, rcond : float or None
        Cutoff for 'small' singular values.
        Eigenvalues smaller than ``rcond*largest_eigenvalue``
        are considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of `mat`. (Default: lower)
    check_finite : boolean, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Returns
    -------
    M : array_like
        The pseudo-inverse of the input matrix is np.dot(M, M.T).
    log_pdet : float
        Logarithm of the pseudo-determinant of the matrix.
    """
    # Compute the symmetric eigendecomposition.
    # The input covariance matrix is required to be real symmetric
    # and positive semidefinite which implies that its eigenvalues
    # are all real and non-negative,
    # but clip them anyway to avoid numerical issues.

    # TODO: the code to set cond/rcond is identical to that in
    # scipy.linalg.{pinvh, pinv2} and if/when this function is subsumed
    # into scipy.linalg it should probably be shared between all of
    # these routines.

    # Note that eigh takes care of array conversion, chkfinite,
    # and assertion that the matrix is square.
    s, u = sp.linalg.eigh(mat, lower=lower, check_finite=check_finite)

    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = u.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(s))

    if np.min(s) < -eps:
        raise ValueError('the covariance matrix must be positive semidefinite')

    s_pinv = _pinv_1d(s, eps)
    U = np.multiply(u, np.sqrt(s_pinv))
    log_pdet = np.sum(np.log(s[s > eps]))

    return U, log_pdet

def gaussian_logdensity(x, μ, Σ, Σ_inv=None, logdetΣ =None, marginal=None, conditional=None, independent=False):
    """

            - marginal (ndarray or list): if provided will return marginal density of these dimensions
    """
    x_,μ_,Σ_ = x,μ,Σ

    d = μ.shape[0]

    def _extract_diag(M):
        assert M.ndim <= 2
        if M.ndim == 2:
            return np.diag(M)
        elif M.ndim ==1 :
            return M


    if Σ.ndim == 2:
        assert Σ.shape == (d,d)
        _isdiagΣ = (np.count_nonzero(Σ - np.diag(np.diagonal(Σ))) == 0)
    elif not independent:
        raise ValueError()
    else:
        # Σ was passed as diagonal
        assert Σ.shape == (d,)
        _isdiagΣ = True

    if independent and _isdiagΣ:
        Σ_ = _extract_diag(Σ_)
    elif independent and (not _isdiagΣ):
        print('Warning: independent=True but Σ is not diagonal. Will treat as such anyways.')
        Σ_ = _extract_diag(Σ_)
        Σ_inv, logdetΣ = None, None
    elif (not independent) and _isdiagΣ:
        # Maybe user forgot this is independent? Better recompute inv and logdet in this case
        independent = True
        Σ_inv, logdetΣ =None, None

    ### Check for PSD-ness of Covariance
    if independent:
        # By this point, independent case should have Σ as vector
        assert (Σ_.ndim ==1) and np.all(Σ_ >= 0), Σ_
    else:
        # Checking for psd'ness might be too expensive
        assert (Σ_.ndim ==2)

    if marginal is not None:
        # Marginalizing a multivariate gaussian -> dropping dimensions
        Σ_ = Σ_[marginal] if independent else Σ_[marginal,:][:,marginal]
        μ_ = μ_[marginal]
        x_ = x_[:,marginal]
        Σ_inv, logdetΣ = None, None # Can't use these anymore
        d = len(marginal)

    if logdetΣ is None or Σ_inv is None:
        if independent:
            #Σ_inv =
            logdetΣ = np.log(Σ_).sum()
        else:
            U, logdetΣ =  _psd_pinv_decomposed_log_pdet(Σ_, rcond = 1e-12)
            Σ_inv = np.dot(U,U.T)


    diff = x_ - μ_   # (x - μ_j) , size (n,d)

    if independent:
        expterm = np.sum((diff ** 2) / Σ_, 1)
    else:
        expterm = np.diag(np.dot(np.dot(diff, Σ_inv),diff.T))
        # TODO: Faster and cleaner way to do this by storing sqrt of precision matrix. See
        # https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/_multivariate.py line 330

    constant = d * np.log(2. * np.pi) + logdetΣ

    logp = -0.5 * (constant + expterm)

    return logp
