import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import pdb
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
import sklearn.naive_bayes


DEBUG = False

from .woe_utils import invlogit#, psd_pinv_decomposed_log_pdet

#### Should try to make woe_wrapper parent class agnostic to torch/numpy/scikit.
class woe_wrapper():
    """
        Child methods should specify the following methods:
            - log_prob (computes loglikelhood of X given Y)
            - log_prob_partial (computes loglikelhood of X_S given Y)
            - prior_lodds
    """
    def __init__(self, model, task, classes, input_size):
        self.model = model
        self.classes = classes
        self.task  = task
        self.input_size = input_size
        self.data_transform = None
        self.cache = None
        self.caching = False

    def _start_caching(self):
        self.caching = True
        self.cache = {}

    def _stop_caching(self):
        self.caching = False
        self.cache = None

    def woe(self, x, y1, y2, **kwargs):
        ll_num   = self.generalized_conditional_ll(x, y1, **kwargs)
        ll_denom = self.generalized_conditional_ll(x, y2, **kwargs)
        woe = ll_num - ll_denom
        return woe

    def log_posterior(self, x, y):
        # Mostly for debugging
        # log p( y | x) = log (x | y) + log p(y) - log Σ_y exp(log p(x|y))
        logpart = logsumexp(np.array([self.log_prob(x, c)+self.log_prior(y) for c in self.classes]))
        logpost = self.log_prob(x, y) + self.log_prior(y) - logpart
        return logpost

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

        #x = x.view(x.shape[0], -1) #.to(args.device) #TORCH
        #x =

        if type(y) == set:
            y = list(y)
        if type(y) is int or (type(y) == list and len(y) == 1):
            if verbose: print('H is simple hypotesis')
            y = y if type(y) is list else [y]
            #y = torch.tensor(y*x.shape[0]) #TORCH
            #y = int2hot(y, k) # TORCH
            if subset is None:
                loglike = self.log_prob(x, y)
            else:
                loglike = self.log_prob_partial(subset, x, y)
            loglike = loglike.squeeze() # to get scalar, to be consistent by logsumexp below
        elif type(y) is list:
            if verbose: print('H is composite hypotesis')
            priors    = self.prior(np.array(range(k)))
            logpriors = np.log(priors[y])
            logprior_set = np.log(priors.sum())
            #loglike = [[] for _ in range(len(y))]
            loglike = np.zeros((len(y),1))

            # Need to compute log prob with respect to each class in y, then do:
            # log p(x|Y∈C) = 1/P(Y∈C) * log ∑_y p(x|Y=y)p(Y=y)
            #              = log ∑_y exp{ log p(x|Y=y) + log p(Y=y) - log P(Y∈C)}
            for i, yi in enumerate(y):
                if self.caching and ((hash_x,hash_S)in self.cache) and (yi in self.cache[(hash_x,hash_S)]):
                    if DEBUG and hash_S != -1: print('using cached: ({},{})'.format(hash_x, hash_S))
                    loglike[i] = self.cache[(hash_x,hash_S)][yi]
                else:
                    # yi = int2hot(torch.tensor([yi]*x.shape[0]), k) -> for pytorch
                    if subset is None:
                        loglike[i] = self.log_prob(x, yi) + logpriors[i] - logprior_set
                    else:
                        loglike[i] = self.log_prob_partial(subset, x, yi) + logpriors[i] - logprior_set
                    if self.caching:
                        if not (hash_x,hash_S) in self.cache:
                            self.cache[(hash_x,hash_S)] = {}
                        self.cache[(hash_x,hash_S)][yi] = loglike[i]

            #loglike = torch.stack(loglike, dim=1)
            #loglike = (loglike, dim=1)

            #loglike = torch.logsumexp(loglike, dim = 1) # pytorch
            loglike = logsumexp(loglike)
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

        # if order == 'fwd':
        #     order = get_block_order(7, int(np.sqrt(input_size)), reverse= False)
        # elif order == 'bwd':
        #     order = get_block_order(7, int(np.sqrt(input_size)), reverse= True)
        # elif order == 'rnd':
        #     order, blockord = randperm_block_order(get_block_order(7, int(np.sqrt(input_size))), 7)
        # else:
        #     raise ValueError("Unrecognized order")


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

    def _process_inputs(self, x = None, y = None, subset = None):
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
        self.class_prior = []
        for c in self.classes:
            self.class_prior.append((Y==c).sum()/len(Y))
            μ = X[Y==c].mean(axis=0)
            means.append(μ)
            Σ = np.cov(X[Y==c].T)
            delta = max(eps - np.linalg.eigvalsh(Σ).min(), 0)
            Σ += np.eye(self.d)*1.1*delta # add to ensure PSD'ness
            Σ += shift_cov*np.eye(self.d) # TODO: FInd more principled smoothing
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
                    pdb.set_trace(header='Failed at crearing normal distrib')
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
        self.class_prior = np.array(self.class_prior)

        self.dl2π = self.d * np.log(2.*np.pi)



    def log_prob(self, x, y):
        """
        """
        x,y = self._process_inputs(x,y)

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

        #assert (logp <= 0).all(), logp
        return logp

    def log_prob_partial(self, subset, x, y):
        x,y,subset = self._process_inputs(x,y,subset)

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
            #Σ_inv    = np.linalg.inv(Σ_ss)
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

    def prior_lodds(self, y_num, y_den=None):
        if y_den is None:
            # Defult to complement
            y_den = np.array(list(set(range(len(self.classes))) - {y_num}))
        odds_num = self.class_prior[y_num]
        if type(y_num) in [np.ndarray, list]:
            odds_num = odds_num.sum()
        odds_den = self.class_prior[y_den]
        if type(y_den) in [np.ndarray, list]:
            odds_den = odds_den.sum()
        return np.log(odds_num/odds_den)

    def prior(self, ys):
        return self.class_prior[ys]


    def log_prior(self, ys):
        return np.log(self.class_prior[ys])


class woe_scikit_gnb(woe_wrapper):
    """
        WOE Wrapper around scikit GNB classifier
    """
    def __init__(self, clf, task=None, classes=None, input_size=None, cond_type='nb'):
        super().__init__(clf, task, classes, input_size)
        self.cond_type = cond_type # 'nb' or 'full'

    def log_prob(self, x, y):
        """
        """
        n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.model.sigma_[y, :]))
        n_ij -= 0.5 * np.sum(((x - self.model.theta_[y, :]) ** 2) /
                         (self.model.sigma_[y,:]), 1)
        return n_ij

    def log_prob_partial(self, subset, x, y):
        if self.cond_type == 'nb':
            # Using naive Bayes, we do P(X_s | X_s1,...X_s, Y) = P (X_s, Y)
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

    def prior_lodds(self, y_num, y_den):
        return np.log(self.model.class_prior_[y_num]/self.model.class_prior_[y_den])

    def priors(self,ys):
        raise NotImplemented()
        #return np.log(self.model.class_prior_[y_num]/self.model.class_prior_[y_den])



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
