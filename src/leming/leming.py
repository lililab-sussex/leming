# Author: Peter Wijeratne (p.wijeratne@sussex.ac.uk)
# LEMING class
import numpy as np
import scipy as sp
from scipy.special import gammaln
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture as gmm
import torch
from torch import logsumexp
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from joblib import Parallel, delayed, cpu_count
from torch.utils.checkpoint import checkpoint
import os

def _fit_gmm_em_feature(X_i, y):
    y_i = y[~np.isnan(X_i)]
    X_i = X_i[~np.isnan(X_i)]
    mm = gmm(n_components=2, covariance_type='diag', tol=1E-3, n_init=100,
             means_init=np.array([np.nanmean(X_i[y_i == 0]), np.nanmean(X_i[y_i == 1])]).reshape(2, 1),
             precisions_init=np.array([1 / np.nanstd(X_i[y_i == 0]) ** 2, 1 / np.nanstd(X_i[y_i == 1]) ** 2]).reshape(2, 1),
             weights_init=np.array([0.5, 0.5]))
    mm.fit(X_i[(y_i == 0).astype(bool) + (y_i == 1).astype(bool)].reshape(-1, 1))
    return [mm.means_[0][0], np.sqrt(mm.covariances_[0][0]),
            mm.means_[1][0], np.sqrt(mm.covariances_[1][0]),
            mm.weights_[0]]

def _gmm_like(theta, X, y):
    pdf_0 = sp.stats.norm.pdf(X[y == 0], loc=theta[0], scale=theta[1]) * theta[4]
    pdf_1 = sp.stats.norm.pdf(X[y == 1], loc=theta[2], scale=theta[3]) * (1 - theta[4])
    pdf_0[np.isnan(pdf_0)] = .5
    pdf_1[np.isnan(pdf_1)] = .5
    like = np.concatenate((pdf_0, pdf_1))
    like[like == 0] = np.finfo(float).eps
    if np.sum(np.isnan(like)) > 0 or np.sum(np.isinf(like)) > 0:
        raise ValueError("NaN or inf encountered in GMM likelihood")
    return -1 * np.sum(np.log(like))
 
def _fit_gmm_optim_feature(X_i, y):
    y_i = y[~np.isnan(X_i)]
    X_i = X_i[~np.isnan(X_i)]
    theta_0 = [np.nanmean(X_i[y_i == 0]), np.nanstd(X_i[y_i == 0]),
               np.nanmean(X_i[y_i == 1]), np.nanstd(X_i[y_i == 1]), 0.5]
    fit = sp.optimize.minimize(_gmm_like, theta_0, args=(X_i, y_i), method='SLSQP')
    return fit.x

class LEMING(BaseEstimator):

    def __init__(self,
                 X=None,
                 labels=None,
                 S_prior=None,
                 n_sinkhorn=20,
                 temperature_start=1E1,
                 temperature_end=1E0,
                 temperature_prior=1E0,
                 gumbel_scale=0,
                 n_mc_samples=20,
                 n_iters=100,
                 step_size=1E-1,
                 use_em=True,
                 verbose=False):
        
        # user-defined variables
        self.X = X
        self.labels = labels
        self.S_prior = S_prior
        self.n_sinkhorn = n_sinkhorn
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_prior = temperature_prior
        self.gumbel_scale = gumbel_scale
        self.n_mc_samples = n_mc_samples
        self.n_iters = n_iters
        self.step_size = step_size
        self.use_em = use_em
        self.verbose = verbose
        
        # automatically-defined variables
        self.dtype = torch.float32
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        torch.set_default_dtype(self.dtype)
        self.eps = torch.finfo(self.dtype).eps
        # assume a uniform prior on P
        self.params = [self.to_var(torch.zeros((self.X.shape[1], self.X.shape[1]), requires_grad=True, device=self.device))]
        self.loss_trace = []
        
    def to_var(self, x):
        if self.is_cuda:
            x = x.cuda()
        return x
    
    def fit_gmms(self, X, y, n_jobs=-1):
        if self.use_em:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_gmm_em_feature)(X[:, i], y) for i in range(X.shape[1])
            )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_gmm_optim_feature)(X[:, i], y) for i in range(X.shape[1])
            )
        self.thetas = results
    
    def calc_prob_mat(self, X):
        prob_mat = np.zeros((X.shape[0], X.shape[1], 2))
        for i in range(X.shape[1]):
            pdf_0 = sp.stats.norm.pdf(X[:,i], loc=self.thetas[i][0], scale=self.thetas[i][1])
            pdf_1 = sp.stats.norm.pdf(X[:,i], loc=self.thetas[i][2], scale=self.thetas[i][3])
            pdf_0[np.isnan(pdf_0)] = 0.5
            pdf_1[np.isnan(pdf_1)] = 0.5
            prob_mat[:, i, 0] = pdf_0.flatten()
            prob_mat[:, i, 1] = pdf_1.flatten()
        self.prob_mat = self.to_var(torch.tensor(prob_mat, dtype=self.dtype))

    def vectorised_log_likelihood_ebm_logspace(self, P):
        # note we omit the uniform prior over k
        k = self.prob_mat.shape[1]+1
        logp_perm_k = torch.zeros((self.prob_mat.shape[0], k, P.shape[2]), device=self.device)
        p_yes = torch.einsum('ij,jkl->ikl', self.prob_mat[:, :, 1], P)
        p_yes[p_yes == 0] = self.eps
        p_no = torch.einsum('ij,jkl->ikl', self.prob_mat[:, :, 0], torch.flip(P, [1]))
        p_no[p_no == 0] = self.eps
        logp_yes = torch.log(p_yes)
        logp_no = torch.log(p_no)
        logcp_yes = torch.cumsum(logp_yes, axis=1)
        logcp_no = torch.cumsum(logp_no, axis=1)
        logp_perm_k[:, 0, :] = logcp_no[:, -1, :]
        logp_perm_k[:, 1:-1, :] = torch.flip(logcp_no[:, :-1, :], [1]) + logcp_yes[:, :-1, :]
        logp_perm_k[:, -1, :] = logcp_yes[:, -1, :]
        logp_perm = logsumexp(logp_perm_k, axis=1)
        return torch.sum(logp_perm)

    def sinkhorn_logspace(self, logP, n_iters=10):
        n = logP.size()[1]
        logP = logP.view(-1, n, n)
        for i in range(n_iters):
            logP = logP - (logsumexp(logP, dim=2, keepdim=True)).view(-1, n, 1)
            logP = logP - (logsumexp(logP, dim=1, keepdim=True)).view(-1, 1, n)
        return logP

    def vectorised_sinkhorn_logspace(self, logP, n_iters=20):
        n = logP.size()[1]
        logP = logP.view(n, n, -1)
        for i in range(n_iters):
            logP = logP - logsumexp(logP, dim=1, keepdim=True).view(n, 1, -1)
            logP = logP - logsumexp(logP, dim=0, keepdim=True).view(1, n, -1)
        return logP

    def sample_gumbel(self, P, n=1):
        return -torch.log(-torch.log(torch.rand((n, P[0], P[1])) + self.eps) + self.eps)

    def vectorised_sample_gumbel(self, P, n=1):
        return -torch.log(-torch.log(torch.rand((P[0], P[1], n)) + self.eps) + self.eps)

    def gumbel_distance(self, log_mu_P, temperature=1E0):
        # from https://arxiv.org/abs/1802.08665 Supplementary Section B.3
        # note the seemingly magic number comes from the Gumbel distribution expectation (which is equal to the Euler-Mascheroni constant: https://en.wikipedia.org/wiki/Euler%27s_constant)
        arr = torch.sum(np.log(self.temperature_prior) - 0.5772156649 * self.temperature_prior / temperature -
                        log_mu_P * self.temperature_prior / temperature -
                        torch.exp(gammaln(1 + self.temperature_prior / temperature) - log_mu_P * self.temperature_prior / temperature)
                        - (np.log(temperature) - 1 - 0.5772156649))
        return arr

    def variational_objective(self, temperature):
        log_mu_P = self.params[0]
        # sample Gumbel noise
        gumbel_noise = self.to_var(self.vectorised_sample_gumbel(log_mu_P.shape, self.n_mc_samples))
        # move \mu closer to Birkhoff polytope
        P = torch.exp(self.vectorised_sinkhorn_logspace((log_mu_P.unsqueeze(2) + gumbel_noise * self.gumbel_scale) / temperature, self.n_sinkhorn))
        # observation likelihood + KL
        elbo = self.to_var(self.vectorised_log_likelihood_ebm_logspace(P) / self.n_mc_samples) \
            + self.to_var(self.gumbel_distance(log_mu_P, temperature))
        return -elbo
    
    def train(self):
        try:
            n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])
        except:
            n_jobs = -1
        self.fit_gmms(self.X, self.labels, n_jobs=n_jobs)
        self.calc_prob_mat(self.X)
        optimizer = torch.optim.Adam(self.params, lr=self.step_size, eps=self.eps)
        gamma = (self.temperature_end / self.temperature_start) ** (1 / self.n_iters)        
        for step in range(self.n_iters):
            optimizer.zero_grad()
            temp = self.temperature_start * (gamma ** step)
            loss = self.variational_objective(max(self.temperature_end, temp))
            self.loss_trace.append(loss)
            if self.verbose:
                print (step, loss, temp)
            loss.backward()
            optimizer.step()

    def predict_stage(self, X, hard_perm=True):
        self.calc_prob_mat(X)
        log_mu_P = self.params[0]
        # move \mu closer to Birkhoff polytope
        log_P = self.sinkhorn_logspace(log_mu_P / self.temperature_end, self.n_sinkhorn)
        # note zero variance
        P = torch.exp(log_P)[0]
        k = self.prob_mat.shape[1]+1
        if hard_perm:
            # round to permutation matrices
            prob_mat_np = self.prob_mat.detach().cpu().numpy()
            P = P.detach().cpu().numpy()
            P_hard = self.round_to_perm(P)
            S_hard = np.einsum('i,ij->j', np.arange(X.shape[1]), P_hard).astype(int)
            p_yes = np.array(prob_mat_np[:, S_hard, 1])
            p_yes[p_yes == 0] = self.eps
            p_no = np.array(prob_mat_np[:, S_hard, 0])
            p_no[p_no == 0] = self.eps
            logp_yes = np.log(p_yes)
            logp_no = np.log(p_no)
            logcp_yes = np.cumsum(logp_yes, axis=1)
            logcp_no = np.cumsum(logp_no, axis=1)
            logp_perm_k = np.zeros((prob_mat_np.shape[0], k))
            logp_perm_k[:, 0] = logcp_no[:, -1]
            logp_perm_k[:, 1:-1] = np.flip(logcp_no[:, :-1], [1]) + logcp_yes[:, :-1]
            logp_perm_k[:, -1] = logcp_yes[:, -1]
        else:
            # just use doubly-stochastic matrix directly
            logp_perm_k = torch.zeros((self.prob_mat.shape[0], k), device=self.device)
            p_yes = torch.einsum('ij,jk->ik', self.prob_mat[:, :, 1], P)
            p_yes[p_yes == 0] = self.eps
            p_no = torch.einsum('ij,jk->ik', self.prob_mat[:, :, 0], torch.flip(P, [1]))
            p_no[p_no == 0] = self.eps
            logp_yes = torch.log(p_yes)
            logp_no = torch.log(p_no)
            logcp_yes = torch.cumsum(logp_yes, axis=1)
            logcp_no = torch.cumsum(logp_no, axis=1)
            logp_perm_k[:, 0] = logcp_no[:, -1]
            logp_perm_k[:, 1:-1] = torch.flip(logcp_no[:, :-1], [1]) + logcp_yes[:, :-1]
            logp_perm_k[:, -1] = logcp_yes[:, -1]
            logp_perm_k = logp_perm_k.detach().cpu().numpy()
        stage_probs = sp.special.softmax(logp_perm_k, axis=1)
        stages = np.argmax(stage_probs, axis=1)
        return stages, stage_probs

    def round_to_perm(self, P):
        N = P.shape[0]
        assert P.shape == (N, N)
        row, col = sp.optimize.linear_sum_assignment(-P)
        P = np.zeros((N, N))
        P[np.arange(N), col] = 1.0
        return P

    def vectorised_round_to_perm(self, P):
        N = P.shape[0]
        P_hard = np.empty(P.shape)
        for i in range(P.shape[2]):
            row, col = sp.optimize.linear_sum_assignment(-P[:,:,i])
            P_i = np.zeros((N, N))
            P_i[np.arange(N), col] = 1.0
            P_hard[:,:,i] = P_i
        return P_hard

    def get_sequence(self, n_samples=0):
        n_feat = self.X.shape[1]
        log_mu_P = self.params[0]
        if n_samples>0:
            # distribution of sequences (non-zero Gumbel noise)
            S_samples = []
            # note that we don't vectorise here for memory reasons; but we could
            for i in range(n_samples):
                # sample Gumbel noise
                gumbel_noise = self.to_var(self.sample_gumbel(log_mu_P.shape)[0])
                # move \mu closer to Birkhoff polytope
                log_P = self.sinkhorn_logspace((log_mu_P + gumbel_noise * self.gumbel_scale) / self.temperature_end, self.n_sinkhorn)
                # note zero variance
                P_sample = torch.exp(log_P)
                P_sample = np.array([x.detach().cpu().numpy() for x in P_sample])
                # round to permutation matrices
                P_hard_sample = self.round_to_perm(P_sample[0])
                # sequences
                S_samples.append(np.einsum('i,ij->j', np.arange(n_feat), P_hard_sample))        
            S_unique, counts = np.unique(S_samples, axis=0, return_counts=True)
            #FIXME: change to consensus ordering
            S_point = S_unique[np.argmax(counts)].astype(int) # mode
            return S_point, S_samples
        else:
            # point estimate of sequence (zero Gumbel noise)
            # move \mu closer to Birkhoff polytope
            log_P = self.sinkhorn_logspace(log_mu_P / self.temperature_end, self.n_sinkhorn)
            # note zero variance
            P_sample = torch.exp(log_P)
            P_sample = np.array([x.detach().cpu().numpy() for x in P_sample])
            # round to permutation matrices
            P_hard_sample = self.round_to_perm(P_sample[0])
            S_point = np.einsum('i,ij->j', np.arange(n_feat), P_hard_sample)
            return S_point, np.array(S_point)

    def plot_stages(self, stages):
        fig, ax = plt.subplots()
        ax.hist(stages)
        ax.set_xlabel('Stage', fontsize=16)
        ax.set_ylabel('Count', fontsize=16)

    def confusion_soft(self, n_samples=100, gumbel_scale=0.01):
        log_mu_P = self.params[0]
        n_feat = log_mu_P.shape[0]
        confusion = torch.zeros((n_feat, n_feat))
        for i in range(n_samples):
            gumbel_noise = self.to_var(self.sample_gumbel(log_mu_P.shape)[0])
            log_P = (log_mu_P + gumbel_noise * gumbel_scale) / self.temperature_end
            log_P = self.sinkhorn_logspace(log_P, self.n_sinkhorn)
            confusion += torch.exp(log_P)[0]
        confusion /= n_samples
        return confusion.detach().numpy()

    def confusion_hard(self, S, S_samples):
        log_mu_P = self.params[0]
        n_feat = log_mu_P.shape[0]
        confusion = np.zeros((n_feat, n_feat))
        for i in range(n_feat):
            confusion[i, :] = np.sum(S_samples == S[i], axis=0)
        return confusion

    def plot_confusion(self, confusion, S):
        n_feat = len(S)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(confusion, interpolation='nearest', cmap='gray_r')
        ax.set_xticks(np.arange(n_feat))
        ax.set_yticks(np.arange(n_feat))
        ax.set_xticklabels(np.arange(n_feat), fontsize=20)
        ax.set_yticklabels(np.arange(n_feat)[S], fontsize=20)
        if n_feat >= 50 and n_feat < 500:
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_major_ticks()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_major_ticks()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 10 != 0]
        elif n_feat >= 500:
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_major_ticks()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_major_ticks()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 100 != 0]
        ax.set_ylabel('Feature', fontsize=20, labelpad=10)
        ax.set_xlabel('Event', fontsize=20)
        plt.subplots_adjust(bottom=0.15, top=0.95)

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.n_iters), [x.detach().numpy() for x in self.loss_trace])
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel('ELBO', fontsize=16)

    def write(self, path='model.pkl'):
        file_out = Path(path)
        pickle_file = open(file_out, 'wb')
        data = {}
        data['params'] = self.params
        data['thetas'] = self.thetas
        data['loss'] = self.loss_trace
        pickle.dump(data, pickle_file)
        pickle_file.close()
    
    def read(self, path='model.pkl'):
        file_in = Path(path)
        pickle_file = open(file_in, 'rb')
        data = pickle.load(pickle_file)
        self.params = data['params']
        self.thetas = data['thetas']
        self.loss_trace = data['loss']
        pickle_file.close()
        self.calc_prob_mat(self.X)
