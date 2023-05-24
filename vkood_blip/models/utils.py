import torch
import random
import numpy as np

import torch.nn as nn
import torch.distributions as dist
from torch import nn
from torch.distributions import MultivariateNormal


def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def mahalanobis(x, mu, phi=1):
    return -0.5 * (1 / phi) * torch.dot((x - mu), (x - mu))

def rescaled_GEM_score(x, mean, phi=1):
    energy = 0
    for mu in mean:
        energy += torch.exp(mahalanobis(x, mu, phi))
    return energy



class EMGaussianMixtureLayer(nn.Module):
    def __init__(self, n_components, n_features, n_iter=10, tol=1e-3):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.n_iter = n_iter
        self.tol = tol

        self.means = nn.Parameter(torch.randn(n_components, n_features))
        self.covariances = nn.Parameter(torch.eye(n_features).expand(n_components, n_features, n_features))
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)

    def forward(self, data):
        for _ in range(self.n_iter):
            # E step: estimate the probabilities
            probs = torch.zeros(data.shape[0], self.n_components, device=data.device)
            for i in range(self.n_components):
                diff = data - self.means[i]
                mult = torch.einsum('ij,jk,ik->i', diff, torch.inverse(self.covariances[i]), diff)
                probs[:, i] = self.weights[i] * torch.exp(-0.5 * mult) / torch.sqrt(torch.det(self.covariances[i]))

            probs = probs / probs.sum(dim=1, keepdim=True)

            # M step: update the parameters
            self.weights.data = probs.mean(dim=0)
            for i in range(self.n_components):
                diff = data - self.means[i]
                self.means.data[i] = (probs[:, i, None] * data).sum(dim=0) / probs[:, i].sum()

                weighted_sum = torch.einsum('ij,ik->ijk', diff, diff) * probs[:, i, None, None]
                self.covariances.data[i] = weighted_sum.sum(dim=0) / probs[:, i].sum()

        return self.means, self.covariances, self.weights


class GaussianMixtureLayer(nn.Module):
    def __init__(self, n_components, n_features, eps=1e-3, max_depth=100):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.eps = eps
        self.max_depth = max_depth

        self.means = nn.Parameter(torch.randn(n_components, n_features))
        self.covariances = nn.Parameter(torch.eye(n_features).expand(n_components, n_features, n_features))
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)

    def forward(self, x):
        with torch.no_grad():
            self.depth = 0
            prev_means = self.means.clone()
            while self.depth < self.max_depth:
                self.update_parameters(x)
                diff = torch.norm(self.means - prev_means)
                if diff <= self.eps:
                    break
                prev_means = self.means.clone()
                self.depth += 1

        attach_gradients = self.training
        if attach_gradients:
            self.update_parameters(x)
            return self.compute_loss(x), self.means, self.covariances
        else:
            return self.compute_loss(x).detach(), self.means.detach(), self.covariances.detach()

    def compute_loss(self, x):
        log_likelihood = 0
        for i in range(self.n_components):
            normal_dist = MultivariateNormal(self.means[i], self.covariances[i])
            log_likelihood += self.weights[i] * normal_dist.log_prob(x)
        return -log_likelihood.sum()

    def compute_weights(self, x):
        weights = torch.zeros(x.shape[0], self.n_components, device=x.device)
        for i in range(self.n_components):
            diff = x - self.means[i]
            mahalanobis_dist = torch.einsum('ij,ij->i', diff, torch.matmul(diff, torch.inverse(self.covariances[i])))
            weights[:, i] = mahalanobis_dist
        return weights

    def update_parameters(self, x):
        weights = self.compute_weights(x)
        sum_weights = weights.sum(dim=0)
        jitter = 1e-6 * torch.eye(self.n_features, device=x.device)
        for i in range(self.n_components):
            diff = x - self.means[i]
            self.means[i] = (weights[:, i, None] * x).sum(dim=0) / sum_weights[i]
            outer = torch.bmm(diff.unsqueeze(-1), diff.unsqueeze(-2))
            self.covariances[i] = (weights[:, i, None, None] * outer).sum(dim=0) / sum_weights[i] + jitter



