functions {
  // Polynomial basis function
  matrix poly_basis(vector x, int P) {
    int N = num_elements(x);
    matrix[N, P] U;
    for (n in 1:N) {
      for (p in 1:P) {
        U[n, p] = pow(x[n], p);
      }
    }
    return U;
  }
}
data {
  int<lower=1> N;
  int<lower=1> P;
  vector[N] x;
  array[N] real y;
}
transformed data {
  // Apply polynomial basis function to x
  vector[N] u = log(x);
  vector[N] u_std = (u - mean(u)) / sd(u);
  matrix[N, P] U = poly_basis(u_std, P);
}
parameters {
  real beta0;
  vector[P] beta;
  real<lower=0> inv_sigma;
}
transformed parameters {
  vector[N] mu;
  mu = beta0 + U*beta;
}
model {
  // Priors
  beta0 ~ normal(0, 10);
  beta ~ normal(0, 1);
  inv_sigma ~ exponential(5);

  // Likelihood
  y ~ normal(mu, 1/inv_sigma);
}
generated quantities {
  vector[N] log_lik;
  array[N] real y_rep;

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], 1/inv_sigma);
    y_rep[i] = normal_rng(mu[i], 1/inv_sigma);
  } 
}