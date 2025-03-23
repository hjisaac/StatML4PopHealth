data {
  int<lower=1> N;
  vector[N] x;
  array[N] real y;
}
transformed data {
  // Apply log transformation to x
  vector[N] log_x = log(x);
}
parameters {
  real beta0;
  real beta;
  real<lower=0> inv_sigma;
}
transformed parameters {
  vector[N] mu;
  mu = beta0 + beta * log_x;
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