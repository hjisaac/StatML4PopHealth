data {
  int<lower=1> N;                     // # observations
  int<lower=1> J;                     // # groups (schools)
  int<lower=1> K;                     // # predictors
  array[N] int<lower=0,upper=1> y;         // response (anaemia: 0/1) // 
  array[N] int<lower=1, upper=J> group;     // school ID for each observation
  matrix[N, K] X;                     // design matrix of covariates
}

parameters {
  real alpha_0;                      // global intercept
  vector[J] alpha;                  // school-specific random effects
  real<lower=0> sigma;              // sd of school effects
  vector[K] beta;                   // fixed effects (predictors)
}

transformed parameters {
  vector[N] pi;                    // linear predictor: logit(pi)
  for (i in 1:N) {
    pi[i] = alpha_0 + alpha[group[i]] + dot_product(X[i], beta);
  }
  //pi = alpha_0 + alpha[group] + X * beta; // alternative: can also be vectorised

}

model {
  // priors
  alpha_0 ~ normal(0, 2);
  sigma ~ cauchy(0, 1);             
  alpha ~ normal(alpha_0, sigma);
  beta ~ normal(0, 1);

  // likelihood using log PMF
  y ~ bernoulli_logit(pi);
}

generated quantities {
 array[N] int<lower=0,upper=1> ypred; // predictions
 
 ypred = bernoulli_logit_rng(pi);

}
