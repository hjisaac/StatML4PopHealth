functions {
  vector sir(real t,
             vector y, 
             real beta, 
             real sigma,
             real N) {

      vector[3] dydt;

      real S = y[1];
      real I = y[2];
      real R = y[3];
      
      dydt[1] = -beta * I * S / N;
      dydt[2] =  beta * I * S / N - sigma * I;
      dydt[3] =  sigma * I;
      
      return dydt;
  }
}
data {
  int<lower=1> n_days;
  vector[3] y0;
  real t0;
  array[n_days] real t;
  int N;
  array[n_days] int<lower=0> cases;
}
parameters {
  real<lower=0> sigma;
  real<lower=0> beta;
  real<lower=0> phi_inv;
}
transformed parameters{
  array[n_days] vector[3] y;
  array[n_days] real infected;

  real<lower=0> phi = 1. / phi_inv;
  
  y = ode_rk45(sir, y0, t0, t, beta, sigma, N);

  for (i in 1:n_days)
    infected[i] = y[i, 2] + 1e-5;
}
model {
    //priors
    beta ~ normal(4, 2); //truncated at 0
    sigma ~ normal(0.8, 0.3); //truncated at 0
    phi_inv ~ exponential(5);
}
generated quantities {
  real R0 = beta / sigma;
  real recovery_time = 1 / sigma;

  array[n_days] real pred_cases;
  pred_cases = neg_binomial_2_rng(infected, phi);
}
