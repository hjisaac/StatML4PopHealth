functions {
  // Spectral density functions
	vector spd_se(vector omega, real sigma, real ell) {
		// Your code here
	}

	vector spd_matern32(vector omega, real sigma, real ell) {
		// Your code here
	}

	vector spd_matern52(vector omega, real sigma, real ell) {
		// Your code here
	}

  // Eigenvalues and eigenvectors
  vector eigenvalues(int M, real L) {
		// Your code here
	}

	matrix eigenvectors(vector x, int M, real L, vector lambda) {
		// Your code here
	}

  // HSGP functions
  vector hsgp_se(vector x, real sigma, real ell, vector lambdas, matrix PHI, vector z) {
		int N = rows(x);
		int M = cols(PHI);
		vector[N] f;
		matrix[M, M] Delta;

		// Spectral densities evaluated at the square root of the eigenvalues

		// Construct the diagonal matrix Delta

		// Compute the HSGP sample

		return f;
	}

  vector hsgp_matern32(vector x, real sigma, real ell, vector lambdas, matrix PHI, vector z) {
		int N = rows(x);
		int M = cols(PHI);
		vector[N] f;
		matrix[M, M] Delta;

		// Spectral densities evaluated at the square root of the eigenvalues

		// Construct the diagonal matrix Delta

		// Compute the HSGP sample

		return f;
	}

  vector hsgp_matern52(vector x, real sigma, real ell, vector lambdas, matrix PHI, vector z) {
		int N = rows(x);
		int M = cols(PHI);
		vector[N] f;
		matrix[M, M] Delta;

		// Spectral densities evaluated at the square root of the eigenvalues

		// Construct the diagonal matrix Delta

		// Compute the HSGP sample

		return f;
	}
}

data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
  real<lower=0> C;
  int<lower=1> M;
}

transformed data {
  // Boundary condition
  real<lower=0> L = C * max(abs(x));
  
  // Compute the eigenvalues
  vector[M] lambdas = eigenvalues(M, L);

  // Compute the eigenvectors
  matrix[N, M] PHI = eigenvectors(x, M, L, lambdas);
}

parameters {
  real alpha;
  real<lower=0> sigma_eps;

  real<lower=0> sigma;
  real<lower=0> ell;
  vector[M] z;
}

transformed parameters {
  vector[N] f = hsgp_se(x, sigma, ell, lambdas, PHI, z);
  vector[N] mu = alpha + f;
}

model {
  // Priors
  alpha ~ normal(0, 1);
  sigma_eps ~ inv_gamma(5, 5);

  // GP related priors
  sigma ~ inv_gamma(5, 5);
  ell ~ inv_gamma(5, 5);
  z ~ normal(0, 1);

  // Likelihood
  y ~ normal(mu, sigma_eps);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma_eps);
    y_rep[n] = normal_rng(mu[n], sigma_eps);
  }
}