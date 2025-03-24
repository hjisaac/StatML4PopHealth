functions {
	vector gp_se(array[] real x, real sigma, real ell, vector z) {
    int n = size(x);
		matrix[n,n] K;
		vector[n] f;
		matrix[n,n] L;

		// 1. Compute the covariance matrix
		K = gp_exp_quad_cov(x, sigma, ell) + diag_matrix(rep_vector(1e-6, n));

		// 2. Perform the Cholesky decomposition
		L = cholesky_decompose(K);

		// 3. Compute the GP sample
		f = L * z;

		return f;
	}
}

data {
	int<lower=1> N;
	array[N] real x;
	vector[N] y;
}

parameters {
	// GP hyperparameters
	real<lower=0> sigma; // marginal variance
	real<lower=0> ell;   // lengthscale

	// Auxiliary variables
	vector[N] z;

	// Baseline term
	real alpha;

	// Noise variance
	real<lower=0> sigma_eps;
}

transformed parameters {
	vector[N] f = gp_se(x, sigma, ell, z);
	vector[N] mu = alpha + f;
}

model {
	// Priors
	alpha ~ normal(0, 1);
	sigma_eps ~ inv_gamma(1,1);

	// GP related prior
	sigma ~ inv_gamma(1, 1);
	ell ~ inv_gamma(1, 1);
	z ~ normal(0, 1);

	// Likelihood
	y ~ normal(mu, sigma_eps);
}

generated quantities {
	vector[N] log_lik;
	vector[N] y_rep;

	for (i in 1:N) {
		log_lik[i] = normal_lpdf(y[i] | f[i], sigma_eps);
		y_rep[i] = normal_rng(mu[i], sigma_eps);
	}
}