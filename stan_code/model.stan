data {
  int<lower=1> N_measurement;
  int<lower=1> N_decomp;
  int<lower=1> N_undecomp;
  int<lower=1> N_group;
  int<lower=1> N_measurement_type;
  int<lower=1> N_reaction;
  vector[N_measurement] y;
  int<lower=1,upper=N_reaction> rxn_ix[N_measurement];
  int<lower=1,upper=N_decomp+N_undecomp> compound_ix_decomp[N_decomp];
  int<lower=1,upper=N_decomp+N_undecomp> compound_ix_undecomp[N_undecomp];
  int<lower=1,upper=N_measurement_type> measurement_type[N_measurement];
  int<lower=0,upper=1> likelihood;
  matrix[N_decomp + N_undecomp, N_reaction] S;
  matrix[N_decomp, N_group] G;
}
transformed data {
  int N_compound = N_decomp + N_undecomp;
  real mean_y = mean(y);
  real sd_y = sd(y);
  vector[N_measurement] y_std = (y - mean_y) / sd_y;
}
parameters {
  // measurement error (standardised scale)
  vector<lower=0>[N_measurement_type] sigma_std;

  // group formation energy variation (standardised scale)
  real<lower=0> tau_g_std;                           

  // undecomposable compound formation energy variation (standardised scale)
  real<lower=0> tau_undecomp_std;                    

  // group formation energy mean (standardised scale)
  real mu_g_std;

  // undecomposable compound formation energy mean (standardised scale)
  real mu_undecomp_std;

  // decomposable compound prediction error scale (standardised scale)
  real<lower=0> sigma_g_std;

  // decomposable compound prediction errors (unit normal scale)
  vector[N_decomp] fe_decomp_z;

  // undecomposable compound formation energy deviations (unit normal scale)
  vector[N_undecomp] fe_undecomp_z;

  // group formation energy deviations (unit normal scale)
  vector[N_group] fe_g_z;
}
transformed parameters {
  real mu_g = mean_y + mu_g_std * sd_y;      
  real mu_undecomp = mean_y + mu_undecomp_std * sd_y;
  vector[N_measurement_type] sigma = sigma_std * sd_y;
  real tau_g = tau_g_std * sd_y;
  real tau_undecomp = tau_undecomp_std * sd_y;
  real sigma_g = sigma_g_std * sd_y;
  vector[N_group] formation_energy_g_std = mu_g_std + fe_g_z * tau_g_std;
  vector[N_compound] formation_energy_c_std;
  {
    vector[N_decomp] decomp = G * formation_energy_g_std + fe_decomp_z * sigma_g_std;
    vector[N_undecomp] undecomp = mu_undecomp_std + fe_undecomp_z * tau_undecomp_std;
    formation_energy_c_std[compound_ix_decomp] = decomp;
    formation_energy_c_std[compound_ix_undecomp] = undecomp;
  }
}
model {
  fe_decomp_z ~ student_t(4, 0, 1);
  fe_undecomp_z ~ student_t(4, 0, 1);
  fe_g_z ~ student_t(4, 0, 1);
  // informative priors on interpretable scale
  // (no jacobian adjustments as transformations are linear)
  sigma[1] ~ normal(10, 10);
  sigma[2] ~ normal(50, 20);
  sigma[3] ~ normal(20, 10);
  sigma_g ~ normal(50, 30);
  tau_g ~ normal(200, 50);
  mu_g ~ normal(-300, 75);
  mu_undecomp ~ normal(-200, 200);
  tau_undecomp ~ normal(500, 300);
  // likelihood on standardised scale
  // (to keep untransformed parameters roughly unit-scale)
  if (likelihood == 1){
    vector[N_reaction] standard_delta_g_std = S' * formation_energy_c_std;
    y_std ~ student_t(4, standard_delta_g_std[rxn_ix], sigma_std[measurement_type]);
  }
}
generated quantities {
  vector[N_measurement] log_lik;
  vector[N_measurement] y_rep;
  vector[N_group] group_formation_energy;
  vector[N_compound] compound_formation_energy; 
  vector[N_reaction] standard_delta_g;
  {
    vector[N_reaction] standard_delta_g_std = S' * formation_energy_c_std;
    group_formation_energy = mean_y + formation_energy_g_std * sd_y;
    compound_formation_energy = mean_y + formation_energy_c_std * sd_y;
    standard_delta_g = mean_y + (S' * formation_energy_c_std) * sd_y;
    for (n in 1:N_measurement){
      log_lik[n] = student_t_lpdf(y_std[n] | 4, standard_delta_g_std[rxn_ix[n]], sigma_std[measurement_type[n]]);
      y_rep[n] = mean_y + student_t_rng(4, standard_delta_g_std[rxn_ix[n]], sigma_std[measurement_type[n]]) * sd_y;
    }
  }
}
