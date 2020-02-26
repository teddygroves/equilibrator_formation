data {
  int<lower=1> N_measurement;
  int<lower=1> N_compound;
  int<lower=1> N_group;
  int<lower=1> N_measurement_type;
  int<lower=1> N_reaction;
  vector[N_measurement] y;
  int<lower=1,upper=N_reaction> rxn_ix[N_measurement];
  int<lower=1,upper=N_measurement_type> measurement_type[N_measurement];
  int<lower=0,upper=1> likelihood;
  matrix[N_compound, N_reaction] S;
  matrix[N_compound, N_group] G;
}
transformed data {
  real mean_y = mean(y);
  real sd_y = sd(y);
  vector[N_measurement] y_std = (y - mean_y) / sd_y;
}
parameters {
  vector<lower=0>[N_measurement_type] sigma_std;     // measurement error (standardised scale)
  real<lower=0> tau_g_std;                           // group formation energy variation (standardised scale)
  real mu_g_std;                                     // group formation energy mean (standardised scale)
  real<lower=0> sigma_g_std;                         // compound formation energy variation from predicted (standardised scale)
  vector[N_compound] fe_c_z;                         // compound formation energy deviations (unit normal scale)
  vector[N_group] fe_g_z;                            // group formation energy deviations (unit normal scale)
}
transformed parameters {
  real mu_g = mean_y + mu_g_std * sd_y;                 // group formation energy mean (interpretable scale)
  vector[N_measurement_type] sigma = sigma_std * sd_y;  // measurement error (interpretable scale)
  real tau_g = tau_g_std * sd_y;                        // group formation energy variation (interpretable scale)
  real sigma_g = sigma_g_std * sd_y;                    // compound formation energy variation from predicted (interpretable scale)
}
model {
  fe_c_z ~ std_normal();
  fe_g_z ~ std_normal();
  // informative priors on interpretable scale (no jacobian adjustments as transformations are linear)
  sigma[1] ~ normal(10, 10);
  sigma[2] ~ normal(50, 20);
  sigma[3] ~ normal(20, 10);
  sigma_g ~ normal(50, 30);
  tau_g ~ normal(200, 50);
  mu_g ~ normal(-300, 75);
  // likelihood on standardised scale to keep untransformed parameters roughly unit-scale
  if (likelihood == 1){
    vector[N_group] formation_energy_g_std = mu_g_std + fe_g_z * tau_g_std;
    vector[N_compound] formation_energy_c_std = G * formation_energy_g_std + fe_c_z * sigma_g_std;
    vector[N_reaction] standard_delta_g_std = S' * formation_energy_c_std;
    y_std ~ normal(standard_delta_g_std[rxn_ix], sigma_std[measurement_type]);
  }
}
generated quantities {
  vector[N_measurement] log_lik;
  vector[N_measurement] y_rep;
  vector[N_group] group_formation_energy;
  vector[N_compound] compound_formation_energy; 
  vector[N_reaction] standard_delta_g;
  {
    vector[N_group] formation_energy_g_std = mu_g_std + fe_g_z * tau_g_std;
    vector[N_compound] formation_energy_c_std = G * formation_energy_g_std + fe_c_z * sigma_g_std;
    vector[N_reaction] standard_delta_g_std = S' * formation_energy_c_std;
    group_formation_energy = mean_y + formation_energy_g_std * sd_y;
    compound_formation_energy = mean_y + formation_energy_c_std * sd_y;
    standard_delta_g = mean_y + (S' * formation_energy_c_std) * sd_y;
    for (n in 1:N_measurement){
      log_lik[n] = normal_lpdf(y_std[n] | standard_delta_g_std[rxn_ix[n]], sigma_std[measurement_type[n]]);
      y_rep[n] = mean_y + normal_rng(standard_delta_g_std[rxn_ix[n]], sigma_std[measurement_type[n]]) * sd_y;
    }
  }
}
