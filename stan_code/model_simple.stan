data {
  int<lower=1> N_measurement_train;
  int<lower=1> N_measurement_test;
  int<lower=1> N_compound;
  int<lower=1> N_group;
  int<lower=1> N_measurement_type;
  int<lower=1> N_reaction_train;
  int<lower=1> N_reaction_test;
  vector[N_measurement_train] y_train;
  vector[N_measurement_test] y_test;
  int<lower=1,upper=N_reaction_train> rxn_ix_train[N_measurement_train];
  int<lower=1,upper=N_reaction_test> rxn_ix_test[N_measurement_test];
  int<lower=1,upper=N_measurement_type> measurement_type_train[N_measurement_train];
  int<lower=1,upper=N_measurement_type> measurement_type_test[N_measurement_test];
  int<lower=0,upper=1> likelihood;
  matrix[N_compound, N_reaction_train] S_train;
  matrix[N_compound, N_reaction_test] S_test;
  matrix[N_compound, N_group] G;
}
transformed data {
  real mean_y_train = mean(y_train);
  real sd_y_train = sd(y_train);
  vector[N_measurement_train] y_train_std = (y_train - mean_y_train) / sd_y_train;
}
parameters {
  vector<lower=0.01>[N_measurement_type] sigma_std;  // measurement error (standardised scale)
  real<lower=0> tau_g_std;                           // group formation energy variation (standardised scale)
  real mu_g_std;                                     // group formation energy mean (standardised scale)
  real<lower=0.01> sigma_g_std;                      // compound formation energy variation from predicted (standardised scale)
  vector[N_compound] fe_c_z;                         // compound formation energy deviations (unit normal scale)
  vector[N_group] fe_g_z;                            // group formation energy deviations (unit normal scale)
}
transformed parameters {
  real mu_g = mean_y_train + mu_g_std * sd_y_train;  // group formation energy mean (interpretable scale)
  vector[N_measurement_type] sigma = sigma_std * sd_y_train;  // measurement error (interpretable scale)
  real tau_g = tau_g_std * sd_y_train;               // group formation energy variation (interpretable scale)
  real sigma_g = sigma_g_std * sd_y_train;           // compound formation energy variation from predicted (interpretable scale)
}
model {
  fe_c_z ~ std_normal();
  fe_g_z ~ std_normal();
  // informative priors on interpretable scale (no jacobian adjustments as transformations are linear)
  sigma ~ normal(10, 3);
  sigma_g ~ normal(50, 30);
  tau_g ~ normal(200, 50);
  mu_g ~ normal(-300, 75);
  // likelihood on standardised scale to keep untransformed parameters roughly unit-scale
  if (likelihood == 1){
    vector[N_group] formation_energy_g_std = mu_g_std + fe_g_z * tau_g_std;
    vector[N_compound] formation_energy_c_std = G * formation_energy_g_std + fe_c_z * sigma_g_std;
    vector[N_reaction_train] standard_delta_g_std = S_train' * formation_energy_c_std;
    y_train_std ~ normal(standard_delta_g_std[rxn_ix_train], sigma_std[measurement_type_train]);
  }
}
generated quantities {
  vector[N_measurement_test] log_lik;
  vector[N_measurement_test] y_pred;
  vector[N_group] group_formation_energy = mu_g + fe_g_z * tau_g;
  vector[N_compound] compound_formation_energy = G * group_formation_energy + fe_c_z * sigma_g;
  vector[N_reaction_test] standard_delta_g_test = S_test' * compound_formation_energy;
  for (n in 1:N_measurement_test){
    log_lik[n] = normal_lpdf(y_test[n] | standard_delta_g_test[rxn_ix_test[n]], sigma[measurement_type_test[n]]);
    y_pred[n] = normal_rng(standard_delta_g_test[rxn_ix_test[n]], sigma[measurement_type_test[n]]);
  }
}
