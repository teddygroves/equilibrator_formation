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
  // measurement error (lower bound is best realistic measurment error)
  vector<lower=0.01>[N_measurement_type] sigma_std; 
  // group formation energy hyperparameters
  real<lower=0> tau_g_std;
  real mu_g_std;
  real<lower=0.01> sigma_g_std;  // group additivity validity
  // compound formation energy on standard normal scale
  vector[N_compound] fe_c_z;
  // group formation energy on standard normal scale
  vector[N_group] fe_g_z;
}
transformed parameters {
  real mu_g = mean_y_train + mu_g_std * sd_y_train;
  vector[N_measurement_type] sigma = sigma_std * sd_y_train;
  real tau_g = tau_g_std * sd_y_train;
  real sigma_g = sigma_g_std * sd_y_train;
}
model {
  vector[N_group] formation_energy_g_std = mu_g_std + fe_g_z * tau_g_std;
  vector[N_compound] formation_energy_c_std = G * formation_energy_g_std + fe_c_z * sigma_g_std;
  vector[N_reaction_train] standard_delta_g_std = S_train' * formation_energy_c_std;
  sigma ~ normal(10, 3);     // unstandardised measurement error
  sigma_g ~ normal(50, 30);  // don't know how accurate the additivity assumption is
  tau_g ~ normal(200, 50);   // group formation energies vary less than compound ones
  mu_g ~ normal(-300, 75);
  fe_c_z ~ std_normal();
  fe_g_z ~ std_normal();
  if (likelihood == 1){
    y_train_std ~ normal(standard_delta_g_std[rxn_ix_train], sigma_std[measurement_type_train]);
  }
}
generated quantities {
  vector[N_measurement_test] log_lik;
  vector[N_measurement_test] y_pred;
  vector[N_reaction_test] standard_delta_g_test;
  vector[N_group] group_formation_energy = mu_g + fe_g_z * tau_g;
  vector[N_compound] compound_formation_energy = G * group_formation_energy + fe_c_z * sigma_g;
  standard_delta_g_test = S_test' * compound_formation_energy;
  for (n in 1:N_measurement_test){
    log_lik[n] = normal_lpdf(y_test[n] | standard_delta_g_test[rxn_ix_test[n]], sigma[measurement_type_test[n]]);
    y_pred[n] = normal_rng(standard_delta_g_test[rxn_ix_test[n]], sigma[measurement_type_test[n]]);
  }
}
