/* 
   Simple statistical model of standard delta g measurements using non-centred
   parameterisation
*/
data {
  int<lower=1> N_measurement;
  int<lower=1> N_compound;
  int<lower=1> N_reaction;
  int<lower=1> N_group;
  vector[N_measurement] y;
  int<lower=1,upper=N_reaction> rxn_ix[N_measurement];
  int<lower=0,upper=1> likelihood;
  matrix[N_compound, N_reaction] S;
  matrix[N_compound, N_group] G;
}
parameters {
  real log_sigma;  // measurement error 
  real log_tau;    // sd of compound fe difference from sum of group fes
  real<lower=0> sd_gfe;
  real mu_gfe;
  vector[N_compound] fe_z;
  vector[N_group] gfe_z;
}
model {
  real sigma = exp(log_sigma);
  real tau = exp(log_tau);
  vector[N_group] group_formation_energy = mu_gfe + gfe_z * sd_gfe;
  vector[N_compound] formation_energy = G * group_formation_energy + fe_z * tau;
  vector[N_reaction] standard_delta_g = S' * formation_energy;
  fe_z ~ std_normal();
  gfe_z ~ std_normal();
  mu_gfe ~ std_normal();
  sd_gfe ~ normal(0, 3);
  log_sigma ~ normal(0, 1);
  log_tau ~ normal(0, 1);
  if (likelihood == 1){
    y ~ normal(standard_delta_g[rxn_ix], sigma);
  }
}
