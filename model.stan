/* Simple statistical model of standard delta g measurements */
data {
  int<lower=1> N_measurement;
  int<lower=1> N_compound;
  int<lower=1> N_reaction;
  int<lower=1> N_group;
  vector[N_measurement] standard_delta_g_measured;
  int<lower=1,upper=N_reaction> measurement_reaction_ix[N_measurement];
  int<lower=0,upper=1> likelihood;
  matrix[N_reaction, N_compound] ST;
  matrix[N_compound, N_group] G;
}
parameters {
  real<lower=0> sigma;
  real<lower=0> tau;
  vector[N_compound] formation_energy;
  vector[N_group] group_formation_energy;
}
transformed parameters {
  vector[N_reaction] standard_delta_g = ST * formation_energy;
}
model {
  formation_energy ~ normal(G * group_formation_energy, tau);
  group_formation_energy ~ normal(0, 100);
  sigma ~ normal(0, 5);
  tau ~ normal(0, 5);
  if (likelihood == 1){
    vector[N_measurement] sdg_hat= standard_delta_g[measurement_reaction_ix];
    standard_delta_g_measured ~ normal(sdg_hat, sigma);
  }
}
