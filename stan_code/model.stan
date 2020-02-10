/* Simple statistical model of standard delta g measurements */
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
  real<lower=0> sigma;  // measurement error 
  real<lower=0> tau;    // sd of compound fe difference from sum of group fes
  vector[N_compound] formation_energy;
  vector[N_group] group_formation_energy;
}
transformed parameters {
  vector[N_reaction] standard_delta_g = S' * formation_energy;
}
model {
  formation_energy ~ normal(G * group_formation_energy, tau);
  group_formation_energy ~ normal(0, 1);
  sigma ~ lognormal(0, 1);
  tau ~ lognormal(0, 1);
  if (likelihood == 1){
    y ~ normal(standard_delta_g[rxn_ix], sigma);
  }
}
