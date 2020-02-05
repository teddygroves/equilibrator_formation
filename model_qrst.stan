data {
  int<lower=1> N_compound;
  int<lower=1> N_reaction;
  int<lower=1> N_group;
  vector[N_reaction] standard_delta_g_measured;
  int<lower=0,upper=1> likelihood;
  matrix[N_reaction, N_compound] QST;
  matrix[N_compound, N_compound] RinvS;
}
parameters {
  real<lower=0> sigma;
  real<lower=0> tau;
  vector[N_compound] formation_energy_aux;
  vector[N_group] group_formation_energy;
}
transformed parameters {
  vector[N_compound] formation_energy = RinvS * formation_energy_aux;
}
model {
  formation_energy ~ normal(G * group_formation_energy, tau);
  group_formation_energy ~ normal(0, 100);
  sigma ~ normal(0, 5);
  tau ~ normal(0, 5);
  if (likelihood == 1){
    standard_delta_g_measured ~ normal(QST * formation_energy_aux, sigma);
  }
}
generated quantities {
  vector[N_reaction] standard_delta_g = QST * formation_energy_aux;
}
