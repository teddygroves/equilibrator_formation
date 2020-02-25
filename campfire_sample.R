library(campfire)
library(rstan)

model_path = "stan_code/model_simple.stan"

data_env = new.env(parent = baseenv())
source("data/input_data_non_default_ionic_strength.R", local = data_env)
data = as.list.environment(data_env)

stan_fit = stan(model_path, data = data, iter = 1)
out = warmup(model_path,
             stan_fit,
             data = data,
             num_chains = 2,
             print_stdout = FALSE)

model = cmdstanr::cmdstan_model(model_path)
fit = do.call(model$sample, out$args)


