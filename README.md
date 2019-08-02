# BOML

Adding a new model requires submodule in ml that contains:
models.py
{training}
parameters.py
{load_metaparameters, gen_hyperparameters}

Adjustments must be made to:
boml.optimization.py
{Optimizer.fetch_model_functions}

Adjustments can be made to:
boml.utils.sanity_checks.py
boml.utils.defaults.py