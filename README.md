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


## Classification
Classification models should have X values in sub-directories that
correspond to their class. The parent directory is used as input
for the training.

## Regression
Regression models should have all X values in a single directory. 
There should be a corresponding DataFrame pickle or csv file, with
a column header of "Name" and column header of "XXX". <br>
"Name": X value file name WITHOUT extension
"XXX": Regression target, defaults to Energy