# BOML
Bayesian optimized machine learning. Under continual development 
with addition of new models and architectures. 

Until further publication, please cite:
Maffettone P.M., Cooper, A.I.  
Deep learning from crystallographic representations of periodic systems. 
In: Fall 2019 ACS National Meeting; 2019 Aug 25--29; San Diego; ACS; 2019. No CINF 84. 

Abstract [here](https://tpa.acs.org/abstract/acsnm258-3206522/deep-learning-from-crystallographic-representations-of-periodic-systems).

Adding a new model requires submodule in ml that contains:
models.py
{change appopriate conditional statement in training.py}
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

### Requires updates on addition of new models
boml.optmization.Optimizer.fetch_model_functions() <br>
boml.ml.XXX_models.training.gen_model()