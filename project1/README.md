
# Designing lung cancer screening programs with machine learning
#### CPH 200A Project 1
#### Due Date: 5PM PST Oct 19, 2023

## Introduction

Lung cancer screening with low-dose computed tomography significantly improves patient lung cancer outcomes, improving survival and reducing morbidity; two large randomized control lung cancer screening trials have demonstrated 20% (NLST trial) and 24% (NELSON trial) reductions in lung cancer mortality respectively. These results have motivated the  development of national lung screening programs. The success of these programs hinges on their ability to the right patients for screening, balancing the benefits early detection against the harms of overscreening. This capacity relies on our ability to estimate a patients risk of developing lung cancer. In this class project, we will develop machine learning tools to predict lung cancer risk from PLCO questionnaires, develop screening guideline simulations, and compare the cost-effectiveness of these proposed guidelines against current NLST criteria. The goal of this project is to give you hands-on experience developing machine learning tools from scratch and analyzing their clinical implications in a real world setting. At the end of this project, you will write a short project report, describing your model and your analyses.  Submit your **project code** and a **project report** by the due date, **5PM pst on Oct 19th, 2023**.

## Part 0: Setup

### Logging into CPH Wynton  Nodes


You can login to the CPH Wynton nodes via ssh. The steps are: 

``` 
ssh username@plog1.wynton.edu 
ssh cph-app1.wynton.edu 
```

Note, we have two CPH-app nodes, namely `cph-app1` and `cph-app2`, that are reserved for Cornerstone coursework.  You can find additional information on Wynton on their [website](https://wynton.ucsf.edu/hpc/index.html).

You might the following unix tools useful: [tmux](https://github.com/tmux/tmux/wiki), [htop](https://htop.dev/) and [oh-my-zsh](https://ohmyz.sh/).

## Setting Development Environment

Starter project code is available in this github. You can clone this repository with the following command: 
```
 git clone git@github.com:yala/Cornerstone200A.git
```

To manage dependencies, we'll be using miniconda. You can load conda with the following command:
```
module load CBI miniconda3/23.3.1-0-py39
```

After loading `conda`, you can then create your `python3.10` environment and install the necessary python packages with the following commands:
```
conda create -n env_name python=3.10
conda activate env_name
pip install -r requirements.txt
```

The plco datasets, which include helpful data dictionaries and readmes, are availale at:
`/wynton/protected/project/cph/cornerstone/plco`

### Optional: Submitting jobs to the larger Wynton cluster with SGE
Additional GPUs are available to use as part of the larger Wynton cluster. You can submit jobs to this cluster using the Sun Grid Engine (SGE) scheduler. You can learn to use SGE (Sun Grid Engine) are available [here](https://wynton.ucsf.edu/hpc/scheduler/submit-jobs.html).

## Part 1: Model Development

In this part of the project, you will extend the starter code available in `vectorizer.py`, `logistic_regression.py` and `main.py`
to develop lung cancer risk models from the PLCO data. 

### Implementing a simple age-based Logistic Regression Classifier (20 pts)
To get started, we will implement logistic regression with Stochastic Gradient Descent to predict lung cancer risk using just patient age. 
Recall, we can define a logistic regression model as:
$p = \sigma(\theta x + b)$

where $\theta$ and $b$ refers to our model parameters, $x$ is our feature vector and $\sigma$ is the [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function.


We will train out model to perform classification using the binary cross entropy loss with L2 regulazarization.

$L(y, p) = - ( y log(p)) + (1-y)log( 1-p)) + \frac{\lambda}{2} ||\theta||^2$

where $y$ is the true label, $p$ is the predicted probability, and $\lambda$ is the regularization parameter.

To complete this part of the project, you will want to extract age data (column name is `"age"`) from the PLCO csv, featurize it, and implement SGD.
You will need to solve for the gradient of the loss with respect your model parameters. Note, pay special attention to the numerical
stability of your update rule. You may also need to play with 
the number of training steps, batch size, learning rate and regularization parameter. 

Your validation set ROC AUC should around `0.60`.

In your project report, please include a plot of your training and validation loss curves and describe the details of your model implementation.

### Implementing a simple grid search dispatcher (10 pts)

A key challenge in developing effective machine learning tools is experiment management. Even for a simple model, such your logistic regression model, and a simple structured dataset (i.e PLCO) there are wide range of hyperparameters to tune. It quickly becomes intractable to identify the best model configuration by hand. In this part of the project, you will develop a small job dispatcher that will run a grid search over a set of hyperparameters. Your dispatcher should take as input a list of hyperparameters and a list of values to search over. Your dispatcher should then run a job for each combination of hyperparameters. Your dispatcher should also keep track of the results of each job and summarize the results in a convenient format.  You can find some helpful starter code in `dispatcher.py`.

Complete the grid search dispatcher and use it to tune the hyperparameters of your age-based logistic regression model. In your project report, include a plot showing the relationship of L2 regularization and model training loss. 

### Building your best PLCO risk model (30 pts)

Now that you have build a simple single feature classifier, you will extend your model to include additional features from the PLCO dataset. A data dictionary from the NCI is available at `/wynton/protected/project/cph/cornerstone/plco/Lung/Lung Person (image only)/dictionary_lung_prsn-aug21-091521.pdf`.

Note, this includes a wide range of questionare features including smoking history, family history, and other demographic information. Some of this data is numeric and some is categorical, and you will need to develop an efficient way to featurize this data. Moreover, you will also need to decide how to handle missing data, and how to deal with scale of various features (e.g. age vs. pack years). For this step, you will find some hints on a suggested `vectorizer` design in `vectorizer.py`. Note, you do not need to use all the features in the questionnare.

Beyond a richer set of features, you will can want to consider more sophisticated models like Random Forest or Gradient Boosted Trees. You're invited to leverage the `sklearn` library for this part of the project to quickly explore other model implementations.

At the end of this phrase, your validation ROC AUC should be greater or equal to `0.83`.

In your project report, please including an test ROC plot of your final model, compared to your age-based model, and describe the details of your final model implementation. Please also include any interesting ablations from your model development process.

## Part 2: Evaluation

### Analyzing overall model performance (15 pts)
Now that you have developed your lung cancer model, and finalized your hyper-parameters, you will now focus on evaluating the performance of your model on the test set and on various subgroups of the test set. 

In your project report, include ROC curves and Precision recall curves of your best model and highlight the operation point of the current NLST criteria (available in the `"nlst_flag"` column). In addition to performance on the overall test, evaluate the performance of your model (using AUC ROC) on the following subgroups:

- sex (`sex` column)
- race (`race7` column)
- educational status (`educat` column)
- cigarette smoking status (`cig_stat` column)
- NLST eligiblity (`nlst_flag` column)

Are there any meaningful performance differences across these groups? What are the limitations of these analyses? What do these analyses tell us about our lung cancer risk model?

### Optional: Model interpretation (5 bonus Pts)
In addition to overall and subgroup analyses, list the top 3 most important features in your model. Note, depending on the type of model (e.g. tree method vs logistic regression) you use, you may need to leverage different model interpretability techniques. In your report, list the most important features and describe how you identified them.

### Simulating Clinical Utility (15 pts)
 Recall that lung cancer screening guidelines must balance the early detection of lung cancer against the harms of overscreening. In this part of the project, you will simulate the clinical utility of your model by comparing the cost-effectiveness of your model against the current national screening criteria (also known as the NLST criteria as available in the `nlst_flag` column).

To start off, compute the sensitivity, specificity and positive predictive value (PPV) of the NLST criteria on the PLCO test set. Note, you can use the `sklearn.metrics` library to compute these metrics.  If you were to match the specificity (i.e. amount of overscreening), sensitivity (i.e. fraction of cancer patients benefiting from early detection) or PPV (i.e. fraction of screened patients that will develop cancer) of the NLST criteria, what performance would your risk model enable?

How would you choose a risk threshold for lung screening and why? Note, this a subjective choice.

For your chosen risk threshold, please compute its performance metrics across the patient subgroups listed above.

## Part 3: Discussion

### Identifying limitations in study design (10 pts)
In the closing section of your project report, please discuss the implications of your findings and the limitations of these analyses in shaping lung cancer screening guidelines. What is missing in these analyses? What additional studies are needed to broaden clinical screening criteria?
