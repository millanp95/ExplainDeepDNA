# ExplainDeepDNA


Source files for an interpretability study in Machine Learning models for DNA classification files.

Requisites:
1. Keras 2.0
2. TensorFlow > 1.13

For reproducing the results in the study: 
1. Import DeepExplain Library.

2. Run the Explain script with the following arguments:
```
python Explain.py < database_name >  < value of k > <model architecture> < label_of _interest>

database_name:[ dengue, HIV, HCV, HVB ]
model_architecture: [ascii, one_hot, avg_pool]
k:[1, 2, 3]
label_of_interest: Depends on the dataset
```
An example: 
```python Explain HIV 4 one_hot B
```
