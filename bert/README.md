# Problem
Re-define the problem as binary classification.
Select two subjects: `material science` and `statistical mechanics`, and label them as `[0,1]`.
So the problem is to classify if an abstract belongs to `statistical mechanics` or not.

# Data
Select 10% of all abstract in the two subjects and train_test_split 9:1.
Using `abstract` only, and dropping `title`.

# Method
Fine-tune bert model following the steps in
`https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04`.

# Performance
