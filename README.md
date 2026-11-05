# AML-KaggleCompetion-Group33

## Preprocessing

preprocess.py contains all pre processing functions. To preprocess data for model training and testing call

```
preprocess_dat(featurization=tfidf_data, n=8000, random_state=12):
```

**featurization** = featurization function for transofrming data into usable data for model training. Default is tf-idf featurization

**n** = sample size of training data, this is calculated after validation split so validation will always be the same 2,000 points for each random state

**random_state** = random state for randomization functions default is 12