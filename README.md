### Native Language Identification
Predict the native-language of a document based on its English translation.

### Task
Have to train a classifier to predict the native language of a document based on its English translation. The training data consists of a collection of 2983 documents (training examples) in a human readable format (.txt files). These .txt files contain the English translation of the original content in a native-language. Each training example is labelled with a number between 0 and 10, giving the native - language origin of its content.

The goal is to predict the label for the 1497 test documents.

### Process
1. Bag of Words with 1-gram(first 2000 most used) and 2-gram(first 600 most used)
2. Also added in the Bag of Words some punctuation statistics like: count of dots, commas, lines and apostrophes, given the fact that a language translation can be influenced by statistics like this.
3. Normalized the data and take under consideration only n-grams who are giving info, meaning if a n-gram is evenly distributed in the 11 classes(0 to 10) will not be taken under consideration.


### Classifiers and accuracy
1. Logistic Regression, accuracy = 91%
2. Multilayer Perceptron, accuracy = ~92%
