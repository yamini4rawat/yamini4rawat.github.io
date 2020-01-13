---
layout: page
title: INTRODUCTION TO MACHINE LEARNING
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script> 
 

# INTRODUCTION TO MACHINE LEARNING

### What is machine learning ?

Coursera recently published a report stating that out of 45 million people enrolled in their platform, 2 million are enrolled in AI and machine learning based content. But what is machine learning? In the nut shell machine learning is just mathematics . While solving any problem through machine learning  we fit our inputs data to a suitable mathematical formula and predict applying the same mathematical formula to test data. The input data and the test data actually belong to the same statistical distribution. If they lie somewhat out of the statistical distribution they are called anomalies or outliers. 

### Types of machine learning

Guess what - there are types of machine learning too. A machine can learn in a supervised or unsupervised  or semi- supervised  or in a reinforced way.



#### Supervised Learning

For supervised machine learning the input data has labels i.e. the target variable $y^i$ is mentioned. So in a supervised machine learning the model takes in the input data and tries to devise a output label for that particular data. Some examples of supervised learning algorithms are *Linear Regression, Logistic Regression, Navies Bayes, Support vector machines etc.*



#### Unsupervised Learning 

In unsupervised learning the output label is missing and the machine has to learn about the data without any output label. The unsupervised machine algorithms are given an input feature but not a targeted output variable. Through unsupervised learning you can do clustering and determine the anomalies and outliers in the data provided. You can also do dimensionality reduction. Examples of unsupervised learning are *K-means clustering algorithm, PCA(Principal Component Analysis), LDA(Latent Dirichlet Allocation).*



#### Semi-supervised Learning 

In a semi-supervised learning we have a combination of labeled and unlabeled data. In semi-supervised learning the data with labeled examples are few and the unlabeled examples are more. First, the algorithm clusters the unlabeled data in clusters and then with the help of labeled data determines the labels of the unlabeled data. The concept of semi-supervised algorithm is kind of same as Supervised but having unlabeled data introduces more robustness and more information and this in turn results in a better model. 



#### Reinforcement Learning

Reinforcement learning is the new popular kid in the town. In Reinforcement learning there is concept of a reward. Rather that making a model the machine tries to learn a policy. A policy is a function that takes the input as a feature vector and outputs an action. The application of the action will give a positive reward or a negative reward. Reinforcement learning is applied in problems where decision making is required in a sequential manner.

It is because of reinforcement learning the machines today have become capable of beating humans at games like Atari.





### Naïve Bayes

In this section we will talk in more detail regarding a supervised machine learning algorithm - *Naïve Bayes*. Naïve Bayes algorithm are based on Bayes' Theorem . Naïve Bayes Algorithm calculates the probability of a input belonging to a particular class considering only one feature at a time.

**Naïve Bayes during classification assumes independence between features**. For example a day might get classified as sunny day if its humid, wind is fast and temperature is hot. But Naïve Bayes will classify the day as sunny considering each feature individually, even when the features might be depended on each other or any new feature. This is the reason why during textual classification Naïve Bayes does not capture the context of a sentence.



#### Mathematics 

Naïve Bayes is based on Bayes theorem. Bayes theorem mathematical is represented as follows:

# $$P(A\mid B)={\frac {P(B\mid A)P(A)}{P(B)}}$$
<br>
* $$P(A|B)$$ - This is a conditional probability. It is the probability of feature $$B$$ belonging to class $$A$$ or a class being identified as $$A$$ when $$B$$ is present. In probability terms its the likelihood of an event $$A$$ occurring when $$B$$ has already occurred. This is also known as posterior probability.

* $$P(B|A)$$ - This too is a conditional probability. Given a class $$A$$ what is the probability that $$B$$ is feature of class $$A$$. This also is a likelihood of event $$B$$ occurring given $$A$$ is true.

* $$P(A)$$ - Probability of event $$A$$ occurring. This is prior probability of class $$A$$.

* $$P(B)$$ - Probability of event $$B$$ occurring. This is prior probability of feature $$B$$.



Now lets dive deep into Naïve Bayes and fit Gaussian Naïve Bayes algorithm to an input data and make predictions on a test data. 

We will be working with Sklearn library to import the Gaussian Naïve Bayes. The dataset used is Iris data set. Iris dataset has 3 classes in it - *Iris-setosa, Iris-versicolor and Iris-virginica*. The Characteristic given in the dataset are SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm. We will use these features to train our model. Now lets import the required packages. 

```python
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.naive_bayes import GaussianNB
```

We will make an object of the GaussianNB classifier and fit our train data to it. Next we make prediction on our test data using the already trained model (fitting is the process of training the data).



```python
>>> gnb = GaussianNB()
>>> gnb.fit(X_train, y_train)
>>> y_pred = gnb.predict(X_test)
>>> print("Number of mislabeled points out of a total %d points : %d"
...       % (X_test.shape[0], (y_test != y_pred).sum()))
Number of mislabeled points out of a total 75 points : 4
```



We can also calculate the accuracy by importing accuracy_score from sklearn library.

```python
>>> from sklearn.metrics import accuracy_score
>>> acc = accuracy_score(y_test,y_pred)
>>> print("accuracy of the dataset is: %s "%acc)
accuracy of the dataset is: 0.9466666666666667 
```



### Conclusion

Naïve Bayes has naïve assumptions and therefore sometimes do not work on complex problem nevertheless it still is one of the fastest and easiest algorithm to use. Due to these feature it sometimes outperform the complex and sophisticated algorithm available. Its is simple, easy to build and very useful when it come to textual classification. We also see that in numerical classification too it works amazing as we got a accuracy of 94% and miss classified just 4 data points.



### References and Credits 

- [Sklearn documnetation](https://scikit-learn.org/stable/modules/naive_bayes.html)

- [Wikipedia Bayes’ Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem),[Wikipedia Naïve Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

- [udacity intro to machine learning course](https://www.udacity.com/course/intro-to-machine-learning--ud120)

