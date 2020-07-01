---
layout: page
title: Maximal margin classifier, Support vector classifier and Support vector machine Part 1
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script> 

# Maximal margin classifier, Support vector classifier and Support vector machine

In this article we will explore the Support Vector Machine algorithm . This article will be divided into two parts. The first one we explore Support vector machine, Support vector classifier and maximal margin classifier. The second one will contain a code implementation of support vector machine for textual classification . 

## Content

1. What is a hyperplane?
2. Maximal Margin Classifier
3. Support vector classifier
4. Support vector machine
5. Conclusion
6. Reference



SVM's or Support vector machine is one the sophisticated machine algorithm's available in the market. The SVM's are a generalization of the maximal margin classifier however people often confuse between the maximal margin classifier , the support vector classifier and the support vector machine. In this article we will try to differentiate between these 3 techniques . We will use the package scikit-learn  to import the SVM classifier.

Before jumping into exploring the 3 algorithm we would first like to understand the concept of hyperplane and how a separating hyperplane can be used to classify data points. 

## What is a Hyperplane?

A hyperplane is a subspace of $$p-1$$ dimension in a $$p$$-dimensional space, that is, if a hyperplane is defined in a $$2$$ dimensional space , a hyperplane will be a 1 dimensional subspace i.e. line. 

In this article we make the following assumptions:

1. Data Points are in a $$2$$-dimensional space.
2.  Target variable $$y$$  is binary where $$y_{i}$$ is a subset of $$(-1,1)$$. 

All the equation written below can be generalized for p-dimensional space.

The equation of a hyperplane in a $$1$$-dimension will be equation of a line represented as 
$$
\beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} = 0
$$


Any $$X$$ which will satisfy the above equation will lie on the hyperplane but what in the case, if the point $$X$$ does not satisfy the above equation ? This case will be represented by  the below situation
$$
\beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} >0 
$$
i.e the point $$X$$ will lie on one side of the hyperplane and if 
$$
\beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} < 0
$$
then point $$X$$ will lie on another side of the hyperplane. 

If there are p-dimensional with $$n$$ training example with $$y$$ i.e. the target variable belonging to either $$-1$$ or $$1$$ the equation will be generalized as below:
$$
\beta_{0} + \beta_{1}x_{i1} + \beta_{2}x_{i2}+......+\beta_{p}x_{ip} > 0 ~if ~ y_{i}= 1
$$
and 
$$
\beta_{0} + \beta_{1}x_{i1} + \beta_{2}x_{i2}+......+\beta_{p}x_{ip} < 0 ~if~ y_{i}= -1
$$


## Maximal Margin Classifier 

Assuming that our data can be perfectly separated by a separating hyperplane , Maximal Margin Classifier will divide the data point using a hyper plane such that the divided data points and the hyperplane has maximum margin. 

But what is Margin? If we draw a perpendicular line from each training observation to a given hyperplane the minimal such distance from the point to the hyperplane is known as margin. 

In Maximal Margin classifier we try to maximize the Margin and select a hyperplane which has maximum distance away from the training points and classify the training points correctly. 



![image-20200225145805222](C:\Users\yamini.rawat\AppData\Roaming\Typora\typora-user-images\image-20200225145805222.png)

The above image show the a hyperplane separating the two classes and dotted line represents the margin. The 3 points which lie on the margin are the once which are determining the margin and hence are known as support vectors. These points  *support* the hyperplane. If the position of these points change the separating hyperplane itself will change. The other points do not affect the hyperplane unless they cross the margin. 

The maximal margin classifier is a simple and elegant classifier however in real time we usually don't encounter data which can  distinctly be separated by a hyperplane . Also as p increase the maximal margin classifier tends to over fit . Data which is non-separable can be classified using Support Vector Classifier.



## Support Vector Classifier

Support vector classifier is a generalized version of Maximal margin classifier which can handle non-separable cases. This is possible because the support vector classifier considers soft margins i.e. the data points can cross the margin and also be on the wrong side of hyperplane which means that those points are misclassified. The margin is Soft because it can be violated by some observation. 

The advantage of Support Vector Classifier over Maximal Margin Classifier is that it gives better classification for most of the training observations. Also it is more robust and prevents over fitting of data.

The support vector classifier is defined by the below equation
$$
yi(β_{0} + β_{1}x_{i1} + β_{2}x_{i2} + . . . + β_{p}x_{ip}) ≥ M(1 − \epsilon_{i}),
$$

$$
\epsilon \ge0, \sum_{i=1}^n\epsilon \le C
$$

where $C$ is nonnegative tuning parameter , M is the margin which we always desire with a higher value and $$\epsilon_{1}...\epsilon_{n}$$ are slack variables.

Slack Variable tells us the position of the observation relative to hyperplane and the margin. If $$\epsilon_{i}=0$$ the $$i_{th}$$observation in on the correct side of the margin if $$\epsilon_{i}>0$$ the observation on the wrong side of the margin and if $$\epsilon > 1$$ the observation is on the wrong side of hyperplane.

now C bounds the sum of $$\epsilon_{i} $$ and it thus decides the budget for violation of the margin.

if $C$ = 0 the there is no budget and it becomes a maximal margin classifier. For $C$ > 0 no more that $C $observation can be on the wrong side of the hyperplane. As $C$ increases we become more tolerant towards violation and if $C$ is decreased we become less tolerant toward violation . The $C$ parameter is better explained in the example below.

![image-20200227233208257](C:\Users\yamini.rawat\AppData\Roaming\Typora\typora-user-images\image-20200227233208257.png)

The figure above shows 4 different value of $C$. The first image has largest value of $C$ and has a wide margin and the last pic has small value of $C$ and has smallest margin. As $C$ increases we become more tolerant and the more observation lie on the wrong side of the margin . However when we decrease the $C$ less and less observations lie on the wrong side of Margin.

The support vector classifier is quite robust because of $C$ parameter involved however this too can only be applied on a linearly separable data. What if the data was not linearly separable? Then what? 

The answer to this is Support vector machine .



## Support Vector Machine. 

A Support Vector machine is like a higher version/order of support vector classifier because unlike SVC , SVM can be applied to data distributed in Non-linear boundaries. SVM is able to work with non-linear boundary because it uses Kernel tricks to enlarge the feature space of the inputs.

 Feature- space of an input variable is the n-dimension in which the input is present. The feature space can be enlarged using the various functions of the predictors like quadratic or cubic or higher order polynomial functions.  Though their are many ways to increase a feature space of the predictors, one should be mindful regarding the huge number of feature they mind end up with and computation required for such large feature.

Now one must wonder how SVM can handle data which is non-linearly separated. In actual ground truth SVM is not dealing with a non-linear data but in fact with a linear data. The only difference is that you use a kernel trick you transform your data from a non-linear boundary to a linear boundary. Hence you say that the SVM are being applied to a data which in non-linearly divided .



Kernels

SVM uses kernels to transform the feature space. Kernels are the functions that take a low-dimensional feature space and map it to a high-dimensional feature space in which makes the data linearly separable. 

According to [scikit-learn documentation](https://scikit-learn.org/stable/modules/svm.html) the kernels can be of following type:

- linear: $$(x,x′)(x,x′)$$
- polynomial: $$(γ(x,x′)+r)d(γ(x,x′)+r)d$$. $$d$$is specified by keyword degree, $$r$$ by $$coef0$$
- rbf: $$exp(−γ∥x−x′∥2)exp(−γ‖x−x′‖2)$$. $$γ$$ is specified by keyword gamma, must be greater than $$0$$
- sigmoid: $$tanh(γ(x,x′)+r)tanh(γ(x,x′)+r)$$ where $$r$$ is specified by $$coef0$$
- custom



Radial Kernel

Radial kernel or the RBF is a popular kernel. The radial Kernel uses $\gamma$ as one of its parameter in addition to the Regularization parameter $C$.

$C$ parameter(explained above) lets us decide between a smooth decision boundary or over a overfit boundary by influencing the margin. $\gamma$ on the other, hand define how much influence a training point will have. If $\gamma$ is large then the support vector will have a  radius of influence only on the support vector itself and a higher $C$ value will also not prevent over fitting. When $\gamma$ is too small the radius of influence will be on the training data and no smooth decision boundary will be decided . You can use grid search and start at  the extreme value for $C$ and $\gamma$ to arrive at suitable values for both



## Conclusion

We saw the difference between all three classifier and how one can be derived from another. Stay tuned for part 2 in which we will perform Text classification using SVMs.

Kindly reach out to me on gmail or LinkedIn to connect.



## References

- [sklearn documentation](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)

- [Introduction to Statistical Learning in R](http://faculty.marshall.usc.edu/gareth-james/ISL/)

- [wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine)

- [udacity intro to machine learning course](https://www.udacity.com/course/intro-to-machine-learning--ud120)

  





