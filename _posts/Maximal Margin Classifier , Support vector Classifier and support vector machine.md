Maximal Margin Classifier , Support vector Classifier and support vector machine

1. SVM or support vector machines are a class of a sophisticated algorithm used in supervised machine learning . The SVMs can be used for classification as well as regression analysis and can be used for separating linear as well as non- linear data.
2. This article has been written with the aim to differentiate between the maximal margin classifier, support vector classifier and support vector Machines.  
3. Maximal Margin Classifier : Maximal Margin classifier separates the training example  based on hyper planes know as the maximal margin hyperplane or optimal separating hyperplane which is farthest from the training examples. To divide training example there can be infinite number of hyper plane but the hyper plane with the maximum *margin* is classified as maximal margin hyper plane . 
4. how do we define margin ? if we calculate the perpendicular distance of each training example from the hyperplane , the smallest such distance is called the margin . Therefore the maximal margin hyper plane is the one with largest margin . Although the maximal margin classifier is good in classification it can lead to overfitting if number of classes to classify is large.
5. Support vectors :  the observation or observations closest to the hyperplane are called support vectors (training example as vectors in p - dimensional space) as they "support" the maximal margin hyperplane . If these points/observation move slightly the maximal margin hyperplane will move too.
6. The maximal margin hyperplane only depend on the small set observations i.e. the support vectors and not all the observations . the movement of other observations unless the observation does not cross the boundary set by margin does not affect the separating plane. The fact that the maximal margin hyperplane directly depends only on a small set of observations is an important property.
7. Generalization of maximal margin classifier to non separable cases is known as support vector classifier. if we cannot exactly separate two classes but we just almost separate the classes by a using a so-called soft margin.
8. 

