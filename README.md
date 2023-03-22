# australia-next-day-rain-prediction
<p>
  <img src="https://img.shields.io/badge/Scala-3.2.1-red" alt="alternatetext">
  <img src="https://img.shields.io/badge/Spark-3.2.1-orange" alt="alternatetext">
  <img src="https://img.shields.io/badge/jdk-17.0.5-green" alt="alternatetext">
</p>

This is a project for the 21/22 *Scalable and Cloud Programming* course of the University of Bologna.

This project gives an implementation of a series of popular classification algorithms using the Scala language and the MapReduce paradigm.

## Classifiers
### Decision Tree
A **Decision Tree** consists of a tree-like structure in which each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents a class label or a predicted value. 

The Decision Tree algorithm builds the tree recursively by selecting the **best feature** to split the data at each node. This selection is based on a certain **impurity measure** that quantifies the homogeneity of the labels in a subset of the data.

In our project we implemented two common impurity measures: 
* Weighted Gini Impurity
* Entropy Information Gain

#### MapReduce implementation
Our algorithm was also implemented with the use of the **Map Reduce paradigm**, where the measures for each label of the dataset are computed in parallel, and subsequently summed to obtain the final value.

### Random Forest
The **Random Forest** ensemble learning method for classification operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes of the individual trees.

During training, the algorithm selects a random subset of features at each split in the decision tree, rather than using all features like traditional decision trees. This process helps to reduce the correlation between the individual trees in the forest and reduces overfitting.

#### MapReduce implementation
Our implementation of the Random Forest classification algorithm relies on our implementation of the Decision Tree, which, can utilize the **Map Reduce paradigm** for the calculation of the impurity measure.

### Gradient Boosting
The **Gradient Boosting** algorithm works by combining multiple weak learners  to form a strong learner.

At each iteration, the algorithm computes the negative gradients of the current predictions using the Mean Squared Error loss function, and trains a new Decision Tree on the residuals of the current predictions. The new tree is then added to the ensemble, and the weights of the data points are updated based on the predictions of the current ensemble. 

#### MapReduce implementation
Our implementation of the Gradient Boosting classification algorithm relies on our implementation of the Decision Tree, which, can utilize the **Map Reduce paradigm** for the calculation of the impurity measure.

### KNN
The **KNN** algorithm classifies new data points by calculating the distance between it and all the data points in the training set. 
It then selects the k-nearest data points based on their distance and assigns the class label of the majority of these neighbors to the new data point. 

#### MapReduce implementation

## Dataset
We choose to test our classifiers with a dataset about [weather observations in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) that covers 10 years and has 22 features for predicting the target column *RainTomorrow*.

We did a number of operations to pre-process it before running it with our algorithms, they can be found in the `preprocessing.py` script.

## Testing
In order to evaluate the validity of our classification algorithm implementations utilizing  the MapReduce paradigm, 
we have decided to compare their performance and accuracy in predicting the labels of the dataset mentioned earlier, with that of the same algorithms implemented *without* the paradigm and the ones by [Spark's MLlib library](https://spark.apache.org/mllib/).

### Results

## Running in Google Cloud Platform
To test the algorithms, here are a series of gcloud shell commands to run in **Google Cloud Platform**.
```bash
gcloud auth login
```
### Creating a bucket in Cloud Storage
Before running the following command, enable **Cloud Storage** as a service in your GCP Project.
```bash
```

### Creating a cluster in Dataproc
Before running the following command, enable **Dataproc** as a service in your GCP Project.
```bash
gcloud dataproc clusters create my-dataproc-cluster --region=europe-west1  --num-workers=2 --worker-machine-type=n1-standard-2 --image-version=2.1-debian11
```

### Uploading data in the bucket
First you need to upload the JAR of the project.
```bash
```
Then you need to upload the dataset.
```bash
```

### Submitting a job in Dataproc
```bash
gcloud dataproc jobs submit spark --region=europe-west1 --cluster=mycluster --class=it.unibo.andrp.Main --jars=gs://bucket-weather-australian/australia-next-day-rain-prediction_3-0.1.0-SNAPSHOT.jar
```

## References

1. Rain in Australia Dataset, J. Young, A. Young, [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
2. 