# australia-next-day-rain-prediction
<p>
  <img src="https://img.shields.io/badge/Scala-2.12.15-red" alt="alternatetext">
  <img src="https://img.shields.io/badge/Spark-3.3.2-orange" alt="alternatetext">
  <img src="https://img.shields.io/badge/jdk-11.0.18-green" alt="alternatetext">
<img src="https://img.shields.io/badge/sbt-1.7.3-blue" alt="alternatetext">

</p>

Project for the course "Scalable and Cloud Programming" of the University of Bologna, A.Y. 2021/2022.

This project gives an implementation of a series of popular classification algorithms using the Scala language and the MapReduce paradigm.

## Dataset

The [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) dataset has been chosen to test the performance of our classifiers.
It contains about 10 years of rain history and has 22 features for predicting the target column named RainTomorrow.

In order to load the dataset inside the project a `DataPoint` structure was made:

```scala
case class DataPoint(features: Seq[Double], label: Double)
```

A Python script was made to pre-process the dataset before it could be used within the project.

## Classifiers

### Decision Tree

This model is made by a tree-like structure in which each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents a class label or a predicted value.

The tree is built recursively by selecting the best feature to split the data at each node. This selection is based on a certain impurity measure that quantifies the homogeneity of the labels in a subset of data.

In our implementation of the algorithm various hyper-parameters were included like impurity measurements, feature selection, maximum depth and minimum division size parameter.

Abbiamo messo questa possibilità così

#### MapReduce

In the implementation of the DecisionTree algorithm, the MapReduce paradigm is utilized for feature selection that maximizes impurity gain. Specifically, an `RDD[Features]` is constructed consisting of the feature index as the key and the corresponding feature value for the sample.

```scala
DataPoint => (featureId, sample.features(featureId))
```

From the previously generated `RDD` of features, the keys are grouped to have vectors as values to find the best threshold. The collect function is executed to return all computations performed in a distributed manner back to the driver, and the feature with the highest impurity gain is selected.

Our algorithm was also implemented with the use of the **Map Reduce paradigm**, where the measures for each label of the dataset are computed in parallel, and subsequently summed to obtain the final value.

### Random Forest
The **Random Forest** ensemble learning method for classification operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes of the individual trees.

During training, the algorithm selects a random subset of features at each split in the decision tree, rather than using all features like traditional decision trees. This process helps to reduce the correlation between the individual trees in the forest and reduces overfitting.

#### MapReduce

Our implementation of the Random Forest classification algorithm relies on our implementation of the Decision Tree, which, can utilize the **Map Reduce paradigm** for the calculation of the impurity measure.

### kNN

The **KNN** algorithm classifies new data points by calculating the distance between it and all the data points in the training set. 
It then selects the k-nearest data points based on their distance and assigns the class label of the majority of these neighbors to the new data point. 

#### MapReduce

In the implementation of the kNN algorithm an RDD is constructed from the training set, where the distance between the test point and each point is calculated. Each row of the RDD will contain a tuple with a key equal to the label of the training data and the value as the distance of the testData from it. The top k rows of the previously calculated and sorted RDD are selected. Finally, the selected k elements are grouped by key and the one with the highest frequency is chosen.

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
gcloud storage buckets create gs://$BUCKET_NAME
```

### Creating a cluster in Dataproc
Before running the following command, enable **Dataproc** as a service in your GCP Project.
```bash
gcloud dataproc clusters create $CLUSTER_NAME --region=$REGION  --num-workers=$NUM_WORKER --worker-machine-type=$WORKER_MACHINE_TYPE --image-version=$IMAGE_VERSION
```

### Uploading data in the bucket
First you need to upload the JAR of the project.
```bash
gsutil cp target/scala-2.12/andrp.jar gs://$BUCKET_NAME/andrp.jar
```
Then you need to upload the dataset.
```bash
gsutil cp data/weatherAUS-reduced.csv gs://$BUCKET_NAME/weatherAUS-reduced.csv
```

### Submitting a job in Dataproc
```bash
gcloud dataproc jobs submit spark --region=$REGION --cluster=$CLUSTER_NAME 
```

## References

1. Rain in Australia Dataset, J. Young, A. Young, [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
2. 