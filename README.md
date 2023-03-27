# australia-next-day-rain-prediction
<p>
  <img src="https://img.shields.io/badge/Scala-2.12.15-red" alt="alternatetext">
  <img src="https://img.shields.io/badge/Spark-3.3.2-orange" alt="alternatetext">
  <img src="https://img.shields.io/badge/jdk-11.0.18-green" alt="alternatetext">
<img src="https://img.shields.io/badge/sbt-1.7.3-blue" alt="alternatetext">

</p>

Project for the course "Scalable and Cloud Programming" of the University of Bologna, A.Y. 2021/2022. This project gives an implementation of a series of popular classification algorithms using the Scala language and the MapReduce paradigm.

## Classifiers
### Decision Tree
A **Decision Tree** consists of a tree-like structure in which each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents a class label or a predicted value. 

The Decision Tree algorithm builds the tree recursively by selecting the **best feature** to split the data at each node. This selection is based on a certain **impurity measure** that quantifies the homogeneity of the labels in a subset of the data.

In our implementation of the Decision tree, we have included the option to choose various hyperparameters, including measures of impurity, feature selection, maximum depth, and the minimum split size parameter.

#### MapReduce implementation
In our implementation of the Decision tree, we utilize the MapReduce paradigm for feature selection that maximizes impurity gain. Specifically, we construct an RDD consisting of the feature index as the key and the corresponding feature value for the sample. From the previously generated RDD of features, the keys are grouped to have vectors as values to find the best threshold. Collect is performed to return all computations performed in a distributed manner back to the driver, and the feature with the highest impurity gain is selected.

### Random Forest
The **Random Forest** ensemble learning method for classification operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes of the individual trees.

During training, the algorithm selects a random subset of features at each split in the decision tree, rather than using all features like traditional decision trees. This process helps to reduce the correlation between the individual trees in the forest and reduces overfitting.

#### MapReduce implementation
Our implementation of the Random Forest classification algorithm relies on our implementation of the Decision Tree, which, can utilize the **Map Reduce paradigm** for selection of the feature with the best impurity gain.


### KNN
The **KNN** algorithm classifies new data points by calculating the distance between it and all the data points in the training set. 
It then selects the k-nearest data points based on their distance and assigns the class label of the majority of these neighbors to the new data point. 

#### MapReduce implementation
In our implementation of the KNN, we utilize the MapReduce paradigm for the classification of a single test point. Specifically, an RDD is constructed from the training set, where the distance between the test point and each point is calculated. Each row of the RDD will contain a tuple with a key equal to the label of the training data and the value as the distance of the testData from it. The top k rows of the previously calculated and sorted RDD are selected. Finally, the selected k elements are grouped by key and the one with the highest frequency is chosen.

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