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

A Python script was made to pre-process the dataset before it could be used within the project, after this operation the processed dataset has the following dimensionality:

$$\langle 227166 \; samples \times 14 \; features \rangle$$

The `DataPoint` class has allowed for a more convenient use of the dataset samples within the project:

```scala
case class DataPoint(features: Seq[Double], label: Double)
```

## Classifiers

### Decision Tree

This model is made by a tree-like structure in which each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents a class label or a predicted value.

The tree is built recursively by selecting the best feature to split the data at each node. This selection is based on a certain impurity measure that quantifies the homogeneity of the labels in a subset of data.

In our implementation of the algorithm various hyper-parameters were included like impurity measurements, feature selection, maximum depth and minimum division size parameter.

The configuration used for both the algorithm within Spark and the one proposed is as follows:

* `MAX_DEPTH = 3`
* `IMPURITY_MEASURE = Impurities.Gini`
* `MIN_SPLIT_SIZE = 5`
* `SUBSET_STRATEGY = SubsetStrategies.All`

#### MapReduce

In the implementation of the DecisionTree algorithm, the MapReduce paradigm is utilized for feature selection that maximizes impurity gain. Specifically, an `RDD` is constructed consisting of the feature index as the key and the corresponding feature value for the sample.

```scala
case class Feature(id: Int, values: Seq[Double])

// RDD[DataPoint] => RDD[(Int, Double)]
```

From the previously generated `RDD[(Int, Double)]`, the keys are grouped to have vectors as values to find the best feature and threshold according to the impurity gain. The computed structure looks like an `RDD[(featureId, threshold, impurityGain)]`.

The `collect()` is then executed to return all computations performed in a distributed manner back to the driver, and the feature with the highest impurity gain is selected.

### Random Forest

This algorithm is an ensemble learning method used for classification. It involves building multiple decision trees during training and then combining their predictions to determine the class, with the final prediction being the mode of the classes predicted by the individual trees.

To reduce the correlation between individual trees in the forest and prevent overfitting, this algorithm randomly selects a subset of features at each division of the decision tree during training, rather than using all features as in traditional decision trees.

The configuration used for both the algorithm within Spark and the one proposed is as follows:

* `NUM_TREES = 5`
* `SUBSET_STRATEGY = SubsetStrategies.Sqrt`

#### MapReduce

The implementation of the Random Forest classifier is dependent on the implementation of the Decision Tree, which can leverage the MapReduce paradigm to compute the best feature and threshold based on impurity gain.

### kNN

The kNN classifier doesn't have a separate training and testing phase, unlike the two models described above.

It classifies a new data point from the `testData` by calculating the distance between it and all the data points inside the `trainData`.

Finally, It identifies the k-nearest data points based on their distance and assigns to the new data point the class label that is most common among these neighbors.

The configuration used for the algorithm proposed is as follows:

* `NUMBER_NEAREST_NEIGHBORS = 3`
* `DISTANCE_METHOD = Distances.Euclidean`

#### MapReduce

In the implementation of the kNN algorithm an RDD is constructed from the training set, where the distance between the test point and each point is calculated. Each row of the RDD will contain a tuple with a key equal to the label of the training data and the value as the distance of the testData from it. The top k rows of the previously calculated and sorted RDD are selected. Finally, the selected k elements are grouped by key and the one with the highest frequency is chosen.

## Evaluate

To assess the validity of the classifiers implemented with the MapReduce paradigm, a comparison was made between their performance and accuracy in predicting the labels of the dataset mentioned and those of the same algorithms implemented sequentially. Classifiers from [Spark's MLlib](https://spark.apache.org/mllib/) library were also compared in the evaluation.

### Prepare JAR

In order to execute the project on a cloud envirement the JAR with all the dependencies is needed. To build and assembly it the [sbt-assembly](https://github.com/sbt/sbt-assembly) plugin was used as shown below:

```bash
sbt clean
sbt compile
sbt assembly
```

The output FatJAR file will be saved inside the `project/target/scala-2.12` folder.

## Cloud testing

To test the algorithm via Google Cloud Platform (GCP), the first step is to enable in a Google Cloud Project the two services:

* Cloud Storage
* Dataproc

Installing the Google Cloud SDK for CLIs is recommended for utilizing GCP on your system. Do so following [this guide](https://cloud.google.com/sdk/docs/install) before perform the authentication.

```bash
gcloud auth login
```

### Creating the bucket in Cloud Storage

It is necessary to store all project files, including JAR executables and CSV datasets, in a Cloud Storage bucket.

```bash
# Creation
gcloud storage buckets create gs://$BUCKET_NAME --location $REGION
```

`$BUCKET_NAME` and `$REGION` can be environment variables, or you can just substitute them with the actual values.
Regions can be found [here](https://cloud.google.com/about/locations).

```bash
# Copy files to bucket
gsutil cp target/scala-2.12/andrp.jar gs://$BUCKET_NAME/andrp.jar

gsutil cp data/weatherAUS-reduced.csv gs://$BUCKET_NAME/weatherAUS-reduced.csv
```

### Provisioning a cluster in Dataproc

```bash
gcloud dataproc clusters create $CLUSTER_NAME --region=$REGION --zone=$ZONE --master-machine-type $MASTER_MACHINE_TYPE --worker-machine-type=$WORKER_MACHINE_TYPE --num-workers=$NUM_WORKER --image-version=$IMAGE_VERSION
```

Again, you can use environment variables or substitute them with values. The meaning of these variables is the following:

* `$CLUSTER_NAME` is the name of the cluster, you may choose one;
* `$REGION` and `$ZONE`, please follow the link in the section above;
* `$MASTER_MACHINE_TYPE` and `$WORKER_MACHINE_TYPE` can be chosen from [this list](https://cloud.google.com/compute/docs/machine-resource);
* `$NUM_WORKERS` is the number of total workers (the master is not included in this number) the master can utilize;
* `$IMAGE_VERSION` is the operating system used for the cluster and can be chosen from [this list](https://cloud.google.com/dataproc/docs/concepts/versioning/dataproc-version-clusters).

### Submitting a job in Dataproc

```bash
gcloud dataproc jobs submit spark --cluster=$CLUSTER_NAME --region=$REGION --jar=gs://$BUCKET_NAME/andrp.jar -- "yarn" "gs://$BUCKET_NAME/weatherAUS-reduced.csv" "sim=$SIMULATION" "lim=$LIMIT_SAMPLES" "ex=$EXECUTION" "gs://$BUCKET_NAME/$OUTPUT_FILE" "$NUM_RUN" 
```

The meaning of the unexplained variables is the following:

* `$SIMULATION`, can either be `true` or `false`;
* `$LIMIT_SAMPLES`, can either be `true` or `false`;
* `$EXECUTION`, can either be `distributed` or `sequential`;
* `$OUTPUT_FILE`, is a string that identifies the output test file;
* `$NUM_RUN`, is an integer value that identifies the number of times the simulation should be repeated.

### Download the results

```bash
gsutil -m cp -r gs://bucket-weather-australian/$OUTPUT_FILE python_scripts/results/.
```

## Results

A separate Python script was developed to analyze the results. This script computes the averages and confidence intervals of the output metrics, and generates plots to illustrate the findings.

### Strong scalability

<details>
  <summary>Click me</summary>

In order to test the strong scalability, all the test were done on 10.000 samples of the dataset with the following cluster configurations:
* 1 Worker, 4 Core
* 2 Worker, 8 Core
* 3 Worker, 12 Core
* 4 Worker, 16 Core

Where each worker uses an N1 processor with 4 core and 15 GB of Memory.

<img src="https://github.com/prushh/australia-next-day-rain-prediction/blob/main/python-scripts/results/strong/images/decisiontree.png" alt="Decision Tree plot" width="500" />

<img src="https://github.com/prushh/australia-next-day-rain-prediction/blob/main/python-scripts/results/strong/images/randomforest.png" alt="Random Forest plot" width="500" />

<img src="https://github.com/prushh/australia-next-day-rain-prediction/blob/main/python-scripts/results/strong/images/knn.png" alt="kNN plot" width="500" />

</details>

### Weak scalability

<details>
  <summary>Click me</summary>

To test the weak scalability, the test were done on a growing size of samples doubling it when the number of available core doubles. In particular, the used configuration are the following:

* 1 Worker, 2 Core, 2500 samples
* 1 Worker, 4 Core, 5000 samples
* 2 Worker, 8 Core, 10000 samples
* 4 Worker, 16 Core, 20000 samples

On these test the last 3 configurations are comparable with the previous one meanwhile the first has only 2 cores and 13 GB of memory.

<img src="https://github.com/prushh/australia-next-day-rain-prediction/blob/main/python-scripts/results/weak/images/decisiontree.png" alt="Decision Tree plot" width="500" />

<img src="https://github.com/prushh/australia-next-day-rain-prediction/blob/main/python-scripts/results/weak/images/randomforest.png" alt="Random Forest plot" width="500" />

<img src="https://github.com/prushh/australia-next-day-rain-prediction/blob/main/python-scripts/results/weak/images/knn.png" alt="kNN plot" width="500" />

</details>