# australia-next-day-rain-prediction
<p>
  <img src="https://img.shields.io/badge/Scala-3.2.1-red" alt="alternatetext">
  <img src="https://img.shields.io/badge/Spark-3.2.1-orange" alt="alternatetext">
  <img src="https://img.shields.io/badge/jdk-17.0.5-green" alt="alternatetext">
</p>

This is a project for the 21/22 "Scalable and Cloud Programming" course of the University of Bologna.

This project gives an implementation of a series of popular classification algorithms using the Scala language and the MapReduce paradigm.

## Classifiers
### Decision Tree

### Random Forest

### Gradient Boosting

### KNN

## Dataset
We choose to test our classifiers with a dataset about [weather observations in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) that covers 10 years and has 22 features for predicting the target column "RainTomorrow".

We did a number of operations to pre-process it before running it with our algorithms, they can be found in the `preprocessing.py` script.
## Running in Google Cloud Platform
To test the algorithms, here are a series of gcloud shell commands to run in **Google Cloud Platform**.
```bash
gcloud auth login
```

### Creating a cluster in Dataproc
Before running the following command, enable **Dataproc** as a service in your GCP Project.
```bash
gcloud dataproc clusters create my-dataproc-cluster --region=europe-west1  --num-workers=2 --worker-machine-type=n1-standard-2 --image-version=2.1-debian11
```

### Submitting a job in Dataproc
```bash
gcloud dataproc jobs submit spark --region=europe-west1 --cluster=mycluster --class=it.unibo.andrp.Main --jars=gs://bucket-weather-australian/australia-next-day-rain-prediction_3-0.1.0-SNAPSHOT.jar
```


