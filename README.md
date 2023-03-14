# australia-next-day-rain-prediction

gcloud shell commands to run on gcp:
gcloud auth login

gcloud dataproc clusters create my-dataproc-cluster --region=europe-west1  --num-workers=2 --worker-machine-type=n1-standard-2 --image-version=2.1-debian11
gcloud dataproc jobs submit spark --region=europe-west1 --cluster=mycluster --class=it.unibo.andrp.Main --jars=gs://bucket-weather-australian/australia-next-day-rain-prediction_3-0.1.0-SNAPSHOT.jar

