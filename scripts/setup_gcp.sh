sudo apt update
## Data from Kaggle 
#!/bin/bash
curl -L -o lichess-blitz-subsample.zip\
  https://www.kaggle.com/api/v1/datasets/download/naddleman/lichess-blitz-subsample
sudo apt install unzip -y 
## Unzip files 
unzip lichess-blitz-subsample.zip
## Create bucket
gcloud storage buckets create gs://my-bigdata-project-vc --project=victors-project-451221 --default-storage-class=STANDARD --location=us-central1 --uniform-bucket-level-access
## Create folders
gcloud storage cp /dev/null gs://my-bigdata-project-vc/landing
gcloud storage cp /dev/null gs://my-bigdata-project-vc/cleaned
gcloud storage cp /dev/null gs://my-bigdata-project-vc/trusted
gcloud storage cp /dev/null gs://my-bigdata-project-vc/code
gcloud storage cp /dev/null gs://my-bigdata-project-vc/models
## Check if bucket exists
gcloud storage ls -l gs://my-bigdata-project-vc
## Copy files to GCS bucket landing folder
gcloud storage cp lichess_blitz/lichess_blitz_subsample_*.pgn gs://my-bigdata-project-vc/landing/
