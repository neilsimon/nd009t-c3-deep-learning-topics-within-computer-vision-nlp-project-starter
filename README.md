# Dog Breed Classifier Project by Neil Simon
# Image Classification using AWS SageMaker

This project takes a pretrained CNN model (Resnet18) and finetunes it for use in classifying dog breeds into 133 classifications based on a Dog Breeds dataset.

The steps that the notebook goes through are:

1. Retrieving a dataset
1. Uncompressing that dataset.
1. Uploading that dataset to an S3 bucket.
1. Setting up hyperparameter tuning using learning rate, weight decay, eps and batch size using the AdamW optimizer on ResNet18.
1. Starting a hyperparameter tuning job using 20 training jobs (2 at a time) with autoshutdown on the AWS SageMaker Hyperparameter Tuning system.
1. Record the best hyperparameters as discovered from the above.
1. Train a fine-tuned model using ResNet18 and the best hyperparameters over a larger number of epochs, recording profiling and debug data.
1. Examine the output from the profiling and debug of the above.
1. Re-run the training with profiling and debug turned off (due to issues deploying model trained with the above).
1. Deploy that model as an endpoint on AWS.
1. Test that endpoint with new images.

## Hyperparameter Tuning
I chose to use the ResNet18 model with a single layer Linear NN. ResNet18 is a well proven pretrained CNN and the results of a single layer NN was enough to get good results from training.
The hyperparameters I chose to tune were:
1. Learning rate
1. eps
1. Weight decay
1. Batch size
These hyperparameters gave me broad coverage of the tunables for the system. I chose to use values between 0.1x and 10x the default as the default values tend to give good results for the first 3, and for batch size I stuck with the options of 32 and 64.


![Hyperparameter Training Job](Hyperparameter_training_job.png?raw=true "Hyperparameter training job")

![Hyperparameters best results](Hyperparameters_best_results.png?raw=true "Hyperparameters best results")

![Training job resource use](Training_job_resource_use.png?raw=true "Training job resource use")

![Training job output](Training_job_output.png?raw=true "Training job output")


### Results
I found debugging the system was painful. The long turnaround between making a change/fix, and it being tested was particularly bad. Additionally, the documentation for much of these systems is rather limited. For instance working out how to create the DataLoaders was far more difficult than it should have been based on what was ultimately required.

## Model Deployment
The deployed model runs a tuned version of ResNet18 and accepts data as preprocessed in the notebook.

![Deployed endpoint](deployed_endpoint.png?raw=true "Deployed endpoint")

![Penny for testing](testImages/Penny.png?raw=true "Penny for testing")

![Using the endpoint](Using the endpoint.png?raw=true "Using the endpoint")

