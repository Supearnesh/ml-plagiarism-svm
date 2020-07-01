# Plagiarism Detection Model

Now that you've created training and test data, you are ready to define and train a model. Your goal in this notebook, will be to train a binary classification model that learns to label an answer file as either plagiarized or not, based on the features you provide the model.

This task will be broken down into a few discrete steps:

* Upload your data to S3.
* Define a binary classification model and a training script.
* Train your model and deploy it.
* Evaluate your deployed classifier and answer some questions about your approach.

To complete this notebook, you'll have to complete all given exercises and answer all the questions in this notebook.
> All your tasks will be clearly labeled **EXERCISE** and questions as **QUESTION**.

It will be up to you to explore different classification models and decide on a model that gives you the best performance for this dataset.

---

## Load Data to S3

In the last notebook, you should have created two files: a `training.csv` and `test.csv` file with the features and class labels for the given corpus of plagiarized/non-plagiarized text data. 

>The below cells load in some AWS SageMaker libraries and creates a default bucket. After creating this bucket, you can upload your locally stored data to S3.

Save your train and test `.csv` feature files, locally. To do this you can run the second notebook "2_Plagiarism_Feature_Engineering" in SageMaker or you can manually upload your files to this notebook using the upload icon in Jupyter Lab. Then you can upload local files to S3 by using `sagemaker_session.upload_data` and pointing directly to where the training data is saved.


```python
import pandas as pd
import boto3
import sagemaker
```


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# create an S3 bucket
bucket = sagemaker_session.default_bucket()
```

## EXERCISE: Upload your training data to S3

Specify the `data_dir` where you've saved your `train.csv` file. Decide on a descriptive `prefix` that defines where your data will be uploaded in the default S3 bucket. Finally, create a pointer to your training data by calling `sagemaker_session.upload_data` and passing in the required parameters. It may help to look at the [Session documentation](https://sagemaker.readthedocs.io/en/stable/session.html#sagemaker.session.Session.upload_data) or previous SageMaker code examples.

You are expected to upload your entire directory. Later, the training script will only access the `train.csv` file.


```python
# should be the name of directory you created to save your features data
data_dir = 'plagiarism_data'

# set prefix, a descriptive name for a directory  
prefix = 'plagiarism_data'

# upload all data to S3
input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix=prefix)
```

### Test cell

Test that your data has been successfully uploaded. The below cell prints out the items in your S3 bucket and will throw an error if it is empty. You should see the contents of your `data_dir` and perhaps some checkpoints. If you see any other files listed, then you may have some old model files that you can delete via the S3 console (though, additional files shouldn't affect the performance of model developed in this notebook).


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# confirm that data is in S3 bucket
empty_check = []
for obj in boto3.resource('s3').Bucket(bucket).objects.all():
    empty_check.append(obj.key)
    print(obj.key)

assert len(empty_check) !=0, 'S3 bucket is empty.'
print('Test passed!')
```

    counties/kmeans-2020-04-29-23-29-12-779/output/model.tar.gz
    counties/kmeans-2020-04-30-00-14-17-609/output/model.tar.gz
    counties/pca-2020-04-29-18-42-33-431/output/model.tar.gz
    counties/pca-2020-04-29-23-49-22-353/output/model.tar.gz
    counties/pca-2020-04-30-00-02-39-867/output/model.tar.gz
    creditcard/linear-learner-2020-05-29-05-18-05-737/output/model.tar.gz
    creditcard/linear-learner-2020-05-29-05-34-23-202/output/model.tar.gz
    creditcard/linear-learner-2020-05-30-20-55-34-624/output/model.tar.gz
    creditcard/linear-learner-2020-05-30-20-56-57-613/output/model.tar.gz
    creditcard/linear-learner-2020-05-30-21-11-54-961/output/model.tar.gz
    creditcard/linear-learner-2020-05-30-21-30-38-043/output/model.tar.gz
    creditcard/linear-learner-2020-05-30-21-43-35-286/output/model.tar.gz
    creditcard/linear-learner-2020-05-30-21-58-33-160/output/model.tar.gz
    creditcard/linear-learner-2020-05-30-22-13-51-669/output/model.tar.gz
    plagiarism_data/test.csv
    plagiarism_data/train.csv
    sagemaker-pytorch-2020-04-20-01-05-17-719/source/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-20-01-05-57-860/source/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-20-01-07-18-470/source/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-20-02-46-31-199/debug-output/training_job_end.ts
    sagemaker-pytorch-2020-04-20-02-46-31-199/output/model.tar.gz
    sagemaker-pytorch-2020-04-20-02-46-31-199/source/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-20-06-19-34-575/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-20-06-49-24-672/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-20-16-28-35-484/source/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-20-16-28-53-211/debug-output/training_job_end.ts
    sagemaker-pytorch-2020-04-20-16-28-53-211/output/model.tar.gz
    sagemaker-pytorch-2020-04-20-16-28-53-211/source/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-20-16-29-22-443/debug-output/training_job_end.ts
    sagemaker-pytorch-2020-04-20-16-29-22-443/output/model.tar.gz
    sagemaker-pytorch-2020-04-20-16-29-22-443/source/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-20-18-45-57-686/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-20-21-43-14-509/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-20-22-23-22-990/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-21-01-38-44-484/debug-output/training_job_end.ts
    sagemaker-pytorch-2020-04-21-01-38-44-484/output/model.tar.gz
    sagemaker-pytorch-2020-04-21-01-38-44-484/source/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-21-01-56-43-576/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-21-02-15-07-693/debug-output/training_job_end.ts
    sagemaker-pytorch-2020-04-21-02-15-07-693/output/model.tar.gz
    sagemaker-pytorch-2020-04-21-02-15-07-693/source/sourcedir.tar.gz
    sagemaker-pytorch-2020-04-21-02-33-39-086/sourcedir.tar.gz
    sagemaker-record-sets/KMeans-2020-04-29-23-28-42-924/.amazon.manifest
    sagemaker-record-sets/KMeans-2020-04-29-23-28-42-924/matrix_0.pbr
    sagemaker-record-sets/KMeans-2020-04-30-00-14-16-993/.amazon.manifest
    sagemaker-record-sets/KMeans-2020-04-30-00-14-16-993/matrix_0.pbr
    sagemaker-record-sets/LinearLearner-2020-05-29-05-17-35-036/.amazon.manifest
    sagemaker-record-sets/LinearLearner-2020-05-29-05-17-35-036/matrix_0.pbr
    sagemaker-record-sets/LinearLearner-2020-05-30-20-55-25-809/.amazon.manifest
    sagemaker-record-sets/LinearLearner-2020-05-30-20-55-25-809/matrix_0.pbr
    sagemaker-record-sets/LinearLearner-2020-05-30-20-56-48-927/.amazon.manifest
    sagemaker-record-sets/LinearLearner-2020-05-30-20-56-48-927/matrix_0.pbr
    sagemaker-record-sets/LinearLearner-2020-05-30-21-30-29-199/.amazon.manifest
    sagemaker-record-sets/LinearLearner-2020-05-30-21-30-29-199/matrix_0.pbr
    sagemaker-record-sets/PCA-2020-04-29-18-42-18-262/.amazon.manifest
    sagemaker-record-sets/PCA-2020-04-29-18-42-18-262/matrix_0.pbr
    sagemaker-record-sets/PCA-2020-04-29-23-49-12-401/.amazon.manifest
    sagemaker-record-sets/PCA-2020-04-29-23-49-12-401/matrix_0.pbr
    sagemaker-record-sets/PCA-2020-04-30-00-02-37-443/.amazon.manifest
    sagemaker-record-sets/PCA-2020-04-30-00-02-37-443/matrix_0.pbr
    sagemaker-scikit-learn-2020-07-01-01-44-22-842/source/sourcedir.tar.gz
    sagemaker-scikit-learn-2020-07-01-01-55-13-973/source/sourcedir.tar.gz
    sagemaker-scikit-learn-2020-07-01-02-13-13-175/source/sourcedir.tar.gz
    sagemaker-scikit-learn-2020-07-01-02-19-23-233/source/sourcedir.tar.gz
    sagemaker-scikit-learn-2020-07-01-02-32-47-817/debug-output/training_job_end.ts
    sagemaker-scikit-learn-2020-07-01-02-32-47-817/output/model.tar.gz
    sagemaker-scikit-learn-2020-07-01-02-32-47-817/source/sourcedir.tar.gz
    sagemaker-scikit-learn-2020-07-01-02-33-02-389/debug-output/training_job_end.ts
    sagemaker-scikit-learn-2020-07-01-02-33-02-389/output/model.tar.gz
    sagemaker-scikit-learn-2020-07-01-02-33-02-389/source/sourcedir.tar.gz
    sagemaker/sentiment_rnn/train.csv
    sagemaker/sentiment_rnn/word_dict.pkl
    Test passed!


---

# Modeling

Now that you've uploaded your training data, it's time to define and train a model!

The type of model you create is up to you. For a binary classification task, you can choose to go one of three routes:
* Use a built-in classification algorithm, like LinearLearner.
* Define a custom Scikit-learn classifier, a comparison of models can be found [here](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html).
* Define a custom PyTorch neural network classifier. 

It will be up to you to test out a variety of models and choose the best one. Your project will be graded on the accuracy of your final model. 
 
---

## EXERCISE: Complete a training script 

To implement a custom classifier, you'll need to complete a `train.py` script. You've been given the folders `source_sklearn` and `source_pytorch` which hold starting code for a custom Scikit-learn model and a PyTorch model, respectively. Each directory has a `train.py` training script. To complete this project **you only need to complete one of these scripts**; the script that is responsible for training your final model.

A typical training script:
* Loads training data from a specified directory
* Parses any training & model hyperparameters (ex. nodes in a neural network, training epochs, etc.)
* Instantiates a model of your design, with any specified hyperparams
* Trains that model 
* Finally, saves the model so that it can be hosted/deployed, later

### Defining and training a model
Much of the training script code is provided for you. Almost all of your work will be done in the `if __name__ == '__main__':` section. To complete a `train.py` file, you will:
1. Import any extra libraries you need
2. Define any additional model training hyperparameters using `parser.add_argument`
2. Define a model in the `if __name__ == '__main__':` section
3. Train the model in that same section

Below, you can use `!pygmentize` to display an existing `train.py` file. Read through the code; all of your tasks are marked with `TODO` comments. 

**Note: If you choose to create a custom PyTorch model, you will be responsible for defining the model in the `model.py` file,** and a `predict.py` file is provided. If you choose to use Scikit-learn, you only need a `train.py` file; you may import a classifier from the `sklearn` library.


```python
# directory can be changed to: source_sklearn or source_pytorch
!pygmentize source_sklearn/train.py
```

    [34mfrom[39;49;00m [04m[36m__future__[39;49;00m [34mimport[39;49;00m print_function
    
    [34mimport[39;49;00m [04m[36margparse[39;49;00m
    [34mimport[39;49;00m [04m[36mos[39;49;00m
    [34mimport[39;49;00m [04m[36mpandas[39;49;00m [34mas[39;49;00m [04m[36mpd[39;49;00m
    
    [34mfrom[39;49;00m [04m[36msklearn.externals[39;49;00m [34mimport[39;49;00m joblib
    
    [37m## DONE: Import any additional libraries you need to define a model[39;49;00m
    [34mfrom[39;49;00m [04m[36msklearn[39;49;00m [34mimport[39;49;00m svm
    
    [37m# Provided model load function[39;49;00m
    [34mdef[39;49;00m [32mmodel_fn[39;49;00m(model_dir):
        [33m"""Load model from the model_dir. This is the same model that is saved[39;49;00m
    [33m    in the main if statement.[39;49;00m
    [33m    """[39;49;00m
        [34mprint[39;49;00m([33m"[39;49;00m[33mLoading model.[39;49;00m[33m"[39;49;00m)
        
        [37m# load using joblib[39;49;00m
        model = joblib.load(os.path.join(model_dir, [33m"[39;49;00m[33mmodel.joblib[39;49;00m[33m"[39;49;00m))
        [34mprint[39;49;00m([33m"[39;49;00m[33mDone loading model.[39;49;00m[33m"[39;49;00m)
        
        [34mreturn[39;49;00m model
    
    
    [37m## DONE: Complete the main code[39;49;00m
    [34mif[39;49;00m [31m__name__[39;49;00m == [33m'[39;49;00m[33m__main__[39;49;00m[33m'[39;49;00m:
        
        [37m# All of the model parameters and training parameters are sent as arguments[39;49;00m
        [37m# when this script is executed, during a training job[39;49;00m
        
        [37m# Here we set up an argument parser to easily access the parameters[39;49;00m
        parser = argparse.ArgumentParser()
    
        [37m# SageMaker parameters, like the directories for training data and saving models; set automatically[39;49;00m
        [37m# Do not need to change[39;49;00m
        parser.add_argument([33m'[39;49;00m[33m--output-data-dir[39;49;00m[33m'[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m'[39;49;00m[33mSM_OUTPUT_DATA_DIR[39;49;00m[33m'[39;49;00m])
        parser.add_argument([33m'[39;49;00m[33m--model-dir[39;49;00m[33m'[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m'[39;49;00m[33mSM_MODEL_DIR[39;49;00m[33m'[39;49;00m])
        parser.add_argument([33m'[39;49;00m[33m--data-dir[39;49;00m[33m'[39;49;00m, [36mtype[39;49;00m=[36mstr[39;49;00m, default=os.environ[[33m'[39;49;00m[33mSM_CHANNEL_TRAINING[39;49;00m[33m'[39;49;00m])
        
        [37m## DONE: Add any additional arguments that you will need to pass into your model[39;49;00m
        
        [37m# args holds all passed-in arguments[39;49;00m
        args = parser.parse_args()
    
        [37m# Read in csv training file[39;49;00m
        training_dir = args.data_dir
        train_data = pd.read_csv(os.path.join(training_dir, [33m"[39;49;00m[33mtrain.csv[39;49;00m[33m"[39;49;00m), header=[36mNone[39;49;00m, names=[36mNone[39;49;00m)
    
        [37m# Labels are in the first column[39;49;00m
        train_y = train_data.iloc[:,[34m0[39;49;00m]
        train_x = train_data.iloc[:,[34m1[39;49;00m:]
        
        
        [37m## --- Your code here --- ##[39;49;00m
        
    
        [37m## DONE: Define a model [39;49;00m
        model = svm.SVC()
        
        
        [37m## DONE: Train the model[39;49;00m
        model.fit(train_x, train_y)
        
        
        [37m## --- End of your code  --- ##[39;49;00m
        
    
        [37m# Save the trained model[39;49;00m
        joblib.dump(model, os.path.join(args.model_dir, [33m"[39;49;00m[33mmodel.joblib[39;49;00m[33m"[39;49;00m))


### Provided code

If you read the code above, you can see that the starter code includes a few things:
* Model loading (`model_fn`) and saving code
* Getting SageMaker's default hyperparameters
* Loading the training data by name, `train.csv` and extracting the features and labels, `train_x`, and `train_y`

If you'd like to read more about model saving with [joblib for sklearn](https://scikit-learn.org/stable/modules/model_persistence.html) or with [torch.save](https://pytorch.org/tutorials/beginner/saving_loading_models.html), click on the provided links.

---
# Create an Estimator

When a custom model is constructed in SageMaker, an entry point must be specified. This is the Python file which will be executed when the model is trained; the `train.py` function you specified above. To run a custom training script in SageMaker, construct an estimator, and fill in the appropriate constructor arguments:

* **entry_point**: The path to the Python script SageMaker runs for training and prediction.
* **source_dir**: The path to the training script directory `source_sklearn` OR `source_pytorch`.
* **role**: Role ARN, which was specified, above.
* **train_instance_count**: The number of training instances (should be left at 1).
* **train_instance_type**: The type of SageMaker instance for training. Note: Because Scikit-learn does not natively support GPU training, Sagemaker Scikit-learn does not currently support training on GPU instance types.
* **sagemaker_session**: The session used to train on Sagemaker.
* **hyperparameters** (optional): A dictionary `{'name':value, ..}` passed to the train function as hyperparameters.

Note: For a PyTorch model, there is another optional argument **framework_version**, which you can set to the latest version of PyTorch, `1.0`.

## EXERCISE: Define a Scikit-learn or PyTorch estimator

To import your desired estimator, use one of the following lines:
```
from sagemaker.sklearn.estimator import SKLearn
```
```
from sagemaker.pytorch import PyTorch
```


```python
# import SKLearn estimator
from sagemaker.sklearn.estimator import SKLearn

# initialize estimator
svm_estimator = SKLearn(entry_point="train.py",
                        source_dir="source_sklearn",
                        role=role,
                        train_instance_count=1,
                        train_instance_type='ml.c4.xlarge')
```

## EXERCISE: Train the estimator

Train your estimator on the training data stored in S3. This should create a training job that you can monitor in your SageMaker console.


```python
%%time

# Train your estimator on S3 training data
svm_estimator.fit({'training': input_data})
```

    2020-07-01 02:51:01 Starting - Starting the training job...
    2020-07-01 02:51:05 Starting - Launching requested ML instances.........
    2020-07-01 02:52:50 Starting - Preparing the instances for training......
    2020-07-01 02:53:46 Downloading - Downloading input data...
    2020-07-01 02:54:17 Training - Downloading the training image..[34m2020-07-01 02:54:38,234 sagemaker-containers INFO     Imported framework sagemaker_sklearn_container.training[0m
    [34m2020-07-01 02:54:38,237 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)[0m
    [34m2020-07-01 02:54:38,247 sagemaker_sklearn_container.training INFO     Invoking user training script.[0m
    [34m2020-07-01 02:54:38,556 sagemaker-containers INFO     Module train does not provide a setup.py. [0m
    [34mGenerating setup.py[0m
    [34m2020-07-01 02:54:38,556 sagemaker-containers INFO     Generating setup.cfg[0m
    [34m2020-07-01 02:54:38,556 sagemaker-containers INFO     Generating MANIFEST.in[0m
    [34m2020-07-01 02:54:38,557 sagemaker-containers INFO     Installing module with the following command:[0m
    [34m/miniconda3/bin/python -m pip install . [0m
    [34mProcessing /opt/ml/code[0m
    [34mBuilding wheels for collected packages: train
      Building wheel for train (setup.py): started
      Building wheel for train (setup.py): finished with status 'done'
      Created wheel for train: filename=train-1.0.0-py2.py3-none-any.whl size=5811 sha256=353886fd507ceca54334c389dedf7b103805785ddc6c3a9ace269a9b32894938
      Stored in directory: /tmp/pip-ephem-wheel-cache-j_isd7ak/wheels/35/24/16/37574d11bf9bde50616c67372a334f94fa8356bc7164af8ca3[0m
    [34mSuccessfully built train[0m
    [34mInstalling collected packages: train[0m
    [34mSuccessfully installed train-1.0.0[0m
    [34m2020-07-01 02:54:39,982 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)[0m
    [34m2020-07-01 02:54:39,993 sagemaker-containers INFO     Invoking user script
    [0m
    [34mTraining Env:
    [0m
    [34m{
        "additional_framework_parameters": {},
        "channel_input_dirs": {
            "training": "/opt/ml/input/data/training"
        },
        "current_host": "algo-1",
        "framework_module": "sagemaker_sklearn_container.training:main",
        "hosts": [
            "algo-1"
        ],
        "hyperparameters": {},
        "input_config_dir": "/opt/ml/input/config",
        "input_data_config": {
            "training": {
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            }
        },
        "input_dir": "/opt/ml/input",
        "is_master": true,
        "job_name": "sagemaker-scikit-learn-2020-07-01-02-51-00-708",
        "log_level": 20,
        "master_hostname": "algo-1",
        "model_dir": "/opt/ml/model",
        "module_dir": "s3://sagemaker-us-east-1-578203129999/sagemaker-scikit-learn-2020-07-01-02-51-00-708/source/sourcedir.tar.gz",
        "module_name": "train",
        "network_interface_name": "eth0",
        "num_cpus": 4,
        "num_gpus": 0,
        "output_data_dir": "/opt/ml/output/data",
        "output_dir": "/opt/ml/output",
        "output_intermediate_dir": "/opt/ml/output/intermediate",
        "resource_config": {
            "current_host": "algo-1",
            "hosts": [
                "algo-1"
            ],
            "network_interface_name": "eth0"
        },
        "user_entry_point": "train.py"[0m
    [34m}
    [0m
    [34mEnvironment variables:
    [0m
    [34mSM_HOSTS=["algo-1"][0m
    [34mSM_NETWORK_INTERFACE_NAME=eth0[0m
    [34mSM_HPS={}[0m
    [34mSM_USER_ENTRY_POINT=train.py[0m
    [34mSM_FRAMEWORK_PARAMS={}[0m
    [34mSM_RESOURCE_CONFIG={"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"}[0m
    [34mSM_INPUT_DATA_CONFIG={"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}[0m
    [34mSM_OUTPUT_DATA_DIR=/opt/ml/output/data[0m
    [34mSM_CHANNELS=["training"][0m
    [34mSM_CURRENT_HOST=algo-1[0m
    [34mSM_MODULE_NAME=train[0m
    [34mSM_LOG_LEVEL=20[0m
    [34mSM_FRAMEWORK_MODULE=sagemaker_sklearn_container.training:main[0m
    [34mSM_INPUT_DIR=/opt/ml/input[0m
    [34mSM_INPUT_CONFIG_DIR=/opt/ml/input/config[0m
    [34mSM_OUTPUT_DIR=/opt/ml/output[0m
    [34mSM_NUM_CPUS=4[0m
    [34mSM_NUM_GPUS=0[0m
    [34mSM_MODEL_DIR=/opt/ml/model[0m
    [34mSM_MODULE_DIR=s3://sagemaker-us-east-1-578203129999/sagemaker-scikit-learn-2020-07-01-02-51-00-708/source/sourcedir.tar.gz[0m
    [34mSM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"training":"/opt/ml/input/data/training"},"current_host":"algo-1","framework_module":"sagemaker_sklearn_container.training:main","hosts":["algo-1"],"hyperparameters":{},"input_config_dir":"/opt/ml/input/config","input_data_config":{"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","is_master":true,"job_name":"sagemaker-scikit-learn-2020-07-01-02-51-00-708","log_level":20,"master_hostname":"algo-1","model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-us-east-1-578203129999/sagemaker-scikit-learn-2020-07-01-02-51-00-708/source/sourcedir.tar.gz","module_name":"train","network_interface_name":"eth0","num_cpus":4,"num_gpus":0,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"},"user_entry_point":"train.py"}[0m
    [34mSM_USER_ARGS=[][0m
    [34mSM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate[0m
    [34mSM_CHANNEL_TRAINING=/opt/ml/input/data/training[0m
    [34mPYTHONPATH=/miniconda3/bin:/miniconda3/lib/python37.zip:/miniconda3/lib/python3.7:/miniconda3/lib/python3.7/lib-dynload:/miniconda3/lib/python3.7/site-packages
    [0m
    [34mInvoking script with the following command:
    [0m
    [34m/miniconda3/bin/python -m train
    
    [0m
    [34m/miniconda3/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp[0m
    [34m/miniconda3/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)[0m
    [34m2020-07-01 02:54:41,731 sagemaker-containers INFO     Reporting training SUCCESS[0m
    
    2020-07-01 02:54:49 Uploading - Uploading generated training model
    2020-07-01 02:54:49 Completed - Training job completed
    Training seconds: 63
    Billable seconds: 63
    CPU times: user 568 ms, sys: 14 ms, total: 582 ms
    Wall time: 4min 12s


## EXERCISE: Deploy the trained model

After training, deploy your model to create a `predictor`. If you're using a PyTorch model, you'll need to create a trained `PyTorchModel` that accepts the trained `<model>.model_data` as an input parameter and points to the provided `source_pytorch/predict.py` file as an entry point. 

To deploy a trained model, you'll use `<model>.deploy`, which takes in two arguments:
* **initial_instance_count**: The number of deployed instances (1).
* **instance_type**: The type of SageMaker instance for deployment.

Note: If you run into an instance error, it may be because you chose the wrong training or deployment instance_type. It may help to refer to your previous exercise code to see which types of instances we used.


```python
%%time

# uncomment, if needed
# from sagemaker.pytorch import PyTorchModel

# deploy your model to create a predictor
svm_predictor = svm_estimator.deploy(initial_instance_count = 1, instance_type = 'ml.t2.medium')
```

    ---------------!CPU times: user 240 ms, sys: 24.3 ms, total: 265 ms
    Wall time: 7min 33s


---
# Evaluating Your Model

Once your model is deployed, you can see how it performs when applied to our test data.

The provided cell below, reads in the test data, assuming it is stored locally in `data_dir` and named `test.csv`. The labels and features are extracted from the `.csv` file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import os

# read in test data, assuming it is stored locally
test_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None, names=None)

# labels are in the first column
test_y = test_data.iloc[:,0]
test_x = test_data.iloc[:,1:]
```

## EXERCISE: Determine the accuracy of your model

Use your deployed `predictor` to generate predicted, class labels for the test data. Compare those to the *true* labels, `test_y`, and calculate the accuracy as a value between 0 and 1.0 that indicates the fraction of test data that your model classified correctly. You may use [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) for this calculation.

**To pass this project, your model should get at least 90% test accuracy.**


```python
# First: generate predicted, class labels
test_y_preds = svm_predictor.predict(test_x)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# test that your model generates the correct number of labels
assert len(test_y_preds)==len(test_y), 'Unexpected number of predictions.'
print('Test passed!')
```

    Test passed!



```python
from sklearn.metrics import accuracy_score

# Second: calculate the test accuracy
accuracy = accuracy_score(test_y, test_y_preds)
print(accuracy)


## print out the array of predicted and true labels, if you want
print('\nPredicted class labels: ')
print(test_y_preds)
print('\nTrue class labels: ')
print(test_y.values)
```

    0.96
    
    Predicted class labels: 
    [1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 0 1 0 1 1 0 0]
    
    True class labels: 
    [1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 0]


### Question 1: How many false positives and false negatives did your model produce, if any? And why do you think this is?

**Answer**: The model produced zero false positives and one false negative, resulting in an accuracy score of 96%. This high score is likely due to a small dataset and clear distinctions between positive and negative plagiarism samples.

### Question 2: How did you decide on the type of model to use? 

**Answer**: I decided to use a Support Vector Classification (SVC) model because support vector machines are good fits for classification problems. Since the size of the dataset is much smaller than 10,000 samples, a LinearSVC is not necessary to optimize training time.

----
## EXERCISE: Clean up Resources

After you're done evaluating your model, **delete your model endpoint**. You can do this with a call to `.delete_endpoint()`. You need to show, in this notebook, that the endpoint was deleted. Any other resources, you may delete from the AWS console, and you will find more instructions on cleaning up all your resources, below.


```python
# delete endpoint
svm_predictor.delete_endpoint()
```

### Deleting S3 bucket

When you are *completely* done with training and testing models, you can also delete your entire S3 bucket. If you do this before you are done training your model, you'll have to recreate your S3 bucket and upload your training data again.


```python
# delete bucket
bucket_to_delete = boto3.resource('s3').Bucket(bucket)
bucket_to_delete.objects.all().delete()
```




    [{'ResponseMetadata': {'RequestId': '76C5CF2CB4F6151B',
       'HostId': '1dkeDFZQ3gixWWWehK1ichk/HvKM9J5wa8uJbeSO1ZaGlpFADz3E/ePWHsEpjoXH83w0F/huaw4=',
       'HTTPStatusCode': 200,
       'HTTPHeaders': {'x-amz-id-2': '1dkeDFZQ3gixWWWehK1ichk/HvKM9J5wa8uJbeSO1ZaGlpFADz3E/ePWHsEpjoXH83w0F/huaw4=',
        'x-amz-request-id': '76C5CF2CB4F6151B',
        'date': 'Wed, 01 Jul 2020 03:16:45 GMT',
        'connection': 'close',
        'content-type': 'application/xml',
        'transfer-encoding': 'chunked',
        'server': 'AmazonS3'},
       'RetryAttempts': 0},
      'Deleted': [{'Key': 'sagemaker-pytorch-2020-04-21-01-38-44-484/debug-output/training_job_end.ts'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-02-51-00-708/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-20-02-46-31-199/output/model.tar.gz'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-02-32-47-817/debug-output/training_job_end.ts'},
       {'Key': 'sagemaker-record-sets/LinearLearner-2020-05-30-20-56-48-927/matrix_0.pbr'},
       {'Key': 'creditcard/linear-learner-2020-05-30-20-56-57-613/output/model.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-20-01-05-57-860/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-20-16-28-53-211/debug-output/training_job_end.ts'},
       {'Key': 'sagemaker-pytorch-2020-04-21-02-15-07-693/source/sourcedir.tar.gz'},
       {'Key': 'counties/kmeans-2020-04-30-00-14-17-609/output/model.tar.gz'},
       {'Key': 'sagemaker/sentiment_rnn/train.csv'},
       {'Key': 'sagemaker-pytorch-2020-04-20-16-28-53-211/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-20-22-23-22-990/sourcedir.tar.gz'},
       {'Key': 'sagemaker-record-sets/KMeans-2020-04-29-23-28-42-924/matrix_0.pbr'},
       {'Key': 'creditcard/linear-learner-2020-05-30-21-58-33-160/output/model.tar.gz'},
       {'Key': 'counties/pca-2020-04-29-23-49-22-353/output/model.tar.gz'},
       {'Key': 'creditcard/linear-learner-2020-05-30-22-13-51-669/output/model.tar.gz'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-02-51-00-708/output/model.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-21-02-15-07-693/output/model.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-20-16-28-53-211/output/model.tar.gz'},
       {'Key': 'sagemaker-record-sets/PCA-2020-04-29-18-42-18-262/.amazon.manifest'},
       {'Key': 'sagemaker-pytorch-2020-04-20-02-46-31-199/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-21-01-38-44-484/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-record-sets/KMeans-2020-04-30-00-14-16-993/.amazon.manifest'},
       {'Key': 'sagemaker-pytorch-2020-04-20-06-19-34-575/sourcedir.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-20-16-29-22-443/source/sourcedir.tar.gz'},
       {'Key': 'creditcard/linear-learner-2020-05-29-05-18-05-737/output/model.tar.gz'},
       {'Key': 'sagemaker-record-sets/PCA-2020-04-29-23-49-12-401/.amazon.manifest'},
       {'Key': 'sagemaker-pytorch-2020-04-21-02-15-07-693/debug-output/training_job_end.ts'},
       {'Key': 'sagemaker-record-sets/LinearLearner-2020-05-30-20-56-48-927/.amazon.manifest'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-02-51-00-708/debug-output/training_job_end.ts'},
       {'Key': 'creditcard/linear-learner-2020-05-30-21-30-38-043/output/model.tar.gz'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-01-55-13-973/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-02-32-47-817/output/model.tar.gz'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-02-33-02-389/debug-output/training_job_end.ts'},
       {'Key': 'sagemaker-pytorch-2020-04-20-01-07-18-470/source/sourcedir.tar.gz'},
       {'Key': 'creditcard/linear-learner-2020-05-30-20-55-34-624/output/model.tar.gz'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-02-13-13-175/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-record-sets/KMeans-2020-04-30-00-14-16-993/matrix_0.pbr'},
       {'Key': 'sagemaker-record-sets/LinearLearner-2020-05-29-05-17-35-036/.amazon.manifest'},
       {'Key': 'counties/pca-2020-04-29-18-42-33-431/output/model.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-20-01-05-17-719/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-record-sets/LinearLearner-2020-05-30-21-30-29-199/.amazon.manifest'},
       {'Key': 'sagemaker-pytorch-2020-04-20-16-28-35-484/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-21-01-38-44-484/output/model.tar.gz'},
       {'Key': 'counties/kmeans-2020-04-29-23-29-12-779/output/model.tar.gz'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-02-19-23-233/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-record-sets/LinearLearner-2020-05-30-21-30-29-199/matrix_0.pbr'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-01-44-22-842/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-20-06-49-24-672/sourcedir.tar.gz'},
       {'Key': 'sagemaker-record-sets/PCA-2020-04-30-00-02-37-443/.amazon.manifest'},
       {'Key': 'sagemaker-record-sets/PCA-2020-04-29-18-42-18-262/matrix_0.pbr'},
       {'Key': 'sagemaker-pytorch-2020-04-20-16-29-22-443/output/model.tar.gz'},
       {'Key': 'creditcard/linear-learner-2020-05-30-21-11-54-961/output/model.tar.gz'},
       {'Key': 'plagiarism_data/test.csv'},
       {'Key': 'sagemaker-record-sets/LinearLearner-2020-05-30-20-55-25-809/.amazon.manifest'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-02-33-02-389/output/model.tar.gz'},
       {'Key': 'sagemaker-record-sets/LinearLearner-2020-05-30-20-55-25-809/matrix_0.pbr'},
       {'Key': 'sagemaker-pytorch-2020-04-20-21-43-14-509/sourcedir.tar.gz'},
       {'Key': 'sagemaker-record-sets/LinearLearner-2020-05-29-05-17-35-036/matrix_0.pbr'},
       {'Key': 'sagemaker-record-sets/PCA-2020-04-29-23-49-12-401/matrix_0.pbr'},
       {'Key': 'sagemaker-record-sets/KMeans-2020-04-29-23-28-42-924/.amazon.manifest'},
       {'Key': 'sagemaker-pytorch-2020-04-20-18-45-57-686/sourcedir.tar.gz'},
       {'Key': 'counties/pca-2020-04-30-00-02-39-867/output/model.tar.gz'},
       {'Key': 'sagemaker-record-sets/PCA-2020-04-30-00-02-37-443/matrix_0.pbr'},
       {'Key': 'sagemaker-pytorch-2020-04-20-02-46-31-199/debug-output/training_job_end.ts'},
       {'Key': 'creditcard/linear-learner-2020-05-29-05-34-23-202/output/model.tar.gz'},
       {'Key': 'sagemaker-pytorch-2020-04-20-16-29-22-443/debug-output/training_job_end.ts'},
       {'Key': 'sagemaker-pytorch-2020-04-21-01-56-43-576/sourcedir.tar.gz'},
       {'Key': 'sagemaker/sentiment_rnn/word_dict.pkl'},
       {'Key': 'creditcard/linear-learner-2020-05-30-21-43-35-286/output/model.tar.gz'},
       {'Key': 'plagiarism_data/train.csv'},
       {'Key': 'sagemaker-pytorch-2020-04-21-02-33-39-086/sourcedir.tar.gz'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-02-33-02-389/source/sourcedir.tar.gz'},
       {'Key': 'sagemaker-scikit-learn-2020-07-01-02-32-47-817/source/sourcedir.tar.gz'}]}]



### Deleting all your models and instances

When you are _completely_ done with this project and do **not** ever want to revisit this notebook, you can choose to delete all of your SageMaker notebook instances and models by following [these instructions](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html). Before you delete this notebook instance, I recommend at least downloading a copy and saving it, locally.

---
## Further Directions

There are many ways to improve or add on to this project to expand your learning or make this more of a unique project for you. A few ideas are listed below:
* Train a classifier to predict the *category* (1-3) of plagiarism and not just plagiarized (1) or not (0).
* Utilize a different and larger dataset to see if this model can be extended to other types of plagiarism.
* Use language or character-level analysis to find different (and more) similarity features.
* Write a complete pipeline function that accepts a source text and submitted text file, and classifies the submitted text as plagiarized or not.
* Use API Gateway and a lambda function to deploy your model to a web application.

These are all just options for extending your work. If you've completed all the exercises in this notebook, you've completed a real-world application, and can proceed to submit your project. Great job!
