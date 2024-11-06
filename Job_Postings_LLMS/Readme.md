# Description 
This series of notebooks uses a scrapped data set of Job posting data that contained `job title, job description, raw location, and company`. This data is highly unstructed and provided the perfect test bed to expierment with LLMs performing different tasks. These models were trained on my personal computer using Linux to enable GPU usage on my Nvidia 2080 super GPU.


## Job Type Binary 
[Production Test Notebook](./Model_trainer_job_type.ipynb)

This Notebook is the first iteration of of using distilledBert LLM to take in Job Title and Job Description data to classify the job posting by by job industry. In this case it was a binary classification classifying the job as tech or not tech. 


## Job Type Clustering 
.... Need to upload the file 

This notebook expiermented with unsupervised clustering methods using distilledBert LLM to label data. This did not provide great results but did provide useful in the hand labeling process as it did provide labels that were "mostly" correct.

## Job Type Multi Classifcation 
[Production Test Notebook](./Model_trainer_job_type_multi.ipynb)

This notebook is used to train classification models to identify job industry from a given job title and job description from job posting data. Those being `technology, medical, marketing and sales, law, service industry, retail, education, customer service, engineering and architecture, skilled trades, hr, finance`.


## Job Type Multi Classifcation Productionalization 
[Production Test Notebook](./Productionalize_job_type.ipynb)

This notebook is used to take the multi classification job type pytorch model to onnx model and then to optomized tensorrt model.

## Location NER Model
[Production Test Notebook](./Model_trainer_NER_location.ipynb)

This notebook is used to train a location NER model to identify 4 entities in the raw location string. Those being `country, state, city, and remote`. 


## Location NER Model Productionalization 
[Production Test Notebook](./Productionalize_location.ipynb)

This notebook is used to take the location NER pytorch model to onnx model and then to optomized tensorrt model.

## Production Test Of both models. 
[Production Test Notebook](./Prod_Test.ipynb)

This notebook was used as a test bed to build out production code that used the tensorrt models to make predictions. This notebook was later turn into python files. 



# Setup Steps for Tensorrt 
1. run `sudo apt install tensorrt`

1. run `sudo apt install libnvinfer-dev libnvinfer-plugin-dev`


Then to test run:
- `trtexec --help`

If that doesn't work then test with:
- `/usr/src/tensorrt/bin/trtexec --help`

If that works then run 
- `export PATH=$PATH:/usr/src/tensorrt/bin`

Then to Double check its working. 
- `trtexec --help`

Potential error solutions:
- run `pip install pycuda`
- run `conda install -c conda-forge libstdcxx-ng`
- Must make sure that the pip version matches the version of trtexec used to create the .trt file 


## Running trtexec
- Setting precision lower to save memory
```bash
# Lower precision less memory 
trtexec --onnx=industry_classifier.onnx --saveEngine=location_classifier.trt --fp16

# Lower precision with batch size 1 or 2
trtexec --onnx=industry_classifier.onnx --saveEngine=location_classifier_v2.trt --fp16 \
        --minShapes=input_ids:1x512,attention_mask:1x512 \ 
        --optShapes=input_ids:2x512,attention_mask:2x512 \
        --maxShapes=input_ids:2x512,attention_mask:2x512

# Batch size 1 or 2 
trtexec --onnx=industry_classifier.onnx --saveEngine=location_classifier_v3.trt \
        --minShapes=input_ids:1x512,attention_mask:1x512 \
        --optShapes=input_ids:2x512,attention_mask:2x512 \
        --maxShapes=input_ids:2x512,attention_mask:2x512
```


- Explictly setting shapes: 
```bash
# batch size 16
trtexec --onnx=location_classifier.onnx --saveEngine=location_classifier_2.trt \
        --minShapes=input_ids:16x256,attention_mask:16x256 \
        --optShapes=input_ids:16x256,attention_mask:16x256 \
        --maxShapes=input_ids:16x256,attention_mask:16x256

# Batch size 1-2-16 
trtexec --onnx=location_classifier.onnx --saveEngine=location_classifier_3.trt \
        --minShapes=input_ids:1x256,attention_mask:1x256 \
        --optShapes=input_ids:2x256,attention_mask:2x256 \
        --maxShapes=input_ids:16x256,attention_mask:16x256
```

## pycuda docs 10.6.0
- https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html
