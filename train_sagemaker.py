import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import logging
import time
import json
import os

# Set up logging with more detail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_s3_access():
    """Verify S3 access and bucket contents"""
    try:
        s3 = boto3.client('s3')
        # Check training data
        logger.info("Checking training data...")
        response = s3.list_objects_v2(
            Bucket='dartopia-coach-connect',
            Prefix='Darts.v2i.yolov11/'
        )
        if 'Contents' in response:
            logger.info(f"Found {len(response['Contents'])} objects in training data")
        else:
            logger.error("No training data found!")
            return False

        # Check model file
        logger.info("Checking model file...")
        try:
            s3.head_object(
                Bucket='dartopia-coach-connect',
                Key='Darts.v2i.yolov11/weights/best.pt'  # Corrected Key
            )
            logger.info("Model file exists")
        except Exception as e:
            logger.error(f"Model file not found: {str(e)}")
            return False
        
        # Verify data.yaml exists
        try:
            s3.head_object(
                Bucket='dartopia-coach-connect',
                Key='Darts.v2i.yolov11/data.yaml'
            )
            logger.info("data.yaml exists")
        except Exception as e:
            logger.error(f"data.yaml not found: {str(e)}")
            return False
        
        return True

    except Exception as e:
        logger.error(f"S3 access error: {str(e)}")
        return False

def main():
    try:
        # Verify S3 access first
        if not verify_s3_access():
            logger.error("S3 verification failed. Exiting.")
            return

        # Set up SageMaker session with explicit region
        region = 'eu-west-1'
        boto_session = boto3.Session(region_name=region)
        sagemaker_session = sagemaker.Session(boto_session=boto_session)
        role = "arn:aws:iam::650251732447:role/service-role/AmazonSageMaker-ExecutionRole-20250310T001863"

        logger.info("Creating PyTorch estimator...")
        
        # Define a unique job name
        job_name = f"pytorch-training-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        logger.info(f"Job name: {job_name}")

        pytorch_estimator = PyTorch(
            entry_point="train.py",
            source_dir=".",
            dependencies=['requirements.txt'],
            role=role,
            instance_count=1,
            instance_type="ml.g4dn.xlarge", # Changed to GPU instance
            framework_version="1.8.1",
            py_version="py3",
            hyperparameters={
                "model": "/opt/ml/input/data/model/best.pt",
                "data": "/opt/ml/input/data/training/data.yaml",
                "epochs": 50,
                "device": 0, # Will be overwritten in train.py
                "lr0": 0.01,
            },
            input_mode='FastFile',
            environment={
                'NCCL_DEBUG': 'INFO',
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
                'NVIDIA_VISIBLE_DEVICES': 'all'
            },
            metric_definitions=[
                {'Name': 'train:loss', 'Regex': 'Epoch [0-9]+ Train Loss: ([0-9.]+)'},
                {'Name': 'val:loss', 'Regex': 'Epoch [0-9]+ Val Loss: ([0-9.]+)'}
            ],
            volume_size=30,
            sagemaker_session=sagemaker_session,
            max_run=7200,  # Increased to 2 hours
            output_path=f"s3://dartopia-coach-connect/training-output",
            job_name=job_name,
            debugger_hook_config=False,
            container_log_level=20,
            max_retry_attempts=1
        )

        # Start training with explicit error handling
        logger.info("Starting training job...")
        
        # Define input channels
        input_channels = {
            "training": "s3://dartopia-coach-connect/Darts.v2i.yolov11/",
            "model": "s3://dartopia-coach-connect/Darts.v2i.yolov11/weights/" # Corrected model path
        }
        
        logger.info(f"Input channels: {json.dumps(input_channels, indent=2)}")
        
        # Start the training job
        pytorch_estimator.fit(
            inputs=input_channels,
            wait=True,
            logs="All" # Removed log_level
        )

        logger.info(f"Training job {job_name} completed successfully")

    except sagemaker.exceptions.UnexpectedStatusException as e:
        logger.error(f"Training failed with unexpected status: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
