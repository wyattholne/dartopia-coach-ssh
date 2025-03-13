import boto3
import os
from io import BytesIO
import zipfile
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def unzip_s3_dataset(bucket_name='dartopia-coach-connect', 
                     zip_key='Darts.v2i.yolov11.zip',
                     region='eu-west-2'):
    """Unzip a dataset directly from S3 to S3."""
    s3_client = boto3.client('s3', region_name=region)
    
    logger.info(f"Starting to process {zip_key} from {bucket_name}")
    
    try:
        # Download the zip file into memory
        zip_obj = s3_client.get_object(Bucket=bucket_name, Key=zip_key)
        buffer = BytesIO(zip_obj['Body'].read())
        
        # Open the zip file
        z = zipfile.ZipFile(buffer)
        
        # Process each file in the zip
        for filename in z.namelist():
            logger.info(f"Extracting: {filename}")
            
            # Read the file content
            file_content = z.read(filename)
            
            # Upload the file to S3
            target_key = f"Darts.v2i.yolov11/{filename}"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=target_key,
                Body=file_content
            )
            
            # Validate image files
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = Image.open(BytesIO(file_content))
                    img.verify()  # Check if the image is valid
                    logger.info(f"Verified image integrity for {filename}")
                except Exception as e:
                    logger.error(f"Invalid image {filename}: {str(e)}")
            
            # Validate label files
            if filename.lower().endswith('.txt'):
                try:
                    label_content = file_content.decode('utf-8').strip().splitlines()
                    for line in label_content:
                        values = line.strip().split()
                        if len(values) != 5 or not all(float(v) >= 0 for v in values[1:]):
                            raise ValueError(f"Invalid label format in {filename}: {line}")
                    logger.info(f"Verified label format for {filename}")
                except Exception as e:
                    logger.error(f"Label validation failed for {filename}: {str(e)}")

        logger.info("Extraction complete!")
        
        # Optionally delete the zip file after extraction
        # s3_client.delete_object(Bucket=bucket_name, Key=zip_key)
        
    except Exception as e:
        logger.error(f"Error during unzipping: {str(e)}")

def verify_dataset_structure():
    """Verify the expected YOLO dataset structure in S3."""
    s3_client = boto3.client('s3', region_name='eu-west-2')
    bucket_name = 'dartopia-coach-connect'
    prefix = 'Darts.v2i.yolov11'
    
    expected_dirs = [
        'train/images',
        'train/labels',
        'valid/images',
        'valid/labels',
        'test/images',
        'test/labels'
    ]
    
    logger.info("\nVerifying dataset structure:")
    for dir_path in expected_dirs:
        full_path = f"{prefix}/{dir_path}/"
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=full_path,
            MaxKeys=1
        )
        
        if 'Contents' in response:
            logger.info(f"✓ Found {dir_path}")
        else:
            logger.warning(f"✗ Missing {dir_path}")

def list_dataset_stats():
    """List statistics about the dataset in S3."""
    s3_client = boto3.client('s3', region_name='eu-west-2')
    bucket_name = 'dartopia-coach-connect'
    prefix = 'Darts.v2i.yolov11'
    
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix
    )
    
    if 'Contents' in response:
        total_files = len(response['Contents'])
        total_size = sum(obj['Size'] for obj in response['Contents'])
        
        logger.info(f"\nDataset Statistics:")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Total size: {total_size / (1024*1024):.2f} MB")
        
        # Count files by type
        image_count = sum(1 for obj in response['Contents'] if obj['Key'].endswith(('.jpg', '.jpeg', '.png')))
        label_count = sum(1 for obj in response['Contents'] if obj['Key'].endswith('.txt'))
        
        logger.info(f"Image files: {image_count}")
        logger.info(f"Label files: {label_count}")
    else:
        logger.warning("No files found in the specified prefix")

if __name__ == "__main__":
    # Install required packages if not present
    try:
        import PIL
    except ImportError:
        os.system("pip install Pillow")
    
    # First, unzip the dataset
    unzip_s3_dataset()
    
    # Verify the structure
    verify_dataset_structure()
    
    # Show statistics
    list_dataset_stats()