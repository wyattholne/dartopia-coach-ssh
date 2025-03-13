import boto3
import os
from io import BytesIO
import zipfile
from PIL import Image
import logging
from pathlib import Path

# Set up logging (like a diary to tell us what's happening)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_aws_connection(bucket_name='dartopia-coach-connect'):
    """Make sure we can talk to your toy box (S3)."""
    try:
        s3_client = boto3.client('s3', region_name='eu-west-2')
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='Darts.v2i.yolov11.zip'
        )
        
        if 'Contents' in response:
            for obj in response['Contents']:
                logger.info(f"Found: {obj['Key']}")
                logger.info(f"Size: {obj['Size'] / (1024*1024):.2f} MB")
                logger.info(f"Last modified: {obj['LastModified']}")
        
        logger.info("\nAWS Connection Successful!")
        return True
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False

def unzip_s3_dataset(bucket_name='dartopia-coach-connect', 
                     zip_key='Darts.v2i.yolov11.zip',
                     region='eu-west-2'):
    """Open the toy box and take out all the toys (unzip the file)."""
    s3_client = boto3.client('s3', region_name=region)
    
    logger.info(f"Starting to open {zip_key} from {bucket_name}")
    
    try:
        # Get the zip file (like picking up the toy box)
        zip_obj = s3_client.get_object(Bucket=bucket_name, Key=zip_key)
        buffer = BytesIO(zip_obj['Body'].read())
        
        # Open the zip file
        z = zipfile.ZipFile(buffer)
        
        # Take out each toy (file) and put it in the right spot
        for filename in z.namelist():
            logger.info(f"Taking out: {filename}")
            
            # Read the toy (file)
            file_content = z.read(filename)
            
            # Put the toy in a new spot (upload to S3)
            target_key = f"Darts.v2i.yolov11/{filename}"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=target_key,
                Body=file_content
            )
            
            # Check if the toy is a picture and make sure it’s not broken
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = Image.open(BytesIO(file_content))
                    img.verify()  # Make sure the picture is okay
                    logger.info(f"Picture looks good: {filename}")
                except Exception as e:
                    logger.error(f"Broken picture {filename}: {str(e)}")
            
            # Check if the toy is a label and make sure it’s right
            if filename.lower().endswith('.txt'):
                try:
                    label_content = file_content.decode('utf-8').strip().splitlines()
                    for line in label_content:
                        values = line.strip().split()
                        if len(values) != 5 or not all(float(v) >= 0 for v in values[1:]):
                            raise ValueError(f"Wrong label format in {filename}: {line}")
                    logger.info(f"Label looks good: {filename}")
                except Exception as e:
                    logger.error(f"Label problem {filename}: {str(e)}")

        logger.info("All done opening the toy box!")
        
    except Exception as e:
        logger.error(f"Oops, something went wrong: {str(e)}")

def verify_dataset_structure():
    """Make sure all the toy boxes are in the right spots."""
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
    
    logger.info("\nChecking if all toy boxes are there:")
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
    """Count how many toys we have and how big they are."""
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
        
        logger.info(f"\nToy Box Stats:")
        logger.info(f"Total toys: {total_files}")
        logger.info(f"Total size: {total_size / (1024*1024):.2f} MB")
        
        # Count pictures and labels
        image_count = sum(1 for obj in response['Contents'] if obj['Key'].endswith(('.jpg', '.jpeg', '.png')))
        label_count = sum(1 for obj in response['Contents'] if obj['Key'].endswith('.txt'))
        
        logger.info(f"Pictures: {image_count}")
        logger.info(f"Labels: {label_count}")
    else:
        logger.warning("No toys found in the toy box!")

if __name__ == "__main__":
    # Make sure we have the right tools to play
    try:
        import PIL
    except ImportError:
        os.system("pip install Pillow")
    
    # Check if we can talk to the toy box
    if check_aws_connection():
        # Open the toy box and take out the toys
        unzip_s3_dataset()
        
        # Make sure all the toy spots are there
        verify_dataset_structure()
        
        # Count all the toys
        list_dataset_stats()