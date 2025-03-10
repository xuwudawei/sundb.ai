import os
import boto3
import logging
import time
from typing import Optional, Callable, Any
from functools import wraps
from botocore.exceptions import ClientError, ConnectionError, ConnectTimeoutError

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries: int = 3, initial_backoff: float = 1.0, backoff_factor: float = 2.0):
    """Decorator that retries the function with exponential backoff when specific exceptions occur
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor by which to multiply backoff time after each failure
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            backoff = initial_backoff
            
            while True:
                try:
                    return func(*args, **kwargs)
                except (ClientError, ConnectionError, ConnectTimeoutError) as e:
                    # Check if we've exceeded max retries
                    if retries >= max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    # Check if this is a retryable error
                    error_code = getattr(e, 'response', {}).get('Error', {}).get('Code', '')
                    if isinstance(e, (ConnectionError, ConnectTimeoutError)) or error_code in [
                        'InternalError', 'ServiceUnavailable', 'ThrottlingException',
                        'RequestTimeout', 'RequestTimeTooSkewed', 'SlowDown'
                    ]:
                        retries += 1
                        wait_time = backoff * (backoff_factor ** (retries - 1))
                        logger.warning(f"Retrying {func.__name__} after error: {str(e)}. "
                                      f"Retry {retries}/{max_retries} in {wait_time:.2f}s")
                        time.sleep(wait_time)
                    else:
                        # Non-retryable error
                        logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                        raise
        return wrapper
    return decorator


class S3ImageStorage:
    """Handles image storage and URL generation using AWS S3"""

    def __init__(
        self,
        bucket_name: str = None,
        region_name: str = 'us-east-1',
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        self.bucket_name = bucket_name or os.getenv('AWS_S3_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError('S3 bucket name must be provided')

        self.s3_client = boto3.client(
            's3',
            region_name=region_name,
            aws_access_key_id=aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        )

    @retry_with_backoff(max_retries=3, initial_backoff=1.0, backoff_factor=2.0)
    def upload_image(self, image_data: bytes, file_name: str) -> str:
        """Upload an image to S3 and return its public URL

        Args:
            image_data: Raw image bytes to upload
            file_name: Name to give the file in S3

        Returns:
            str: Public URL of the uploaded image
        """
        try:
            # Upload the image
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_name,
                Body=image_data,
                ContentType='image/jpeg'  # Adjust based on actual image type if needed
            )

            # Generate the public URL
            url = f'https://{self.bucket_name}.s3.amazonaws.com/{file_name}'
            logger.info(f'Successfully uploaded image to {url}')
            return url

        except ClientError as e:
            logger.error(f'Error uploading image to S3: {str(e)}')
            raise

    @retry_with_backoff(max_retries=3, initial_backoff=1.0, backoff_factor=2.0)
    def delete_image(self, file_name: str) -> bool:
        """Delete an image from S3

        Args:
            file_name: Name of the file to delete

        Returns:
            bool: True if deletion was successful
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_name
            )
            logger.info(f'Successfully deleted image {file_name} from S3')
            return True

        except ClientError as e:
            logger.error(f'Error deleting image from S3: {str(e)}')
            return False