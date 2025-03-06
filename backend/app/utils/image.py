import base64
from pathlib import Path

def image_base64_url(image_path: str) -> str:
    """
    Convert an image file to a base64 URL format suitable for embedding.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 URL representation of the image
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
    # Determine MIME type based on file extension
    mime_type = "image/jpeg"  # Default MIME type
    if image_path.suffix.lower() in [".png"]:
        mime_type = "image/png"
    elif image_path.suffix.lower() in [".gif"]:
        mime_type = "image/gif"
    elif image_path.suffix.lower() in [".webp"]:
        mime_type = "image/webp"
    
    return f"data:{mime_type};base64,{encoded_string}"