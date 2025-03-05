
import base64
import logging
import io
import os
from typing import Dict
import json
import requests
from PIL import Image as PILImage
from app.rag.chat_config import get_default_llm

# Set up basic logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



# Default prompts for image analysis
DEFAULT_TEXT_EXTRACTION_PROMPT = """
You are an expert in extracting text from images. 
Analyze the following image and extract ALL text content visible in it.
If there are multiple columns, tables, or complex layouts, preserve the structure as much as possible.
If you cannot see any text in the image, respond with 'No text detected'.
"""

DEFAULT_IMAGE_DESCRIPTION_PROMPT = """
Provide a detailed description of this image. Include:
1. The main subject or focus of the image
2. Any visible text content (transcribe it accurately)
3. Visual elements like charts, graphs, diagrams (describe their purpose and content)
4. The overall context and purpose of the image
5. Any notable details that would be important for understanding the image

Be comprehensive but concise.
"""


class GPT4OVisionLLM:
    """
    A simple wrapper that calls OpenAI's GPT-4O vision endpoint directly.
    
    It encodes the image as base64 and sends a single chat completion request with a text prompt and the image.
    """
    
    def __init__(self, api_key: str = None, api_url: str = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.api_url = api_url or os.getenv("LLM_API_URL")
        if not self.api_key:
            raise ValueError("LLM API key must be provided")
        if not self.api_url:
            raise ValueError("LLM API URL must be provided")

    def call_gpt4o(self, prompt: str, image_data: bytes) -> str:
        """
        Encodes the image and sends a chat completion request to OpenAI's GPT-4O.
        
        Args:
            prompt: The text prompt for the model.
            image_data: The raw image bytes.
        
        Returns:
            The response from the model as a string.
        """
        try:
            img = PILImage.open(io.BytesIO(image_data))
            image_format = img.format.lower() if img.format else "jpeg"
        except Exception as e:
            logger.error(f"Error opening image: {e}")
            return "Invalid image data"

        # Encode the image as base64 and build a data URL
        encoded = base64.b64encode(image_data).decode("utf-8")
        data_url = f"data:image/{image_format};base64,{encoded}"

        # Build the messages payload.
        payload_dict = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                }
            ]
        }
        payload = json.dumps(payload_dict)
        
        url = self.api_url
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            response.raise_for_status()
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error during LLM API call: {e}")
            return "Error processing the image with the LLM"


class LLMImageAnalyzer:
    """
    Class for analyzing images using GPT-4O vision capabilities via OpenAI's API.
    
    It leverages GPT-4O to extract text and generate a description by passing the prompt and image together.
    """
    
    def __init__(
        self, 
        llm,  # Accept any LLM type
        text_extraction_prompt: str = DEFAULT_TEXT_EXTRACTION_PROMPT,
        image_description_prompt: str = DEFAULT_IMAGE_DESCRIPTION_PROMPT
    ):
        """
        Initialize the analyzer with an LLM that calls GPT-4O.
        
        Args:
            llm: An instance of GPT4OVisionLLM
            text_extraction_prompt: Prompt for extracting text from images.
            image_description_prompt: Prompt for generating image descriptions.
        """
        api_key = os.getenv("LLM_API_KEY")
        api_url = os.getenv("LLM_API_URL")
        if not api_key:
            raise ValueError("LLM API key must be provided")
        if not api_url:
            raise ValueError("LLM API URL must be provided")
        self._llm = GPT4OVisionLLM(api_key, api_url)
        
        self._text_extraction_prompt = text_extraction_prompt
        self._image_description_prompt = image_description_prompt
    
    def extract_text_from_image(self, image_data: bytes) -> str:
        """
        Extract text from an image by calling GPT-4O.
        
        Args:
            image_data: Raw image data bytes.
            
        Returns:
            Extracted text from the image.
        """
        try:
            return self._llm.call_gpt4o(self._text_extraction_prompt, image_data)
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return "Error extracting text from image"
    
    def generate_image_description(self, image_data: bytes) -> str:
        """
        Generate a description of an image by calling GPT-4O.
        
        Args:
            image_data: Raw image data bytes.
            
        Returns:
            A detailed description of the image.
        """
        try:
            return self._llm.call_gpt4o(self._image_description_prompt, image_data)
        except Exception as e:
            logger.error(f"Error extracting description from image: {e}")
            return "Error extracting description from image"
    
    def analyze_image(self, image_data: bytes) -> Dict[str, str]:
        """
        Perform comprehensive analysis of an image.
        
        Args:
            image_data: Raw image data bytes.
            
        Returns:
            A dictionary containing the extracted text and description.
        """
        text_content = self.extract_text_from_image(image_data)
        description = self.generate_image_description(image_data)
        return {
            "text_content": text_content,
            "description": description
        }


