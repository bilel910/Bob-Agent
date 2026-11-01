import base64
import logging
import os
from typing import Optional, Union

from together import Together

from ai_companion.core.exceptions import ImageToTextError
from ai_companion.core.prompts import IMAGE_ANALYSIS_PROMPT
from ai_companion.settings import settings


def _mime_type(image_bytes: bytes) -> str:
    """Guess the image type from its first bytes (Chainlit uploads are often PNG)."""
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class ImageToText:
    """A class to handle image-to-text conversion (description and OCR) using Together's vision models."""

    REQUIRED_ENV_VARS = ["TOGETHER_API_KEY"]

    def __init__(self):
        """Initialize the ImageToText class and validate environment variables."""
        self._validate_env_vars()
        self._client: Optional[Together] = None
        self.logger = logging.getLogger(__name__)

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def client(self) -> Together:
        """Get or create Together client instance using singleton pattern."""
        if self._client is None:
            self._client = Together(api_key=settings.TOGETHER_API_KEY)
        return self._client

    async def analyze_image(self, image_data: Union[str, bytes], prompt: str = "") -> str:
        """Describe an image and transcribe any text on it, including handwriting.

        Args:
            image_data: Either a file path (str) or binary image data (bytes)
            prompt: Optional prompt to guide the image analysis

        Returns:
            str: Description of the image, followed by the text found on it

        Raises:
            ValueError: If the image data is empty or invalid
            ImageToTextError: If the image analysis fails
        """
        try:
            # Handle file path
            if isinstance(image_data, str):
                if not os.path.exists(image_data):
                    raise ValueError(f"Image file not found: {image_data}")
                with open(image_data, "rb") as f:
                    image_bytes = f.read()
            else:
                image_bytes = image_data

            if not image_bytes:
                raise ValueError("Image data cannot be empty")

            # Convert image to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            # Create the messages for the vision API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or IMAGE_ANALYSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{_mime_type(image_bytes)};base64,{base64_image}"},
                        },
                    ],
                }
            ]

            # Make the API call
            response = self.client.chat.completions.create(
                model=settings.ITT_MODEL_NAME,
                messages=messages,
                max_tokens=settings.ITT_MAX_TOKENS,
            )

            if not response.choices or not response.choices[0].message.content:
                raise ImageToTextError("No response received from the vision model")

            description = response.choices[0].message.content
            self.logger.info(f"Generated image description: {description}")

            return description

        except Exception as e:
            raise ImageToTextError(f"Failed to analyze image: {str(e)}") from e
