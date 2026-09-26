import base64
import logging
import os
import time
from typing import Optional
from urllib.parse import quote

import requests
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from together import Together

from ai_companion.core.exceptions import TextToImageError
from ai_companion.core.prompts import IMAGE_ENHANCEMENT_PROMPT, IMAGE_SCENARIO_PROMPT
from ai_companion.core.structured_output import get_structured_runnable
from ai_companion.settings import settings


class ScenarioPrompt(BaseModel):
    """Class for the scenario response"""

    narrative: str = Field(..., description="The AI's narrative response to the question")
    image_prompt: str = Field(..., description="The visual prompt to generate an image representing the scene")


class EnhancedPrompt(BaseModel):
    """Class for the text prompt"""

    content: str = Field(
        ...,
        description="The enhanced text prompt to generate an image",
    )


class TextToImage:
    """A class to handle text-to-image generation using Together AI."""

    REQUIRED_ENV_VARS = ["GROQ_API_KEY", "TOGETHER_API_KEY"]

    def __init__(self):
        """Initialize the TextToImage class and validate environment variables."""
        self._validate_env_vars()
        self._together_client: Optional[Together] = None
        self.logger = logging.getLogger(__name__)

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def together_client(self) -> Together:
        """Get or create Together client instance using singleton pattern."""
        if self._together_client is None:
            self._together_client = Together(api_key=settings.TOGETHER_API_KEY)
        return self._together_client

    def _generate_together(self, prompt: str) -> bytes:
        """Generate via Together AI. Needs a valid key AND third-party data sharing."""
        # Only send `steps` when explicitly configured - different models expect
        # very different step counts, and omitting it lets each use its default.
        extra = {"steps": settings.TTI_STEPS} if settings.TTI_STEPS else {}

        response = self.together_client.images.generate(
            prompt=prompt,
            model=settings.TTI_MODEL_NAME,
            width=1024,
            height=768,
            n=1,
            response_format="b64_json",
            **extra,
        )
        return base64.b64decode(response.data[0].b64_json)

    def _generate_pollinations(self, prompt: str) -> bytes:
        """Generate via Pollinations, which needs no API key.

        Used as a fallback so image generation still works while Together is
        unavailable. The service returns a transient 5xx fairly often, so retry.
        """
        url = "https://image.pollinations.ai/prompt/" + quote(prompt, safe="")
        params = {"width": 1024, "height": 768, "nologo": "true"}

        attempts = 4
        last_error = None
        for attempt in range(attempts):
            try:
                r = requests.get(url, params=params, timeout=120)
                if r.status_code == 200 and r.content[:3] in (b"\x89PN", b"\xff\xd8\xff"):
                    return r.content
                last_error = f"HTTP {r.status_code}"
            except Exception as e:  # network blip - worth one more try
                last_error = str(e)

            self.logger.warning(f"Pollinations attempt {attempt + 1}/{attempts} failed: {last_error}")
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))  # brief backoff; 5xx here is usually load

        raise TextToImageError(f"Pollinations failed after {attempts} attempts: {last_error}")

    async def generate_image(self, prompt: str, output_path: str = "") -> bytes:
        """Generate an image, preferring Together and falling back to Pollinations."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        self.logger.info(f"Generating image for prompt: '{prompt}'")

        provider = settings.TTI_PROVIDER
        image_data = None

        if provider in ("auto", "together"):
            try:
                image_data = self._generate_together(prompt)
                self.logger.info("Image generated via Together")
            except Exception as e:
                if provider == "together":
                    raise TextToImageError(f"Failed to generate image: {str(e)}") from e
                # 'auto': Together is commonly blocked by the org-level
                # third-party data sharing setting - fall back rather than fail.
                self.logger.warning(f"Together image generation failed, falling back: {str(e)[:200]}")

        if image_data is None:
            image_data = self._generate_pollinations(prompt)
            self.logger.info("Image generated via Pollinations (fallback)")

        if output_path:
            directory = os.path.dirname(output_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_data)
            self.logger.info(f"Image saved to {output_path}")

        return image_data

    async def create_scenario(self, chat_history: list = None) -> ScenarioPrompt:
        """Creates a first-person narrative scenario and corresponding image prompt based on chat history."""
        try:
            formatted_history = "\n".join([f"{msg.type.title()}: {msg.content}" for msg in chat_history[-5:]])

            self.logger.info("Creating scenario from chat history")

            llm = ChatGroq(
                model=settings.SMALL_TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.4,
                max_retries=2,
            )

            structured_llm = get_structured_runnable(llm, ScenarioPrompt)

            chain = (
                PromptTemplate(
                    input_variables=["chat_history"],
                    template=IMAGE_SCENARIO_PROMPT,
                )
                | structured_llm
            )

            scenario = chain.invoke({"chat_history": formatted_history})
            self.logger.info(f"Created scenario: {scenario}")

            return scenario

        except Exception as e:
            raise TextToImageError(f"Failed to create scenario: {str(e)}") from e

    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance a simple prompt with additional details and context."""
        try:
            self.logger.info(f"Enhancing prompt: '{prompt}'")

            llm = ChatGroq(
                model=settings.SMALL_TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.25,
                max_retries=2,
            )

            structured_llm = get_structured_runnable(llm, EnhancedPrompt)

            chain = (
                PromptTemplate(
                    input_variables=["prompt"],
                    template=IMAGE_ENHANCEMENT_PROMPT,
                )
                | structured_llm
            )

            enhanced_prompt = chain.invoke({"prompt": prompt}).content
            self.logger.info(f"Enhanced prompt: '{enhanced_prompt}'")

            return enhanced_prompt

        except Exception as e:
            raise TextToImageError(f"Failed to enhance prompt: {str(e)}") from e
