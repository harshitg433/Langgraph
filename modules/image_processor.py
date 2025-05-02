import os
import base64
import logging
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return ""

def image_summarize(img_base64: str, prompt: str) -> str:
    """
    Generate a summary of an image using the OpenAI model
    """
    try:
        chat = ChatOpenAI(model="gpt-4-turbo-2024-04-09", max_tokens=1024)
        msg = chat.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ])
        ])
        return msg.content
    except Exception as e:
        logger.error(f"Error summarizing image: {str(e)}")
        return "Image summary unavailable due to processing error."

def generate_img_summaries(path: str) -> Tuple[List[str], List[str]]:
    """
    Generate summaries for all images in a directory
    """
    img_base64_list, image_summaries = [], []
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    try:
        if not os.path.exists(path):
            logger.warning(f"Image directory {path} does not exist")
            return img_base64_list, image_summaries
            
        for img_file in sorted(os.listdir(path)):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(path, img_file)
                base64_image = encode_image(img_path)
                if base64_image:
                    img_base64_list.append(base64_image)
                    summary = image_summarize(base64_image, prompt)
                    image_summaries.append(summary)
                    logger.info(f"Processed image: {img_file}")
    except Exception as e:
        logger.error(f"Error generating image summaries: {str(e)}")
    
    return img_base64_list, image_summaries