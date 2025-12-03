"""Chain implementations for the nano banana pipeline."""

from .document_content import DocumentContentChain
from .image_generation import (
    ImageGenerationChain,
    build_batch_request,
    build_full_prompt,
    get_mime_type,
)
from .image_prompt import ImagePromptChain
from .multi_hop_question import MultiHopQuestionChain
from .single_hop_qa import SingleHopQAChain


def create_document_content_chain(config):
    """Create a document content chain."""
    return DocumentContentChain(config)


def create_image_generation_chain(config):
    """Create an image generation chain."""
    return ImageGenerationChain(config)


__all__ = [
    "DocumentContentChain",
    "ImageGenerationChain",
    "ImagePromptChain",
    "MultiHopQuestionChain",
    "SingleHopQAChain",
    "build_batch_request",
    "build_full_prompt",
    "create_document_content_chain",
    "create_image_generation_chain",
    "get_mime_type",
]
