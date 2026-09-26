from io import BytesIO
from PIL import Image, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

def load_image(payload: bytes, content_type: str | None, max_bytes: int) -> Image.Image:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError("Upload a JPEG, PNG, or WebP image.")
    if not payload or len(payload) > max_bytes:
        raise ValueError("Image is empty or exceeds the upload-size limit.")
    try:
        image = Image.open(BytesIO(payload))
        image.verify()
        image = Image.open(BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a readable image.") from error
    if image.width < 16 or image.height < 16:
        raise ValueError("Image dimensions must be at least 16×16 pixels.")
    return image
