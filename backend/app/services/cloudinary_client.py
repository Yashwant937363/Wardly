import os
import cloudinary
import cloudinary.uploader


def _cloudinary_configured() -> None:
    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
        secure=True,
    )

def get_cloudinary_folder_path(owner_id: str, upload_id: str) -> str:
    return f"wardrobe/{owner_id}/{upload_id}"
 
def upload_to_cloudinary(file_path: str, folder: str) -> str:
    """Upload a local file to Cloudinary and return its secure_url."""
 
    result = cloudinary.uploader.upload(file_path, folder=folder)
    return result["secure_url"]
 
 
def build_thumbnail_url(original_url: str, width: int = 300) -> str:
    """
    Cloudinary serves resized versions on-the-fly by inserting transformation
    params into the URL path — no separate thumbnail file needs to be stored.
    """
    return original_url.replace(
        "/upload/", f"/upload/w_{width},c_fit,q_auto,f_auto/"
    )