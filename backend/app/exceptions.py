class UnusableImageError(Exception):
    """Raised when Gemini flags the image as unusable for cataloging."""
 
 
class UnsupportedCategoryError(Exception):
    """Raised when the given category has no cloth-parser mapping."""