"""Server entry point for Hugging Face Spaces."""

import uvicorn
from app.main import app


def main():
    """Run the server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
