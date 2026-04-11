"""Server entry point for DualEye H-MARL Traffic Environment."""

import uvicorn


def main():
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
