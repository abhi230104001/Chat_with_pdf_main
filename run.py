
# import uvicorn
# from app.config import get_settings


# def main() -> None:
#     settings = get_settings()
#     print(f"\n{'='*60}")
#     print("  Chat with PDF — RAG Chatbot")
#     print(f"{'='*60}")
#     print(f"  Server:  http://{settings.app_host}:{settings.app_port}")
#     print(f"  Docs:    http://localhost:{settings.app_port}/api/docs")
#     print(f"  Debug:   {settings.debug}")
#     print(f"  API Key: {'✓ configured' if settings.has_api_key() else '✗ NOT SET (set GOOGLE_API_KEY in .env)'}")
#     print(f"{'='*60}\n")

#     uvicorn.run(
#         "app.main:app",
#         host=settings.app_host,
#         port=settings.app_port,
#         reload=settings.debug,
#         log_level=settings.log_level.lower(),
#     )


# if __name__ == "__main__":
#     main()


"""
run.py — Development server launcher.

Usage:
    python run.py

Reads host/port/debug from settings.
"""

import uvicorn
import webbrowser
from threading import Timer
from app.config import get_settings


def main() -> None:
    settings = get_settings()
    print(f"\n{'='*60}")
    print("  Chat with PDF — RAG Chatbot")
    print(f"{'='*60}")
    print(f"  Server:  http://{settings.app_host}:{settings.app_port}")
    print(f"  Docs:    http://localhost:{settings.app_port}/api/docs")
    print(f"  Debug:   {settings.debug}")
    print(f"  API Key: {'✓ configured' if settings.has_api_key() else '✗ NOT SET (set GOOGLE_API_KEY in .env)'}")
    print(f"{'='*60}\n")

    # Automatically open the frontend in the browser
    Timer(
        1.5,
        lambda: webbrowser.open(
            f"http://localhost:{settings.app_port}/"
        ),
    ).start()

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
