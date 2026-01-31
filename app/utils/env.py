import os


def running_locally() -> bool:
    """Checks if the application is running in a local environment.

    Returns:
        bool: True if running locally, False otherwise.
    """
    return os.getenv("STREAMLIT_SERVER_HEADLESS") != "1"
