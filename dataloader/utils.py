# dataloader/utils.py

import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"Function '{func.__name__}' took {time.time() - start_time:.2f}s to complete.")
        return result
    return wrapper

def download_file(url, dest_path):
    # Implement file download logic
    pass
