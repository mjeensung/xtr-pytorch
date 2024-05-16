import os
from typing import Optional, Union
from huggingface_hub import hf_hub_download

def load_file_path(
    model_name_or_path: str,
    filename: str,
) -> Optional[str]:
    # If file is local
    file_path = os.path.join(model_name_or_path, filename)
    if os.path.exists(file_path):
        return file_path

    # If file is remote
    try:
        return hf_hub_download(
            model_name_or_path,
            filename=filename,
        )
    except Exception:
        return
