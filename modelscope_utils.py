"""Helpers for loading models and datasets from ModelScope."""
from pathlib import Path


_DATASET_IDS = {
    "wikitext": "modelscope/wikitext",
    "ptb_text_only": "xeon09112/ptb_text_only",
    "allenai/c4": "allenai/c4",
    "ai2_arc": "modelscope/ai2_arc",
    "glue": "modelscope/glue",
    "super_glue": "modelscope/super_glue",
    "piqa": "opencompass/piqa",
}


def modelscope_model_path(model):
    """Return a local model path, downloading the model from ModelScope."""
    if Path(model).exists():
        return model

    from modelscope import snapshot_download

    return snapshot_download(model)


def load_modelscope_dataset(
    path,
    name=None,
    split=None,
    data_dir=None,
    data_files=None,
    cache_dir=None,
    download_mode=None,
):
    """Load a ModelScope dataset using the project's Hugging Face names."""
    from modelscope.msdatasets import MsDataset

    dataset_id = _DATASET_IDS.get(path, path)
    if path == "allenai/c4" and name == "allenai--c4":
        name = "en"

    kwargs = {
        "subset_name": name,
        "split": split,
        "data_dir": data_dir,
        "data_files": data_files,
        "cache_dir": cache_dir,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if download_mode is not None:
        kwargs["download_mode"] = getattr(download_mode, "value", download_mode)

    try:
        return MsDataset.load(dataset_id, **kwargs)
    except Exception as first_error:
        # Some ModelScope mirrors omit the original config name.
        if not name:
            raise first_error
        kwargs.pop("subset_name", None)
        try:
            return MsDataset.load(dataset_id, **kwargs)
        except Exception:
            raise first_error
