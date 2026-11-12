"""Helpers for loading models and datasets from ModelScope."""
from pathlib import Path


_DATASET_IDS = {
    "wikitext": "modelscope/wikitext",
    "ptb_text_only": "torushy/flm-ptb-text-only",
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
    dataset_id = _DATASET_IDS.get(path, path)

    if path == "ptb_text_only":
        # ModelScope's PTB mirror is a datasets.save_to_disk() export. MsDataset
        # 1.18 mistakes its state.json files for source records, so read the
        # downloaded Arrow dataset directly.
        from datasets import load_from_disk
        from modelscope.hub.snapshot_download import dataset_snapshot_download

        root = Path(dataset_snapshot_download(dataset_id, cache_dir=cache_dir))
        loaded = load_from_disk(str(root / "dataset"))
        return loaded[split] if split is not None else loaded

    from modelscope.msdatasets import MsDataset

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
        # ModelScope 1.18 uses its own enum, not datasets.DownloadMode.
        from modelscope.utils.constant import DownloadMode as ModelScopeDownloadMode

        mode_name = getattr(download_mode, "name", download_mode)
        if mode_name == "REUSE_CACHE_IF_EXISTS":
            mode_name = "REUSE_DATASET_IF_EXISTS"
        try:
            kwargs["download_mode"] = ModelScopeDownloadMode[mode_name]
        except KeyError as error:
            raise ValueError(f"Unsupported ModelScope download mode: {mode_name}") from error

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
