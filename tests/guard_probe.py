"""Helper for the FM-71 guard test: binds the download names at the top level, as a test module can."""

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.hub_mixin import hf_hub_download as mixin_download
