import glob
from pathlib import Path
from typing import TYPE_CHECKING

import lightning
import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from MPoL.visibilities_matching.interface import ProcessedData, ProcessingStrategy

    from .interface import DataPipeline

_TEMP_FOLDER = "MPoL_data/tensors"


class MPoLDataset(Dataset):
    def __init__(self, files: list[str]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> ProcessedData:
        return torch.load(
            Path(f"{_TEMP_FOLDER}/{self.files[index]}").with_suffix(".pt")
        )


class MPoLDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        pipeline: DataPipeline,
        strategy: ProcessingStrategy,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.files = glob.glob(f"{data_dir}")
        self.pipeline = pipeline
        self.strategy = strategy
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        # Create the temporary folder if it doesn't exist.
        Path(_TEMP_FOLDER).mkdir(parents=True, exist_ok=True)

        for file in self.files:
            # Only run this if we've not already saved the data before.
            tensor_file = Path(f"{_TEMP_FOLDER}/{file}").with_suffix(".pt")
            if tensor_file.exists():
                continue

            raw_data = self.pipeline.extract_data(file)
            processed_data = self.strategy.process_data(raw_data)
            torch.save(processed_data, tensor_file)

    def setup(self, stage: str) -> None:
        self.dataset = MPoLDataset(self.files)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size)
