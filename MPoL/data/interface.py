from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .data import LooseVisibilities


class DataPipeline(Protocol):
    """Protocol to follow when setting up data for ingest into MPoL.

    1. Users must implement an `extract_data` method that takes in one data file
    and returns a LooseVisibilities object.
    """

    def extract_data(self, file: str) -> LooseVisibilities:
        """Setup the data for ingesting into MPoL.

        Implement your custom logic for reading the data here ensuring that the
        output is a LooseVisibility object. The data dimensions should be
        (num_channels, num_pixels, num_pixels).
        """
        ...
