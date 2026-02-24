from typing import Dict
from enum import IntEnum, auto


class DSET(IntEnum):
    LiberoObject = auto()
    Droid = auto()
    PickPlaceCan = auto()
    OpenDrawer = auto()
    OpenOven = auto()


dataset_locations: Dict[str, Dict[int, str]] = {}
dataset_locations["default"] = {
    DSET.LiberoObject: "./data_processed/libero/libero_object/**/*.h5",

    DSET.Droid: "./data_processed/droid/data/**/*.h5",

    DSET.PickPlaceCan: "./data_processed/pick-place/*.h5",
    DSET.OpenOven: "./data_processed/oven/*.h5",
    DSET.OpenDrawer: "./data_processed/drawer/*.h5",

    # add other datasets here
}

LOC = dataset_locations["default"]

