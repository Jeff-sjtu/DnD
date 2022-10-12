"""3DPW dataset, developed based on VIBE."""

from .dataset_3d import Dataset3D
from dnd.models.builder import DATASET


@DATASET.register_module
class ThreeDPW(Dataset3D):
    def __init__(self, cfg, ann_file, overlap=0.75, train=True):
        db_name = '3dpw'

        # overlap = overlap if is_train else 0
        super(ThreeDPW, self).__init__(
            cfg=cfg,
            ann_file=ann_file,
            overlap=overlap,
            train=train,
            dataset_name=db_name,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')
