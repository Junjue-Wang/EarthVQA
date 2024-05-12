from albumentations.core.transforms_interface import ImageOnlyTransform, NoOp, to_tuple
import numpy as np

class RandomDropChannel(ImageOnlyTransform):
    """
    Args:
        drop_rate (float)

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
            self,
            drop_rate=0.1,
            always_apply=False,
            p=0.5,
    ):
        super(RandomDropChannel, self).__init__(always_apply, p)
        self.drop_rate = drop_rate

    def apply(self, img, **params):
        h, w, c = img.shape
        zero_channel = int(c * self.drop_rate)
        masked_indexes = np.random.randint(0, c, size=zero_channel)
        img[:, :, masked_indexes] = 0
        return img



    def get_transform_init_args_names(self):
        return ("drop_rate")