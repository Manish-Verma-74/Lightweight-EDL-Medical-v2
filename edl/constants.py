"""Fixed study constants. Anything here is part of the frozen protocol."""

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
MELANOMA_IDX = CLASS_TO_IDX["mel"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

BACKBONES = ["efficientnet_b0", "mobilenet_v3_small", "shufflenet_v2"]
FRAMEWORKS = ["softmax", "edl", "r_edl", "f_edl"]
AUGMENTATIONS = ["base_flips", "mixup", "cutmix", "randaugment"]
SEEDS = [42, 123, 2024]

# Display names used in tables and figures.
FRAMEWORK_DISPLAY = {"softmax": "Softmax", "edl": "EDL", "r_edl": "R-EDL", "f_edl": "F-EDL (no SN)"}
AUGMENTATION_DISPLAY = {"base_flips": "Base (flips)", "mixup": "Mixup", "cutmix": "CutMix",
                        "randaugment": "RandAugment"}

PARTITIONS = ["train", "val", "test"]
