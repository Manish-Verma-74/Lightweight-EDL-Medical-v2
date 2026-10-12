import numpy as np
import pytest

torch = pytest.importorskip("torch")
tvm = pytest.importorskip("torchvision.models")

from edl.augment import apply_cutmix, apply_mixup  # noqa: E402
from edl.constants import BACKBONES, FRAMEWORKS  # noqa: E402
from edl.frameworks import build_framework  # noqa: E402
from edl.models.backbones import SPEC, WEIGHTS_MEMBER, build_backbone  # noqa: E402
from edl.models.model import LightweightModel  # noqa: E402

K = 7


def test_pinned_weight_enums_exist_without_downloading():
    for name, (_, enum_name) in SPEC.items():
        assert hasattr(getattr(tvm, enum_name), WEIGHTS_MEMBER), name
    assert WEIGHTS_MEMBER == "IMAGENET1K_V1"


def test_feature_dimensions():
    dims = {b: build_backbone(b, pretrained=False)[1] for b in BACKBONES}
    assert dims == {"efficientnet_b0": 1280, "mobilenet_v3_small": 1024, "shufflenet_v2": 1024}


def soft_targets(kind, x, labels):
    rng = np.random.default_rng(0)
    onehot = torch.eye(K)[labels]
    fn = apply_mixup if kind == "mixup" else apply_cutmix
    return fn(x, onehot, 0.2 if kind == "mixup" else 1.0, rng)


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_forward_backward_for_all_twelve_pairs_with_hard_mixup_and_cutmix_targets(backbone, framework, cfg):
    torch.manual_seed(0)
    model = LightweightModel(backbone, framework, K, pretrained=False)
    fw = build_framework(framework, cfg)
    x = torch.randn(4, 3, 96, 96)
    labels = torch.tensor([0, 2, 4, 6])

    model.eval()
    pred = fw.predict(model(x))
    assert pred["probs"].shape == (4, K) and pred["native_uncertainty"].shape == (4,)
    assert torch.allclose(pred["probs"].sum(1), torch.ones(4), atol=1e-4)
    for key in ("probs", "native_uncertainty", "msp_uncertainty", "entropy"):
        assert torch.isfinite(pred[key]).all(), key

    model.train()
    for kind in ("hard", "mixup", "cutmix"):
        xb, target = (x, labels) if kind == "hard" else soft_targets(kind, x.clone(), labels)
        model.zero_grad()
        loss = fw.loss(model(xb), target, epoch=1)
        assert torch.isfinite(loss), kind
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), kind


def test_evaluation_mode_inference_is_deterministic():
    model = LightweightModel("efficientnet_b0", "f_edl", K, pretrained=False).eval()
    x = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_wrong_names_are_rejected():
    with pytest.raises(ValueError):
        build_backbone("resnet50", pretrained=False)
    with pytest.raises(ValueError):
        LightweightModel("efficientnet_b0", "dropout", K, pretrained=False)
