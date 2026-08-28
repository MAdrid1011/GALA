"""Fixed-metric volume helpers; no quality value is inferred from cycle data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import numpy as np

from gala_sim.config import GalaConfig
@dataclass(frozen=True)
class QualityConfig:
    data_min: float
    data_max: float
    ssim_window: int
    ssim_sigma: float
    ssim_boundary: str
    lpips_slices: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    lpips_network: str
    lpips_version: str
    lpips_backbone_sha256: str
    lpips_calibration_sha256: str

    def __post_init__(self) -> None:
        if self.data_max <= self.data_min or self.ssim_window <= 1 or self.ssim_window % 2 == 0:
            raise ValueError("quality ranges and SSIM window are invalid")
        if self.ssim_sigma <= 0:
            raise ValueError("quality slice and sigma settings are invalid")
        if self.ssim_boundary != "reflect":
            raise ValueError("quality SSIM boundary must be reflect")
        if len(self.lpips_slices) != 3 or any(not indexes for indexes in self.lpips_slices):
            raise ValueError("quality slice and sigma settings are invalid")
        if any(
            any(not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in indexes)
            or any(left >= right for left, right in zip(indexes, indexes[1:]))
            for indexes in self.lpips_slices
        ):
            raise ValueError("quality LPIPS slice indexes must be strictly increasing")
        digests = (self.lpips_backbone_sha256, self.lpips_calibration_sha256)
        if (
            self.lpips_network != "alex"
            or not self.lpips_version
            or any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
                   for value in digests)
        ):
            raise ValueError("quality LPIPS network identity is invalid")

    @classmethod
    def from_gala(cls, config: GalaConfig) -> "QualityConfig":
        names = (
            "quality.data_min",
            "quality.data_max",
            "quality.ssim_window",
            "quality.ssim_sigma",
            "quality.ssim_boundary",
            "quality.lpips_slices",
            "quality.lpips_network",
            "quality.lpips_version",
            "quality.lpips_backbone_sha256",
            "quality.lpips_calibration_sha256",
        )
        try:
            parameters = {name: config.parameter(name) for name in names}
        except KeyError as error:
            raise ValueError("quality configuration is incomplete") from error
        pending = [
            name for name, parameter in parameters.items()
            if parameter.get("status") != "frozen" or parameter.get("value") is None
        ]
        if pending:
            raise ValueError("quality configuration is not frozen: " + ", ".join(pending))
        raw_slices = parameters["quality.lpips_slices"]["value"]
        if (
            not isinstance(raw_slices, (list, tuple))
            or len(raw_slices) != 3
            or any(not isinstance(indexes, (list, tuple)) for indexes in raw_slices)
        ):
            raise ValueError("quality LPIPS slices must contain three axis lists")
        try:
            slices = tuple(tuple(int(index) for index in indexes) for indexes in raw_slices)
            return cls(
                data_min=float(parameters["quality.data_min"]["value"]),
                data_max=float(parameters["quality.data_max"]["value"]),
                ssim_window=int(parameters["quality.ssim_window"]["value"]),
                ssim_sigma=float(parameters["quality.ssim_sigma"]["value"]),
                ssim_boundary=str(parameters["quality.ssim_boundary"]["value"]),
                lpips_slices=slices,  # type: ignore[arg-type]
                lpips_network=str(parameters["quality.lpips_network"]["value"]),
                lpips_version=str(parameters["quality.lpips_version"]["value"]),
                lpips_backbone_sha256=str(
                    parameters["quality.lpips_backbone_sha256"]["value"]
                ),
                lpips_calibration_sha256=str(
                    parameters["quality.lpips_calibration_sha256"]["value"]
                ),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("quality configuration values are invalid") from error


@dataclass(frozen=True)
class QualityMetrics:
    psnr: float
    ssim: float
    lpips: float


def _check_volumes(reference: np.ndarray, candidate: np.ndarray, config: QualityConfig) -> None:
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("quality volumes must have identical three-dimensional shapes")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("quality volumes contain non-finite values")
    if float(reference.min()) != config.data_min or float(reference.max()) != config.data_max:
        raise ValueError("quality reference range does not match the frozen configuration")
    if any(
        index < 0 or index >= reference.shape[axis]
        for axis, indexes in enumerate(config.lpips_slices)
        for index in indexes
    ):
        raise ValueError("LPIPS slice is outside the reference volume")


def _psnr(reference: np.ndarray, candidate: np.ndarray, config: QualityConfig) -> float:
    mse = float(np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((config.data_max - config.data_min) ** 2 / mse))


def _ssim(reference: np.ndarray, candidate: np.ndarray, config: QualityConfig) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:  # pragma: no cover - environment diagnosis
        raise RuntimeError("scikit-image is required for SSIM") from error
    return float(structural_similarity(
        reference,
        candidate,
        data_range=config.data_max - config.data_min,
        win_size=config.ssim_window,
        gaussian_weights=True,
        sigma=config.ssim_sigma,
        use_sample_covariance=False,
        channel_axis=None,
    ))


def _normalize_lpips_slice(
    volume: np.ndarray, *, axis: int, index: int, config: QualityConfig,
) -> np.ndarray:
    image = np.take(volume, index, axis=axis)
    normalized = (image - config.data_min) / (config.data_max - config.data_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.asarray(normalized * 2.0 - 1.0, dtype=np.float32)


def _verify_lpips_weights(torch: object, lpips: object, config: QualityConfig) -> None:
    try:
        from torchvision.models import AlexNet_Weights
    except ImportError as error:  # pragma: no cover - environment diagnosis
        raise RuntimeError("torchvision is required for LPIPS weight verification") from error
    checkpoint_name = Path(urlparse(AlexNet_Weights.IMAGENET1K_V1.url).path).name
    backbone = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name  # type: ignore[attr-defined]
    calibration = (
        Path(lpips.__file__).resolve().parent  # type: ignore[attr-defined]
        / "weights" / f"v{config.lpips_version}" / f"{config.lpips_network}.pth"
    )
    for path in (backbone, calibration):
        if not path.is_file():
            raise RuntimeError(f"LPIPS weight is unavailable: {path}")


def _lpips(reference: np.ndarray, candidate: np.ndarray, config: QualityConfig) -> float:
    try:
        import torch
        import lpips
    except ImportError as error:  # pragma: no cover - environment diagnosis
        raise RuntimeError("PyTorch and lpips are required for LPIPS") from error
    net = lpips.LPIPS(
        net=config.lpips_network,
        version=config.lpips_version,
        pretrained=True,
        lpips=True,
        spatial=False,
        pnet_rand=False,
        pnet_tune=False,
        use_dropout=True,
        eval_mode=True,
        verbose=False,
    ).eval()
    _verify_lpips_weights(torch, lpips, config)
    values = []
    for axis, indexes in enumerate(config.lpips_slices):
        for index in indexes:
            ref = _normalize_lpips_slice(reference, axis=axis, index=index, config=config)
            cand = _normalize_lpips_slice(candidate, axis=axis, index=index, config=config)
            ref_tensor = torch.from_numpy(ref)[None, None].repeat(1, 3, 1, 1)
            cand_tensor = torch.from_numpy(cand)[None, None].repeat(1, 3, 1, 1)
            with torch.inference_mode():
                values.append(float(net(ref_tensor, cand_tensor, normalize=False).item()))
    return float(np.mean(values))


def measure_quality(reference: np.ndarray, candidate: np.ndarray, config: QualityConfig) -> QualityMetrics:
    _check_volumes(reference, candidate, config)
    return QualityMetrics(
        psnr=_psnr(reference, candidate, config),
        ssim=_ssim(reference, candidate, config),
        lpips=_lpips(reference, candidate, config),
    )


def compare_quality(reference: QualityMetrics, candidate: QualityMetrics) -> dict[str, float]:
    return {
        "psnr_delta_db": candidate.psnr - reference.psnr,
        "ssim_delta": candidate.ssim - reference.ssim,
        "lpips_delta": candidate.lpips - reference.lpips,
    }
