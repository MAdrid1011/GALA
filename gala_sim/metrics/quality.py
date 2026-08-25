"""Fixed-metric volume helpers; no quality value is inferred from cycle data."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from gala_sim.config import GalaConfig


@dataclass(frozen=True)
class QualityConfig:
    data_min: float
    data_max: float
    ssim_window: int
    ssim_sigma: float
    lpips_slices: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    lpips_network: str

    def __post_init__(self) -> None:
        if self.data_max <= self.data_min or self.ssim_window <= 1 or self.ssim_window % 2 == 0:
            raise ValueError("quality ranges and SSIM window are invalid")
        if (
            self.ssim_sigma <= 0
            or len(self.lpips_slices) != 3
            or any(not indexes for indexes in self.lpips_slices)
            or not self.lpips_network
        ):
            raise ValueError("quality slice and sigma settings are invalid")

    @classmethod
    def from_gala(cls, config: GalaConfig) -> "QualityConfig":
        names = (
            "quality.data_min",
            "quality.data_max",
            "quality.ssim_window",
            "quality.ssim_sigma",
            "quality.lpips_slices",
            "quality.lpips_network",
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
                lpips_slices=slices,  # type: ignore[arg-type]
                lpips_network=str(parameters["quality.lpips_network"]["value"]),
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


def _lpips(reference: np.ndarray, candidate: np.ndarray, config: QualityConfig) -> float:
    try:
        import torch
        import lpips
    except ImportError as error:  # pragma: no cover - environment diagnosis
        raise RuntimeError("PyTorch and lpips are required for LPIPS") from error
    net = lpips.LPIPS(net=config.lpips_network).eval()
    values = []
    scale = config.data_max - config.data_min
    for axis, indexes in enumerate(config.lpips_slices):
        for index in indexes:
            ref = np.take(reference, index, axis=axis)
            cand = np.take(candidate, index, axis=axis)
            ref = np.asarray((ref - config.data_min) / scale, dtype=np.float32)
            cand = np.asarray((cand - config.data_min) / scale, dtype=np.float32)
            ref_tensor = torch.from_numpy(ref)[None, None].repeat(1, 3, 1, 1) * 2 - 1
            cand_tensor = torch.from_numpy(cand)[None, None].repeat(1, 3, 1, 1) * 2 - 1
            with torch.inference_mode():
                values.append(float(net(ref_tensor, cand_tensor).item()))
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
