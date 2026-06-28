import numpy as np
import numpy.typing as npt
from calibration_tool.types import Calibration
from matplotlib import pyplot as plt
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent.joinpath(
    "calibration_data/calibration_2026-06-14_12-44-52.jsonl"
)


def load_calibration_data(file_path: Path = DEFAULT_PATH) -> Calibration:
    with open(file_path, "r") as f:
        calibration_data = Calibration.model_validate_json(f.read())
    return calibration_data


def _iqr_filter(values: np.ndarray) -> np.ndarray:
    """Return values within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Falls back to all if all filtered."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    mask = (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)
    filtered = values[mask]
    return filtered if len(filtered) > 0 else values


def _extract_calibration_data(calibration: Calibration) -> tuple:
    """Extract and average gaze readings per calibration point.

    Applies IQR filtering per point before averaging to reject blinks/saccades.

    Returns:
        (pitch_avg, yaw_avg, pixel_x, pixel_y, raw_pitch, raw_yaw, raw_pixel_x, raw_pixel_y)
    """
    pitch_avg = []
    yaw_avg = []
    pixel_x = []
    pixel_y = []
    raw_pitch = []
    raw_yaw = []
    raw_pixel_x = []
    raw_pixel_y = []

    for point in calibration.calibration_points:
        yaw_values = np.array(
            [
                result.yaw.item() if hasattr(result.yaw, "item") else float(result.yaw)
                for result in point.GazeResults
            ]
        )
        pitch_values = np.array(
            [
                result.pitch.item() if hasattr(result.pitch, "item") else float(result.pitch)
                for result in point.GazeResults
            ]
        )

        pitch_avg.append(float(np.mean(_iqr_filter(pitch_values))))
        yaw_avg.append(float(np.mean(_iqr_filter(yaw_values))))
        pixel_x.append(float(point.pixel_coordinates[0]))
        pixel_y.append(float(point.pixel_coordinates[1]))

        for pitch, yaw in zip(pitch_values, yaw_values):
            raw_pitch.append(float(pitch))
            raw_yaw.append(float(yaw))
            raw_pixel_x.append(float(point.pixel_coordinates[0]))
            raw_pixel_y.append(float(point.pixel_coordinates[1]))

    return (
        np.array(pitch_avg),
        np.array(yaw_avg),
        np.array(pixel_x),
        np.array(pixel_y),
        np.array(raw_pitch),
        np.array(raw_yaw),
        np.array(raw_pixel_x),
        np.array(raw_pixel_y),
    )


def calculate_gaze_bounds(calibration: Calibration) -> tuple[float, float, float, float]:
    """Return (pitch_min, pitch_max, yaw_min, yaw_max) from IQR-filtered calibration averages."""
    pitch, yaw, _, _, _, _, _, _ = _extract_calibration_data(calibration)
    return float(pitch.min()), float(pitch.max()), float(yaw.min()), float(yaw.max())


def calculate_polynomial_mapping_univariate(calibration: Calibration) -> npt.NDArray[np.float64]:
    """Univariate: pitch→x, yaw→y independently."""
    pitch, yaw, pixel_x, pixel_y, _, _, _, _ = _extract_calibration_data(calibration)
    polyfit_pitch_to_x = np.polyfit(pitch, pixel_x, deg=3)
    polyfit_yaw_to_y = np.polyfit(yaw, pixel_y, deg=3)
    return np.array([polyfit_pitch_to_x, polyfit_yaw_to_y])


def calculate_polynomial_mapping_bivariate(calibration: Calibration, alpha: float = 1.0) -> tuple:
    """Bivariate with cross-terms: (pitch, yaw) → (pixel_x, pixel_y).

    Uses Ridge regression (L2 penalty) to regularize the 10-feature design matrix,
    which is nearly rank-deficient with a typical 9-point calibration grid.

    Args:
        calibration: Calibration data
        alpha: Ridge regularization strength. Higher = smoother polynomial, less
            prone to extrapolation divergence. 0 reduces to unregularized least squares.

    Returns (coeffs_x, coeffs_y) shape (10,) for degree-3 2D polynomial.
    """
    pitch, yaw, pixel_x, pixel_y, _, _, _, _ = _extract_calibration_data(calibration)

    # Design matrix with cross-terms: 1, pitch, yaw, pitch², pitch·yaw, yaw², pitch³, pitch²·yaw, pitch·yaw², yaw³
    A = np.column_stack(
        [
            np.ones_like(pitch),
            pitch,
            yaw,
            pitch**2,
            pitch * yaw,
            yaw**2,
            pitch**3,
            pitch**2 * yaw,
            pitch * yaw**2,
            yaw**3,
        ]
    )

    # Ridge: w = (AᵀA + alpha·I)⁻¹ Aᵀy
    ridge_matrix = A.T @ A + alpha * np.eye(A.shape[1])
    coeffs_x = np.linalg.solve(ridge_matrix, A.T @ pixel_x)
    coeffs_y = np.linalg.solve(ridge_matrix, A.T @ pixel_y)
    return coeffs_x, coeffs_y


def calculate_polynomial_mapping(calibration: Calibration) -> npt.NDArray[np.float64]:
    """Default: univariate for backward compatibility."""
    return calculate_polynomial_mapping_univariate(calibration)


def _eval_bivariate(pitch, yaw, coeffs):
    """Evaluate 2D polynomial with cross-terms."""
    A = np.column_stack(
        [
            np.ones_like(pitch) if np.isscalar(pitch) else np.ones(len(pitch)),
            pitch,
            yaw,
            pitch**2,
            pitch * yaw,
            yaw**2,
            pitch**3,
            pitch**2 * yaw,
            pitch * yaw**2,
            yaw**3,
        ]
    )
    return float(A @ coeffs) if np.isscalar(pitch) else A @ coeffs


def visualize_polynomial_fit(
    calibration: Calibration, fit_type: str = "univariate", save_path: Path | str | None = None
) -> None:
    """Visualize calibration polynomial fit quality.

    Args:
        calibration: Calibration data
        fit_type: "univariate", "bivariate", or "both"
        save_path: Optional path to save figure
    """
    pitch, yaw, pixel_x, pixel_y, raw_pitch, raw_yaw, raw_pixel_x, raw_pixel_y = (
        _extract_calibration_data(calibration)
    )

    if fit_type == "univariate":
        _visualize_univariate(
            pitch, yaw, pixel_x, pixel_y, raw_pitch, raw_yaw, raw_pixel_x, raw_pixel_y, save_path
        )
    elif fit_type == "bivariate":
        _visualize_bivariate(
            pitch, yaw, pixel_x, pixel_y, raw_pitch, raw_yaw, raw_pixel_x, raw_pixel_y, save_path
        )
    elif fit_type == "both":
        _visualize_both(
            pitch, yaw, pixel_x, pixel_y, raw_pitch, raw_yaw, raw_pixel_x, raw_pixel_y, save_path
        )
    else:
        raise ValueError(f"Unknown fit_type: {fit_type}. Use 'univariate', 'bivariate', or 'both'.")


def _visualize_univariate(
    pitch, yaw, pixel_x, pixel_y, raw_pitch, raw_yaw, raw_pixel_x, raw_pixel_y, save_path
):
    """Univariate visualization."""
    polyfit_pitch_to_x = np.polyfit(pitch, pixel_x, deg=3)
    polyfit_yaw_to_y = np.polyfit(yaw, pixel_y, deg=3)

    pitch_smooth = np.linspace(pitch.min(), pitch.max(), 100)
    yaw_smooth = np.linspace(yaw.min(), yaw.max(), 100)

    pixel_x_fit = np.polyval(polyfit_pitch_to_x, pitch_smooth)
    pixel_y_fit = np.polyval(polyfit_yaw_to_y, yaw_smooth)

    pixel_x_pred = np.polyval(polyfit_pitch_to_x, pitch)
    pixel_y_pred = np.polyval(polyfit_yaw_to_y, yaw)

    residuals_x = pixel_x - pixel_x_pred
    residuals_y = pixel_y - pixel_y_pred

    rmse_x = float(np.sqrt(np.mean(residuals_x**2)))
    rmse_y = float(np.sqrt(np.mean(residuals_y**2)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Univariate Polynomial Fit (Pitch→X, Yaw→Y)", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(raw_pitch, raw_pixel_x, alpha=0.2, s=10, label="Raw", color="lightblue")
    ax.scatter(pitch, pixel_x, alpha=1, s=100, label="Averaged", color="blue", zorder=5)
    ax.plot(pitch_smooth, pixel_x_fit, "r-", linewidth=2, label="Fit (deg=3)")
    ax.set_xlabel("Pitch (rad)")
    ax.set_ylabel("Pixel X")
    ax.set_title(f"Pitch → X (RMSE: {rmse_x:.2f} px)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(raw_yaw, raw_pixel_y, alpha=0.2, s=10, label="Raw", color="lightgreen")
    ax.scatter(yaw, pixel_y, alpha=1, s=100, label="Averaged", color="green", zorder=5)
    ax.plot(yaw_smooth, pixel_y_fit, "r-", linewidth=2, label="Fit (deg=3)")
    ax.set_xlabel("Yaw (rad)")
    ax.set_ylabel("Pixel Y")
    ax.set_title(f"Yaw → Y (RMSE: {rmse_y:.2f} px)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(pitch, residuals_x, s=100, color="blue", zorder=5)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.fill_between(
        [pitch.min(), pitch.max()], -rmse_x, rmse_x, alpha=0.2, color="red", label="±1 RMSE"
    )
    ax.set_xlabel("Pitch (rad)")
    ax.set_ylabel("Residual (px)")
    ax.set_title("Pitch → X Residuals")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(yaw, residuals_y, s=100, color="green", zorder=5)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.fill_between(
        [yaw.min(), yaw.max()], -rmse_y, rmse_y, alpha=0.2, color="red", label="±1 RMSE"
    )
    ax.set_xlabel("Yaw (rad)")
    ax.set_ylabel("Residual (px)")
    ax.set_title("Yaw → Y Residuals")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    _save_or_show(fig, save_path, "univariate")
    print("\n=== UNIVARIATE FIT ===")
    print(f"Pitch → X: RMSE={rmse_x:.3f} px")
    print(f"Yaw → Y:   RMSE={rmse_y:.3f} px")


def _visualize_bivariate(
    pitch, yaw, pixel_x, pixel_y, raw_pitch, raw_yaw, raw_pixel_x, raw_pixel_y, save_path
):
    """Bivariate visualization with cross-terms."""
    A = np.column_stack(
        [
            np.ones_like(pitch),
            pitch,
            yaw,
            pitch**2,
            pitch * yaw,
            yaw**2,
            pitch**3,
            pitch**2 * yaw,
            pitch * yaw**2,
            yaw**3,
        ]
    )
    coeffs_x = np.linalg.lstsq(A, pixel_x, rcond=None)[0]
    coeffs_y = np.linalg.lstsq(A, pixel_y, rcond=None)[0]

    pixel_x_pred = _eval_bivariate(pitch, yaw, coeffs_x)
    pixel_y_pred = _eval_bivariate(pitch, yaw, coeffs_y)

    residuals_x = pixel_x - pixel_x_pred
    residuals_y = pixel_y - pixel_y_pred

    rmse_x = float(np.sqrt(np.mean(residuals_x**2)))
    rmse_y = float(np.sqrt(np.mean(residuals_y**2)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Bivariate Polynomial Fit (Pitch×Yaw→X,Y with Cross-Terms)", fontsize=16, fontweight="bold"
    )

    ax = axes[0, 0]
    scatter = ax.scatter(pitch, pixel_x, c=yaw, s=100, cmap="RdYlGn", zorder=5, label="Observed")
    ax.scatter(
        pitch,
        pixel_x_pred,
        marker="x",
        s=150,
        color="red",
        linewidth=3,
        label="Predicted",
        zorder=4,
    )
    ax.set_xlabel("Pitch (rad)")
    ax.set_ylabel("Pixel X")
    ax.set_title(f"Pitch→X with Yaw Influence (RMSE: {rmse_x:.2f} px)")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Yaw (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    scatter = ax.scatter(yaw, pixel_y, c=pitch, s=100, cmap="RdYlBu_r", zorder=5, label="Observed")
    ax.scatter(
        yaw, pixel_y_pred, marker="x", s=150, color="red", linewidth=3, label="Predicted", zorder=4
    )
    ax.set_xlabel("Yaw (rad)")
    ax.set_ylabel("Pixel Y")
    ax.set_title(f"Yaw→Y with Pitch Influence (RMSE: {rmse_y:.2f} px)")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Pitch (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(pitch, residuals_x, s=100, color="blue", zorder=5)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.fill_between(
        [pitch.min(), pitch.max()], -rmse_x, rmse_x, alpha=0.2, color="red", label="±1 RMSE"
    )
    ax.set_xlabel("Pitch (rad)")
    ax.set_ylabel("Residual (px)")
    ax.set_title("X Residuals (Bivariate)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(yaw, residuals_y, s=100, color="green", zorder=5)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.fill_between(
        [yaw.min(), yaw.max()], -rmse_y, rmse_y, alpha=0.2, color="red", label="±1 RMSE"
    )
    ax.set_xlabel("Yaw (rad)")
    ax.set_ylabel("Residual (px)")
    ax.set_title("Y Residuals (Bivariate)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    _save_or_show(fig, save_path, "bivariate")
    print("\n=== BIVARIATE FIT ===")
    print(f"X (Pitch primary): RMSE={rmse_x:.3f} px")
    print(f"Y (Yaw primary):   RMSE={rmse_y:.3f} px")


def _visualize_both(
    pitch, yaw, pixel_x, pixel_y, raw_pitch, raw_yaw, raw_pixel_x, raw_pixel_y, save_path
):
    """Compare univariate vs bivariate."""
    # Univariate
    poly_pitch_to_x = np.polyfit(pitch, pixel_x, deg=3)
    poly_yaw_to_y = np.polyfit(yaw, pixel_y, deg=3)
    uni_x_pred = np.polyval(poly_pitch_to_x, pitch)
    uni_y_pred = np.polyval(poly_yaw_to_y, yaw)
    uni_rmse_x = float(np.sqrt(np.mean((pixel_x - uni_x_pred) ** 2)))
    uni_rmse_y = float(np.sqrt(np.mean((pixel_y - uni_y_pred) ** 2)))

    # Bivariate
    A = np.column_stack(
        [
            np.ones_like(pitch),
            pitch,
            yaw,
            pitch**2,
            pitch * yaw,
            yaw**2,
            pitch**3,
            pitch**2 * yaw,
            pitch * yaw**2,
            yaw**3,
        ]
    )
    coeffs_x = np.linalg.lstsq(A, pixel_x, rcond=None)[0]
    coeffs_y = np.linalg.lstsq(A, pixel_y, rcond=None)[0]
    biv_x_pred = _eval_bivariate(pitch, yaw, coeffs_x)
    biv_y_pred = _eval_bivariate(pitch, yaw, coeffs_y)
    biv_rmse_x = float(np.sqrt(np.mean((pixel_x - biv_x_pred) ** 2)))
    biv_rmse_y = float(np.sqrt(np.mean((pixel_y - biv_y_pred) ** 2)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Univariate vs Bivariate Comparison", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(pitch, pixel_x, s=100, alpha=0.6, label="Observed", color="blue", zorder=5)
    ax.scatter(
        pitch,
        uni_x_pred,
        marker="^",
        s=100,
        alpha=0.7,
        label="Univariate",
        color="orange",
        zorder=4,
    )
    ax.scatter(
        pitch,
        biv_x_pred,
        marker="x",
        s=150,
        color="red",
        linewidth=2.5,
        label="Bivariate",
        zorder=3,
    )
    ax.set_xlabel("Pitch (rad)")
    ax.set_ylabel("Pixel X")
    ax.set_title(f"X: Univariate={uni_rmse_x:.2f}px vs Bivariate={biv_rmse_x:.2f}px")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(yaw, pixel_y, s=100, alpha=0.6, label="Observed", color="green", zorder=5)
    ax.scatter(
        yaw, uni_y_pred, marker="^", s=100, alpha=0.7, label="Univariate", color="orange", zorder=4
    )
    ax.scatter(
        yaw, biv_y_pred, marker="x", s=150, color="red", linewidth=2.5, label="Bivariate", zorder=3
    )
    ax.set_xlabel("Yaw (rad)")
    ax.set_ylabel("Pixel Y")
    ax.set_title(f"Y: Univariate={uni_rmse_y:.2f}px vs Bivariate={biv_rmse_y:.2f}px")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    uni_res_x = pixel_x - uni_x_pred
    biv_res_x = pixel_x - biv_x_pred
    ax.scatter(pitch, uni_res_x, s=80, alpha=0.6, label="Univariate", color="orange")
    ax.scatter(pitch, biv_res_x, s=80, alpha=0.6, label="Bivariate", color="red")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Pitch (rad)")
    ax.set_ylabel("Residual (px)")
    ax.set_title("X Residuals")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    uni_res_y = pixel_y - uni_y_pred
    biv_res_y = pixel_y - biv_y_pred
    ax.scatter(yaw, uni_res_y, s=80, alpha=0.6, label="Univariate", color="orange")
    ax.scatter(yaw, biv_res_y, s=80, alpha=0.6, label="Bivariate", color="red")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Yaw (rad)")
    ax.set_ylabel("Residual (px)")
    ax.set_title("Y Residuals")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    _save_or_show(fig, save_path, "comparison")

    print("\n=== COMPARISON ===")
    print(
        f"X: Univariate={uni_rmse_x:.3f}px vs Bivariate={biv_rmse_x:.3f}px ({(uni_rmse_x - biv_rmse_x) / uni_rmse_x * 100:.1f}% improvement)"
    )
    print(
        f"Y: Univariate={uni_rmse_y:.3f}px vs Bivariate={biv_rmse_y:.3f}px ({(uni_rmse_y - biv_rmse_y) / uni_rmse_y * 100:.1f}% improvement)"
    )


def _save_or_show(fig, save_path, fit_type):
    """Save or display figure."""
    if save_path:
        if isinstance(save_path, str):
            save_path = Path(save_path)
        if "{" not in str(save_path):
            save_path = save_path.parent / f"{save_path.stem}_{fit_type}{save_path.suffix}"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    calibration_data = load_calibration_data()

    print("Generating calibration fit visualizations...")
    output_dir = Path(__file__).parent / "calibration_data"
    output_dir.mkdir(exist_ok=True)

    print("\n1. Univariate visualization (for report):")
    visualize_polynomial_fit(
        calibration_data, fit_type="univariate", save_path=output_dir / "calib_fit_univariate.png"
    )

    print("\n2. Bivariate visualization (for report):")
    visualize_polynomial_fit(
        calibration_data, fit_type="bivariate", save_path=output_dir / "calib_fit_bivariate.png"
    )

    print("\n3. Comparison visualization (for report):")
    visualize_polynomial_fit(
        calibration_data, fit_type="both", save_path=output_dir / "calib_fit_comparison.png"
    )
