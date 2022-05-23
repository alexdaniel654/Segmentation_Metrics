import numpy as np
import pandas as pd
import pytest

from segmentationmetrics import SegmentationMetrics
from skimage.morphology import ball


class TestSegmentationMetrics:
    # Generate three spherical masks, two that are similar and one that
    # doesn't overlap at all
    img_shape = (256, 256, 256)
    canvas = np.zeros(img_shape)

    # Base sphere
    centre_a = (128, 128, 128)
    radius_a = 80
    ball_a = ball(radius_a)
    img_a = canvas.copy()
    img_a[centre_a[0] - ball_a.shape[0] // 2:
          centre_a[0] + ball_a.shape[0] // 2 + 1,
          centre_a[1] - ball_a.shape[1] // 2:
          centre_a[1] + ball_a.shape[1] // 2 + 1,
          centre_a[2] - ball_a.shape[2] // 2:
          centre_a[2] + ball_a.shape[2] // 2 + 1] = ball_a

    # Slightly smaller and offset
    centre_b = (132, 132, 132)
    radius_b = 77
    ball_b = ball(radius_b)
    img_b = canvas.copy()
    img_b[centre_b[0] - ball_b.shape[0] // 2:
          centre_b[0] + ball_b.shape[0] // 2 + 1,
          centre_b[1] - ball_b.shape[1] // 2:
          centre_b[1] + ball_b.shape[1] // 2 + 1,
          centre_b[2] - ball_b.shape[2] // 2:
          centre_b[2] + ball_b.shape[2] // 2 + 1] = ball_b

    # Fits in the corner, not overlapping with either sphere_a or sphere_b
    centre_c = (50, 50, 50)
    radius_c = 40
    ball_c = ball(radius_c)
    img_c = canvas.copy()
    img_c[centre_c[0] - ball_c.shape[0] // 2:
          centre_c[0] + ball_c.shape[0] // 2 + 1,
          centre_c[1] - ball_c.shape[1] // 2:
          centre_c[1] + ball_c.shape[1] // 2 + 1,
          centre_c[2] - ball_c.shape[2] // 2:
          centre_c[2] + ball_c.shape[2] // 2 + 1] = ball_c

    def test_basic_case(self):
        # Overlapping spheres, isotropic voxels
        sm = SegmentationMetrics(self.img_a, self.img_b, (1, 1, 1))
        assert sm.get_dict() == pytest.approx({'accuracy': 0.9810,
                                               'dice': 0.9216,
                                               'hausdorff_distance': 8.6023,
                                               'jaccard': 0.8546,
                                               'mean_surface_distance': 3.6676,
                                               'precision': 0.8719,
                                               'predicted_volume': 2143.641,
                                               'sensitivity': 0.9773,
                                               'specificity': 0.9815,
                                               'true_volume': 1912.319,
                                               'volume_difference': 231.3220},
                                              rel=1e-20, abs=1e-4)
        # Verify dataframe is returned
        assert type(sm.get_df()) == pd.DataFrame

    def test_options(self):
        # Confirm changing Hausdorff percentile and surface distance
        # symmetry flags work as expected
        sm = SegmentationMetrics(self.img_a, self.img_b, (1, 1, 1),
                                 percentile=99, symmetric=False)
        assert np.isclose(sm.hausdorff_distance, 9.2736, rtol=1e-20, atol=1e-4)
        assert np.isclose(sm.mean_surface_distance, (3.7923, 3.5430),
                          rtol=1e-20, atol=1e-4).all()

    def test_nonisotropic_voxels(self):
        # Overlapping spheres, non-isotropic voxels
        sm = SegmentationMetrics(self.img_a[..., ::2], self.img_b[..., ::2],
                                 (1, 1, 2))
        assert sm.get_dict() == pytest.approx({'accuracy': 0.9811,
                                               'dice': 0.9218,
                                               'hausdorff_distance': 8.4853,
                                               'jaccard': 0.8549,
                                               'mean_surface_distance': 3.5352,
                                               'precision': 0.8724,
                                               'predicted_volume': 2142.914,
                                               'sensitivity': 0.9770,
                                               'specificity': 0.9816,
                                               'true_volume': 1913.474,
                                               'volume_difference': 229.4400},
                                              rel=1e-20, abs=1e-4)

    def test_2d(self):
        sm = SegmentationMetrics(self.img_a[..., 128], self.img_b[..., 128],
                                 (1, 1))
        assert sm.get_dict() == pytest.approx({'accuracy': 0.9688,
                                               'dice': 0.9470,
                                               'hausdorff_distance': 8.4853,
                                               'jaccard': 0.8993,
                                               'mean_surface_distance': 3.8745,
                                               'precision': 0.9110,
                                               'predicted_volume': 20.081,
                                               'sensitivity': 0.9860,
                                               'specificity': 0.9619,
                                               'true_volume': 18.553,
                                               'volume_difference': 1.5280},
                                              rel=1e-20, abs=1e-4)

    def test_no_overlap(self):
        # Non-overlapping spheres
        sm = SegmentationMetrics(self.img_a, self.img_c, (1, 1, 1))
        assert sm.get_dict() == pytest.approx({'accuracy': 0.8563,
                                               'dice': 0.0,
                                               'hausdorff_distance': 169.2661,
                                               'jaccard': 0.0,
                                               'mean_surface_distance': 84.2791,
                                               'precision': 0.0,
                                               'predicted_volume': 2143.641,
                                               'sensitivity': 0.0,
                                               'specificity': 0.8702,
                                               'true_volume': 267.761,
                                               'volume_difference': 1875.88},
                                              rel=1e-20, abs=1e-4)
        assert type(sm.get_df()) == pd.DataFrame

