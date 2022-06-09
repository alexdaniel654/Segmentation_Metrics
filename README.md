# Segmentation Metrics
[![Python CI](https://github.com/alexdaniel654/Segmentation_Metrics/actions/workflows/python_ci.yml/badge.svg)](https://github.com/alexdaniel654/Segmentation_Metrics/actions/workflows/python_ci.yml) 
[![PyPI version](https://badge.fury.io/py/segmentationmetrics.svg)](https://badge.fury.io/py/segmentationmetrics)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/494534661.svg)](https://zenodo.org/badge/latestdoi/494534661)
### Volumetric binary mask segmentation accuracy metrics

A small package for assessing the accuracy of binary segmentations. There are lots of metrics that can be used to compare how close two segmentations are, here voxel overlap, surface and volume based metrics are all calculated at once and returned either as individual metrics, a dictionary or a Pandas DataFrame.

The surface based metrics in this package are calculated using code from [deepmind's surface-distance](https://github.com/deepmind/surface-distance) repository, however as this is not available as a PyPI package, the code has been included as a submodule here.

## Calculated Metrics
### Voxel overlap based metrics
* [Dice Score/F1 Score](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
* [Jaccard Score](https://en.wikipedia.org/wiki/Jaccard_index)
* [Sensitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
* [Specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
* [Precision](https://en.wikipedia.org/wiki/Precision_and_recall)
* [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification)

### Surface based metrics
* [Mean Surface Distance](https://www.creatis.insa-lyon.fr/Challenge/CETUS/evaluation.html#:~:text=Mean%20surface%20distance%3A%20the%20mean,computed%20in%20a%20similar%20way.) (in mm) - The symmetric mean surface distance is returned by default i.e. the mean of the distance from surface A to surface B and surface B to surface A.
* [Hausdorff Distance](https://en.wikipedia.org/wiki/Hausdorff_distance) (in mm) - Computes the robust distance based on the percentile of distances rather than the maximum distance.

### Volume based metrics
* Volume Difference (in millilitres)

## Example Usage
```python
import nibabel as nib # Package for reading MRI data
import segmentationmetrics as sm

img_manual = nib.load('mask_manually_segmented.nii.gz') # Load manually generated ground truth mask
img_automatic = nib.load('mask_automatically_segmented.nii.gz') # Load automatically generated mask

# Get voxel data from image object
mask_manual = img_manual.get_fdata()
mask_automatic = img_automatic.get_fdata()

# Get zoom from header
zoom = img_manual.header.get_zooms()

# Generate metrics
metrics = sm.SegmentationMetrics(mask_automatic, mask_manual, zoom)

# Print the dice score
print(f'The Dice score is {metrics.dice:.2f}')
```
`The Dice score is 0.85`
```python
# Get and print a DataFrame containing all the scores for this mask pair
df = metrics.get_df()
print(df)
```
```
                                      Metric       Score
dice                                    Dice    0.844512
jaccard                              Jaccard    0.730870
sensitivity                      Sensitivity    0.732352
specificity                      Specificity    0.999926
precision                          Precision    0.997239
accuracy                            Accuracy    0.990492
mean_surface_distance  Mean Surface Distance    1.459697
hausdorff_distance        Hausdorff Distance    7.027224
volume_difference          Volume Difference -107.212906
true_volume                      True Volume  403.632624
predicted_volume            Predicted Volume  296.419718
```
```python
# As above but with asymmetric mean surface distance and Hausdorff distance defined by the 99th percentil rather than the 95th percentile.
metrics = sm.SegmentationMetrics(mask_automatic, mask_manual, zoom, symmetric=False, percentile=99)
df = metrics.get_df()
print(df)
```
```
                                      Metric                                    Score
dice                                    Dice                                 0.844512
jaccard                              Jaccard                                  0.73087
sensitivity                      Sensitivity                                 0.732352
specificity                      Specificity                                 0.999926
precision                          Precision                                 0.997239
accuracy                            Accuracy                                 0.990492
mean_surface_distance  Mean Surface Distance  (1.6603182056644057, 1.259075931110695)
hausdorff_distance        Hausdorff Distance                                 9.335755
volume_difference          Volume Difference                              -107.212906
true_volume                      True Volume                               403.632624
predicted_volume            Predicted Volume                               296.419718
```