# Segmentation Metrics
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