import numpy as np
import pandas as pd

from . import surface_distance as sd


class SegmentationMetrics:
    """
    Attributes
    ----------
    dice : float
        Dice similarity score.
    jaccard : float
        Jaccard similarity score.
    sensitivity : float
        Sensitivity/recall/true positive rate.
    specificity : float
        Specificity/selectivity/true negative rate.
    precision : float
        Precision/positive predictive value.
    accuracy : float
        Accuracy.
    mean_surface_distance : float or tuple
        The mean surface distance, defaults to symmetric.
    hausdorff_distance : float
        The robust Hausdorff distance, defaults to 95th percentile.
    true_volume : float
        The volume of the true mask (in milliliters)
    predicted_volume : float
        The volume of the predicted mask (in milliliters)
    volume_difference : float
        The difference between the true and predicted volumes (in 
        milliliters). Positive values show the predicted volume is larger 
        than the true volume, negative values show the true volume is larger
        than the predicted volume.
    """
    def __init__(self, prediction, truth, zoom, percentile=95, symmetric=True):
        """
        Initialises the SegmentationMetrics class instance.

        Parameters
        ----------
        prediction : np.ndarray
            An array of bools or ints (0 and 1) representing the predicted
            mask.
        truth : np.ndarray
            An array of bools or ints (0 and 1) representing the ground truth
            mask.
        zoom : tuple
            The length of each voxel dimension in millimeters.
        percentile : int, default 95
            The percentile of surface distances to define as the Hausdorff
            distance.
        symmetric : bool, default True
            If true, the symmetric mean surface distance is calculated i.e.
            the returned mean surface distance is the average of the means
            surface distance from surface A to surface B and the mean
            surface distance from surface B to surface A. If false, a tuple
            is returned with both mean surface distances.
        """
        self.prediction = prediction > 0.5
        self.truth = truth > 0.5
        self.zoom = zoom
        self.labels = np.unique(self.truth[self.truth > 0])

        if self.labels.size == 0:
            self.dice = np.nan
            self.jaccard = np.nan
            self.sensitivity = np.nan
            self.specificity = np.nan
            self.precision = np.nan
            self.accuracy = np.nan
            self.mean_surface_distance = np.nan
            self.hausdorff_distance = np.nan
            self.true_volume = np.nan
            self.predicted_volume = np.nan
            self.volume_difference = np.nan
        else:
            dice_vals = [self._dice(label) for label in self.labels]
            jaccard_vals = [self._jaccard(label) for label in self.labels]
            sensitivity_vals = [self._sensitivity(label) for label in self.labels]
            specificity_vals = [self._specificity(label) for label in self.labels]
            precision_vals = [self._precision(label) for label in self.labels]
            accuracy_vals = [self._accuracy(label) for label in self.labels]
            mean_surface_distance_vals = []
            hausdorff_distance_vals = []
            for label in self.labels:
                if np.sum(self.truth == label) == 0 or np.sum(self.prediction == label) == 0:
                    # If there are no voxels of this label in either the truth or prediction, set surface distances to infinity
                    mean_surface_distance_vals.append(np.nan)
                    hausdorff_distance_vals.append(np.nan)
                else:
                    self._surface_dist = sd.compute_surface_distances(self.prediction == label,
                                                                    self.truth == label,
                                                                    self.zoom)
                    mean_surface_distance_vals.append(self._av_dist(symmetric))
                    hausdorff_distance_vals.append(self._hausdorff_dist(percentile))
            
            self.dice = np.mean(dice_vals)
            self.jaccard = np.mean(jaccard_vals)
            self.sensitivity = np.mean(sensitivity_vals)
            self.specificity = np.mean(specificity_vals)
            self.precision = np.mean(precision_vals)
            self.accuracy = np.mean(accuracy_vals)
            self.mean_surface_distance = np.mean(mean_surface_distance_vals, axis=0)
            self.hausdorff_distance = np.mean(hausdorff_distance_vals)
            self.true_volume = np.sum([self._true_volume(label) for label in self.labels])
            self.predicted_volume = np.sum([self._predicted_volume(label) for label in self.labels])
            if self.labels.size == 1:
                self.volume_difference = self._volume_difference(self.labels[0])
            else:
                self.volume_difference = np.sum(np.abs([self._volume_difference(label) for label in self.labels]))
            # self.true_volume = self._true_volume()
            # self.predicted_volume = self._predicted_volume()
            # self.volume_difference = self._volume_difference()

    def get_dict(self):
        """
        Generate a dictionary of segmentation accuracy metrics.

        Returns
        -------
        metrics : dict
            Segmentation accuracy.
        """
        return {'dice': self.dice,
                'jaccard': self.jaccard,
                'sensitivity': self.sensitivity,
                'specificity': self.specificity,
                'precision': self.precision,
                'accuracy': self.accuracy,
                'mean_surface_distance': self.mean_surface_distance,
                'hausdorff_distance': self.hausdorff_distance,
                'volume_difference': self.volume_difference,
                'true_volume': self.true_volume,
                'predicted_volume': self.predicted_volume}

    def get_df(self):
        """
        Generate a Pandas DataFrame containing the segmentation accuracy
        metrics.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with metric in one column and score in the next column.
        """
        df = pd.DataFrame.from_dict(self.get_dict(),
                                    orient='index',
                                    columns=['Score'])
        df['Metric'] = ['Dice', 'Jaccard', 'Sensitivity', 'Specificity',
                        'Precision', 'Accuracy', 'Mean Surface Distance',
                        'Hausdorff Distance', 'Volume Difference',
                        'True Volume', 'Predicted Volume']
        df = df[['Metric', 'Score']]
        return df

    def _dice(self, label=1):
        tp, fp, fn, tn = self._get_confusion_counts(label)
        if tp + fp + fn == 0:
            dice = 0.0
        else:
            dice = (2 * tp) / (2 * tp + fp + fn)
        return dice

    def _jaccard(self, label=1):
        tp, fp, fn, tn = self._get_confusion_counts(label)
        if tp + fp + fn == 0:
            jaccard = 0.0
        else:
            jaccard = tp / (tp + fp + fn)
        return jaccard

    def _sensitivity(self, label=1):
        tp, fp, fn, tn = self._get_confusion_counts(label)
        if tp + fn == 0:
            sensitivity = 0.0
        else:
            sensitivity = tp / (tp + fn)
        return sensitivity

    def _specificity(self, label=1):
        tp, fp, fn, tn = self._get_confusion_counts(label)
        if tn + fp == 0:
            specificity = 0.0
        else:
            specificity = tn / (tn + fp)
        return specificity

    def _precision(self, label=1):
        tp, fp, fn, tn = self._get_confusion_counts(label)
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        return precision

    def _accuracy(self, label=1):
        tp, fp, fn, tn = self._get_confusion_counts(label)
        if tp + fp + fn + tn == 0:
            accuracy = 0.0
        else:
            accuracy = (tp + tn) / (tp + fp + fn + tn)
        return accuracy

    def _av_dist(self, symmetric=True):
        av_surf_dist = sd.compute_average_surface_distance(self._surface_dist)
        if symmetric:
            msd = np.mean(av_surf_dist)
        else:
            msd = av_surf_dist
        return msd

    def _hausdorff_dist(self, percentile=95):
        return sd.compute_robust_hausdorff(self._surface_dist, percentile)

    def _true_volume(self, label=1):
        return np.sum(self.truth == label) * np.prod(self.zoom) / 1000

    def _predicted_volume(self, label=1):
        return np.sum(self.prediction == label) * np.prod(self.zoom) / 1000

    def _volume_difference(self, label=1):
        return self._predicted_volume(label) - self._true_volume(label)
    
    def _get_confusion_counts(self, label):
        tp = np.sum((self.prediction == label) & (self.truth == label))
        fp = np.sum((self.prediction == label) & (self.truth != label))
        fn = np.sum((self.prediction != label) & (self.truth == label))
        tn = np.sum((self.prediction != label) & (self.truth != label))
        return tp, fp, fn, tn
