import numpy as np
import pandas as pd

from . import surface_distance as sd


class SegmentationMetrics:
    """"
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
        self.dice = self._dice()
        self.jaccard = self._jaccard()
        self.sensitivity = self._sensitivity()
        self.specificity = self._specificity()
        self.precision = self._precision()
        self.accuracy = self._accuracy()
        self._surface_dist = sd.compute_surface_distances(self.prediction,
                                                          self.truth,
                                                          self.zoom)
        self.mean_surface_distance = self._av_dist(symmetric)
        self.hausdorff_distance = self._hausdorff_dist(percentile)
        self.true_volume = self._true_volume()
        self.predicted_volume = self._predicted_volume()
        self.volume_difference = self._volume_difference()

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

    def _dice(self):
        return np.sum(self.prediction[self.truth == 1]) * 2.0 / \
               (np.sum(self.prediction) + np.sum(self.truth))

    def _jaccard(self):
        return np.sum(self.prediction[self.truth == 1]) / \
               (np.sum(self.prediction[self.truth == 1]) +
                np.sum(self.prediction != self.truth))

    def _sensitivity(self):
        return np.sum(self.prediction[self.truth == 1]) / \
               (np.sum(self.prediction[self.truth == 1]) +
                np.sum((self.truth == 1) & (self.prediction == 0)))

    def _specificity(self):
        return np.sum((self.truth == 0) & (self.prediction == 0)) / \
               (np.sum((self.truth == 0) & (self.prediction == 0)) +
                np.sum((self.truth == 0) & (self.prediction == 1)))

    def _precision(self):
        return np.sum(self.prediction[self.truth == 1]) / \
               (np.sum(self.prediction[self.truth == 1]) +
                np.sum((self.truth == 0) & (self.prediction == 1)))

    def _accuracy(self):
        return (np.sum(self.prediction[self.truth == 1]) +
                np.sum((self.truth == 0) & (self.prediction == 0))) / \
               self.truth.size

    def _av_dist(self, symmetric=True):
        av_surf_dist = sd.compute_average_surface_distance(self._surface_dist)
        if symmetric:
            msd = np.mean(av_surf_dist)
        else:
            msd = av_surf_dist
        return msd

    def _hausdorff_dist(self, percentile=95):
        return sd.compute_robust_hausdorff(self._surface_dist, percentile)

    def _true_volume(self):
        return np.sum(self.truth) * np.prod(self.zoom) / 1000

    def _predicted_volume(self):
        return np.sum(self.prediction) * np.prod(self.zoom) / 1000

    def _volume_difference(self):
        return self.predicted_volume - self.true_volume
