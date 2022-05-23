import numpy as np
import pandas as pd

from . import surface_distance as sd


class SegmentationMetrics:

    def __init__(self, prediction, truth, zoom):
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
        self.mean_surface_distance = self._av_dist()
        self.hausdorff_distance = self._hausdorff_dist(95)
        self.true_volume = self._true_volume()
        self.predicted_volume = self._predicted_volume()
        self.volume_difference = self._volume_difference()

    def get_dict(self):
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

    def _av_dist(self):
        av_surf_dist = sd.compute_average_surface_distance(self._surface_dist)
        return np.mean(av_surf_dist)

    def _hausdorff_dist(self, percentile=95):
        return sd.compute_robust_hausdorff(self._surface_dist, percentile)

    def _true_volume(self):
        return np.sum(self.truth) * np.prod(self.zoom) / 1000

    def _predicted_volume(self):
        return np.sum(self.prediction) * np.prod(self.zoom) / 1000

    def _volume_difference(self):
        return self.predicted_volume - self.true_volume
