import unittest

import numpy as np

from time_series_preprocessing import create_context_windows as series_windows
from video_preprocessing import create_context_windows as video_windows, mean_embeddings


class PreprocessingTests(unittest.TestCase):
    def test_series_windows_keep_channels_and_drop_partial_tail(self):
        values = np.arange(22).reshape(11, 2)
        result = series_windows(values, 4, stride=3)
        self.assertEqual(result.values.shape, (3, 2, 4))
        np.testing.assert_array_equal(result.starts, [0, 3, 6])
        np.testing.assert_array_equal(result.ends, [4, 7, 10])
        np.testing.assert_array_equal(result.values[-1], values[6:10].T)

    def test_video_windows_repeat_endpoints_and_preserve_centers(self):
        windows, centers = video_windows(4, 3, stride=2)
        np.testing.assert_array_equal(windows, [[0, 0, 1], [1, 2, 3]])
        np.testing.assert_array_equal(centers, [0, 2])
        np.testing.assert_allclose(
            mean_embeddings(np.array([[0], [3], [6], [9]]), 3),
            [[1], [3], [6], [8]],
        )
        windows, centers = video_windows(1, 5)
        np.testing.assert_array_equal(windows, [[0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(centers, [0])

    def test_invalid_video_window_settings(self):
        for length, width, stride in [(0, 5, 1), (4, 2, 1), (4, 5, 0), (4, True, 1)]:
            with self.assertRaises(ValueError):
                video_windows(length, width, stride=stride)


if __name__ == '__main__':
    unittest.main()
