from scipy.signal import savgol_filter
#from scipy.signal import argrelextrema
from .translate import *
import numpy as np

class TraceCorrection(object):
    """
    Methods for smoothing traces
    """
    @classmethod
    def trim_by_min(cls, trace):
        """
        returns a new Trace with values preceding either min set to zero
        """
        early_min, late_min = cls.get_minima(trace.coordinates)
        nc = trace.coordinates[:]
        empty_point = (0,0)
        nc[:early_min] = early_min * [empty_point]
        nc[late_min:] = [empty_point for i in range(late_min, len(nc))]
        return Trace(image=trace.image, tracer="trimmed_min", coordinates=nc, metadata=trace.metadata)

    @classmethod
    def get_minima(cls, coords):
        """
        Divide a Trace in 2 and find the position of the min y value for each section
        """
        x,y = zip(*coords)
        # switch sign
        y = -np.array(y)
        midpoint = int(len(x)/2)
        return (np.argmin(y[:midpoint]), midpoint + np.argmin(y[midpoint:]))

    @classmethod
    def sg_filtering(cls, t, window=None, order=9):
        """
        Applies Savitsky-Golay filtering to a Trace's y values
        """
        if window:
            if window % 2 == 0:
                raise WindowException("Window size must be odd")
            if order >= window:
                raise WindowException("Order must be less than window size")
        x,y = zip(*t.coordinates)
        # window must be odd
        window_size = window if window else max([i for i in range(0, len(y)) if i % 2 != 0])
        y_hat = savgol_filter(y, window_size, order)
        if len(y_hat) != 32 and len(y) == 32:
            print(t.image)
        # zip together the smoothed y values with the old x values
        new_coords = list(zip(x,y_hat))
        tracer = "svagol_filter-w={}-order={}".format(window_size, order)
        return Trace(image=t.image, tracer=tracer, coordinates=new_coords, metadata=t.metadata)

    @classmethod
    def threshold_coordinates(cls, coords, threshold):
        """
        """
        # unzip the coordinates
        x,y = zip(*coords)
        # convert to lists
        x = list(x)
        y = list(y)
        # traverse these values
        indices = range(0, len(y) - 1)
        #reversed_y = list(reversed(y))
        for i in indices:
            # moving from the right edge...
            current_y = y[i]
            # get the point immediately to the left
            #print("current_y: {}".format(current_y))
            next_y = y[i + 1]
            #print("next_y: {}".format(next_y))
            # is the next value positive or negative?
            sign = 1 if next_y > current_y else -1
            # what is the actual diff?
            diff = abs(next_y - current_y)
            # set the diff to the actual diff or the threshold (whichever is less)
            diff = diff if diff < threshold else threshold
            diff *= sign
            # thresholded y value
            adjusted_y = current_y + diff
            y[i+1] = adjusted_y
        return list(zip(x,y))

    @classmethod
    def threshold_trace(cls, t, anterior_mean, body_mean, front_mean):
        """
        """
        coords = t.coordinates
        edge_length = int(len(coords) / 4 * 1.5)
        rear = coords[:edge_length]
        body = coords[edge_length:-1 * edge_length]
        front = coords[-1 * edge_length:]
        # remove rear mininimum
        reversed_rear = list(reversed(rear))
        thresholded_rear = reversed(cls.threshold_coordinates(reversed_rear, anterior_mean))
        thresholded_body = cls.threshold_coordinates(body, body_mean)
        thresholded_front = cls.threshold_coordinates(front, front_mean)
        thresholded_coords = cls.thresholded_front + thresholded_body + thresholded_rear
        tracer = "thresholded-rm={0}-bm={1}-fm={2}".format(anterior_mean, body_mean, front_mean)
        return Trace(image=t.image, tracer=tracer, coordinates=thresholded_coords, metadata=t.metadata)


class WindowException(Exception):
    pass
