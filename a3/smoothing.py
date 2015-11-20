from scipy.signal import savgol_filter
from .translate import *


def threshold_trace(t, anterior_mean, body_mean, front_mean):
    coords = t.coordinates
    edge_length = int(len(coords) / 4 * 1.5)
    rear = coords[:edge_length]
    body = coords[edge_length:-1 * edge_length]
    front = coords[-1 * edge_length:]
    # print(t)
    # print("edge length: {}".format(edge_length))
    # print("rear: {0}".format(len(rear)))
    # print("body: {0}".format(len(body)))
    # print("front: {0}".format(len(front)))
    # print("all point: {}".format(len(rear) + len(body) + len(front)))
    def threshold_coordinates(coords, threshold):
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
            #print("adjusted_y: {}".format(adjusted_y))
            y[i+1] = adjusted_y
        return list(zip(x,y))
    thresholded_rear = threshold_coordinates(list(reversed(rear)), anterior_mean)
    thresholded_body = threshold_coordinates(body, body_mean)
    thresholded_front = threshold_coordinates(front, front_mean)
    thresholded_coords = thresholded_front + thresholded_body + thresholded_rear
    tracer = "thresholded-rm={0}-bm={1}-fm={2}".format(anterior_mean, body_mean, front_mean)
    return Trace(image=t.image, tracer=tracer, coordinates=thresholded_coords, metadata=t.metadata)

def trim_trace(t, left_k=2, right_k=2):
    """
    trim k points from either side of a Trace's non-empty coordinates
    """
    trimmed_nonempty = t.nonempty[0+left_k:right_k*-1]
    tracer = "trimmed_left-k={0}_right-k{1}=".format(left_k, right_k)
    return Trace(image=t.image, tracer=tracer, coordinates=trimmed_nonempty, metadata=t.metadata)

def smooth_trace(t, window=None, order=3):
    """
    Applies savgol filtering to a trace's y values
    """
    if window:
        if window % 2 == 0:
            raise WindowException("Window size must be odd")
        if order >= window:
            raise WindowException("Order must be less than window size")
    # only correct the non-empty points
    x,y = zip(*t.nonempty)
    # window must be odd
    window_size = window if window else len(y) - 1
    y_hat = savgol_filter(y, window_size, order)
    # zip together the smoothed y values with the old x values
    new_coords = list(zip(x,y_hat))
    tracer = "svagol_filter-w={}-order={}".format(window, order)
    return Trace(image=t.image, tracer=tracer, coordinates=new_coords, metadata=t.metadata)

class WindowException(Exception):
    pass
