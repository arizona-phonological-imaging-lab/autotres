from .utils import get_path
from glob import glob
import json
import os
import argparse
import re


class TraceFactory(object):
    # regex for finding tracer id and image name
    image_name_pattern = re.compile("(.*?(jpg|png))")
    tracer_id_pattern = re.compile("(jpg|png)\.(.+)?\.traced\.txt")
    old_trace_pattern = "*.traced.txt"
    # yes, this is actually what the web tracer is expecting.
    # TODO: fix the web tracer
    default_roi = "{\"srcX\":0,\"srcY\":0,\"destX\":720,\"destY\":480}"

    @classmethod
    def read_old_trace_file(cls, f):
        """
        Reads an old trace file and returns a Trace instance
        """
        def get_tracer(f):
            return re.search(TraceFactory.tracer_id_pattern, os.path.basename(f)).group(2)

        def get_image_name(f):
            return re.search(TraceFactory.image_name_pattern, os.path.basename(f)).group(0)

        def get_coordinates(f):
            """
            Read the x,y coordinates from an old trace file
            """
            coords = []
            with open(f, 'r') as trace_file:
                for line in trace_file:
                    i, x, y = tuple(line.split())
                    coords.append((float(x), float(y)))
            return coords

        # necessary components of a Trace
        image = get_image_name(f)
        tracer = get_tracer(f)
        coordinates = get_coordinates(f)
        # store the image location
        parent_dir = os.path.dirname(f)
        meta = MetaData(image_loc=parent_dir)
        return Trace(image, tracer, coordinates, metadata=meta)

    @classmethod
    def traces_from_old_traces(cls, d):
        """
        Reads a directory of old traces and returns a list of Trace instances
        """
        # expand the path, if needed
        d = get_path(d)
        # create a glob pattern
        glob_pattern = os.path.join(d, TraceFactory.old_trace_pattern)
        # apply the pattern to retrieve the old traces
        old_traces = glob(glob_pattern)
        return [cls.read_old_trace_file(t) for t in old_traces]

    @classmethod
    def traces_to_json(cls, traces, outfile, subject_id, project_id, roi=None, include_empty=True):
        """
        Takes a list of Trace instances and returns a json container
        """
        def write_json(outfile, json_data):
            """
            Closure for writing json file
            """
            with open(get_path(outfile), 'w') as out:
                json.dump(json_data, out)

        roi = roi if roi else TraceFactory.default_roi
        tracer_id = traces[0].tracer
        subject_id = subject_id
        project_id = project_id
        trace_data = dict([t.to_json(include_empty) for t in traces])
        json_d = dict([("roi", roi), ("tracer-id", tracer_id),("subject-id", subject_id), ("project-id", project_id), ('trace-data', trace_data)])
        # write the json to a file
        write_json(outfile, json_d)

    @classmethod
    def JSON_file_to_traces(cls, json_file):
        """
        reads JSON data and returns a list of Traces
        """
        with open(get_path(json_file), 'r') as f:
            json_data = json.load(f)
        # build metadata
        roi = json_data['roi']
        tracer_id = json_data['tracer-id']
        project_id = json_data['project-id']
        subject_id = json_data['subject-id']
        meta = MetaData(project_id=project_id, subject_id=subject_id, roi=roi)
        # construct traces
        traces = []
        for (image, td) in json_data['trace-data'].items():
            # convert list of (x,y) dicts into x,y tuples
            coordinates = [(coord['x'], coord['y']) for coord in td]
            # build a Trace
            traces.append(Trace(image, tracer_id, coordinates, metadata=meta))
        return traces

class MetaData(object):
    """
    Information shared between sets of traces
    """
    def __init__(self, image_loc=None, project_id=None, subject_id=None, roi=None):
        self.image_loc = image_loc
        self.project_id = project_id
        self.subject_id = subject_id
        self.roi = roi

class Trace(object):
    """
    A representation of a tongue trace
    """
    def __init__(self, image, tracer, coordinates, metadata=None):
        # what to use for empty points
        self._empty_val = 0
        self.image = image
        self.tracer = tracer
        self.metadata = metadata
        self.coordinates = sorted(self.filter_points(coordinates), key=lambda x: x[0])
        self.nonempty = [(x,y) for (x,y) in self.coordinates if x != -1 and y != -1 and x != 0 and y != 0]

    def __repr__(self):
        return "Trace of {0} by {1} with {2} coordinates ({3} non-empty)".format(self.image, self.tracer, len(self.coordinates), len(self.nonempty))

    def filter_points(self, points):
        # replace (-1, -1) with (0, 0)
        return [(x,y) if (x != -1) else (self._empty_val, self._empty_val) for (x,y) in points]

    def coords_to_json(self, include_empty):
        json_list = []
        pairs = self.coordinates if include_empty else self.nonempty
        for pair in pairs:
            x, y = pair[0], pair[-1]
            if x != -1 and y != -1:
                json_list.append({"x":x, "y":y})
        return json_list

    def to_json(self, use_nonempty):
        return (self.image, self.coords_to_json(use_nonempty))




if __name__ == "__main__":
    pass
