import numpy as np

class ContourComparison(object):
    """
    functions to compare two sets of traces
    """

    @classmethod
    def MSD(cls, gold_traces, experimental_traces):
        """
        mean sum of distances (MSD) method of Li et al. (2005)
        $MSD(U,V) = \frac{1}{2n}\Big(\sum\limits_{i=1}^{n}\min\limits_{j}\mid{v_i - u_j}\mid + \sum\limits_{i=1}^{n}\min\limits_{j}\mid{u_i - v_j}\mid \Big)$
        """
        traces_LUT = dict()
        for g in gold_traces:
            # is the gold trace empty?  If so, skip it...
            if not g.nonempty:
                continue
            image = g.image
            traces_LUT[image] = [g]
            for e in experimental_traces:
                # assume we have only one corresponding trace in the experimental set...
                if e.image == image:
                    traces_LUT[image].append(e)
                    traces_LUT[image] = tuple(traces_LUT[image])
                    break
        # compare traces for each image
        images, scores = zip(*[cls.compare_traces(*traces) for (image, traces) in traces_LUT.items()])
        return (np.mean(scores), (image, scores))

    @classmethod
    def compare_traces(cls, gold, experimental):
        """
        mean sum of distances (MSD) method of Li et al. (2005)
        $MSD(U,V) = \frac{1}{2n}\Big(\sum\limits_{i=1}^{n}\min\limits_{j}\mid{v_i - u_j}\mid + \sum\limits_{i=1}^{n}\min\limits_{j}\mid{u_i - v_j}\mid \Big)$
        """
        assert(gold.image == experimental.image)
        image = gold.image
        g_coords = gold.coordinates
        e_coords = experimental.coordinates
        gold_x, gold_y = zip(*g_coords)
        exper_x, exper_y = zip(*e_coords)

        #find all the inter-point distances
        distances = np.zeros((len(gold_x), len(exper_x)))
        # for each point in gold trace...
        for j in range(len(gold_x)):
            # for each point in experimental trace...
            for k in range(len(exper_x)):
                distances[j,k] = np.sqrt((gold_x[j] - exper_x[k])**2 + (gold_y[j] - exper_y[k])**2)

        #find minimum dist for each point
        min_dist_j = np.amin(distances, axis=0)
        min_dist_k = np.amin(distances, axis=1)
        mean_dist_j = np.mean(min_dist_j)
        mean_dist_k = np.mean(min_dist_k)
        md = np.mean([min_dist_j, min_dist_k])
        #print("{image}:\t{msd}".format(image=image, msd=md))
        return (image, md)
