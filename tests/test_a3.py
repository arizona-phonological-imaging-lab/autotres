from a3.roi import ROI
import unittest

class TestROI(unittest.TestCase):
    """
    Tests ROI behavior
    """
    def test_roi(self):
        print("ROI(1,2,3,4) == ROI([1,2,3,4]) ? => {0}".format(ROI(1,2,3,4) == ROI([1,2,3,4])))
        assert(ROI(1,2,3,4) == ROI([1,2,3,4]))



if __name__ == '__main__':
    unittest.main()
