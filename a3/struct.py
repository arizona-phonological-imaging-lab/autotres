import pandas as pd


class LossRecord(object):
    """
    Keep a record of the loss during training
    """
    def __init__(self):
        self.columns = ["Epoch", "Training Loss", "Validation Loss"]
        self.data = pd.DataFrame(columns=self.columns)

    def __iadd__(self, other):
        assert(type(other) == list)
        # temporarily copy the df
        a = self.data.copy(deep=True)
        # assign the new df to self.data
        self.data = a.append(pd.DataFrame([other],columns=self.columns), ignore_index=True)
        return self
