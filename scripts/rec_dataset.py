class RecommendDataset(object):
    def __init__(self):
        pass


class UserSequenceDataset(RecommendDataset):
    def build(self):
        pass

    """
    element of this dataset
    - user_id
    - item_list
    """


class UserClickLabel(object):
    """
    element of this dataset
    - user_id
    - item_list
    - label_list
    """
    pass



"""
types of features
1. text
2. numeric
3. categorical
4. vector
5. image
6. datetime
"""