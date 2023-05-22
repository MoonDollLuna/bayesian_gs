import pandas
import numpy as np
from pgmpy.estimators import BDeuScore

data = pandas.DataFrame({"A": ["a", "a", "a", "a", "b", "b", "b", "b", "a", "b"],
                         "B": ["A", "A", "B", "B", "A", "A", "B", "B", "B", "A"],
                         "C": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"]})

bdeu_scorer = BDeuScore(data)
score = bdeu_scorer.local_score("A", ["B", "C"])
print(score)
