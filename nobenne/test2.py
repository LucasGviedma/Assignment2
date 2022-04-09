from __future__ import print_function, division
import numpy as np
from metric import NDCGScorer, NDCGScorer_qid, map_scorer
from adarank import AdaRank
from read_data import read_data


parser = read_data()
parser.read_ml()