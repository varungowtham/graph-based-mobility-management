from typing import Dict, Tuple, List, Set
from dataclasses import dataclass

from functools import reduce
import csv
from rdflib import Graph
from rdflib.term import Node, URIRef
from src.utils import getGraph, loadJSONFromFile, roundOffFloatToNearestHalf
from functools import partial

def construcLabel(prefix: str, val: str) -> str:
    return "http://purl.org/toco/" + prefix + "_" + val + "_dB"