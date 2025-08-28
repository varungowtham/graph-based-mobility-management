from typing import Dict, Tuple, List, Set
from dataclasses import dataclass

from functools import reduce
import csv
from rdflib import Graph
from rdflib.term import Node, URIRef

import dataclasses, json

# Load a JSON file to memory
def loadJSONFromFile(filePath: str):
    with open(filePath) as f:
        d = json.load(f)
        d = json.loads(d)
        return d
 
# Rounds off a floating point number to the nearest 0.5   
def roundOffFloatToNearestHalf(number):
    return (round(number * 2)) / 2

# Class to help encode a dataclass into JSON
class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

# Get an RDF graph on file system into the program in the form of RDFLib graph.
def getGraph(filePath: str) -> Graph:
    g = Graph()
    g.parse(filePath)
    return g

# Create a list of zeros of specific length
def zeroListMaker(n: int) -> List[int]:
    return [0] * n

# This class holds the graph properties.
# Each element and relation is given a specific index
# Holds N matching for a (E,R) pair
@dataclass
class GraphProperties:
    elemLabelToIndexDict: Dict[str, int]
    relLabelToIndexDict: Dict[str, int]
    labeledE2MultiVector: Dict[Tuple[str, str], List[int]]
    numElem: int
    numRel: int

# Used in conjuction with reduce to assign each element of a graph 
# their specific index values
def createIndexMap(acc: Tuple[Dict[str, int], int], 
                   node: str) -> Tuple[Dict[str, int], int]:
    res = acc[0]
    index = acc[1]
    res[node] = index
    return (res, index+1)

# Generates a set of elem,rel combination
def generateElemRelCombination(
    elems: Set[str], rels: Set[str]
    ) -> Set[Tuple[str, str]]:
    combination = set()
    for elem in elems:
        for rel in rels:
            combination.add((elem, rel))
    return combination

# Generates GraphProperties class.
def generateGraphProperties(gr: Graph) -> GraphProperties:
    elems = set(gr.all_nodes())
    rels = set(gr.predicates())
    
    elemLabels = set(map(lambda x: x.__str__(), list(elems)))
    relLabels = set(map(lambda x: x.__str__(), list(rels)))
    
    numElems = len(elems)
    numRels = len(rels)
    
    numElemZeros = zeroListMaker(numElems)
    
    elemToIndexDict, _ = reduce (createIndexMap, elemLabels, (dict(), 0))
    reltoIndexDict, _ = reduce (createIndexMap, relLabels, (dict(), 0))
    
    elemRelCombination = generateElemRelCombination(elemLabels, relLabels)
    
    e1Rele2MultiVectorList: Dict[Tuple[str, str], List[int]] = dict()
    
    for elemRel in elemRelCombination:
        e1Rele2MultiVectorList[elemRel] = numElemZeros
        
    for subj, pred, obj in gr:
        subjStr = subj.__str__()
        predStr = pred.__str__()
        objStr = obj.__str__()
        e1Rele2MultiVectorList[(subjStr, predStr)][elemToIndexDict[objStr]] = 1
    
    return GraphProperties(elemToIndexDict, reltoIndexDict, e1Rele2MultiVectorList, numElems, numRels)

@dataclass
class GraphIndexProperties:
    elemLabelsToIndexMap: Dict[str, int]
    relLabelsToIndexMap: Dict[str, int]
    numElem: int
    numRel: int
    
def generateGraphIndexProperties(grProperties: GraphProperties) -> GraphIndexProperties:
    elemLabToIndexMap = grProperties.elemLabelToIndexDict
    elemStringToIndexMap = dict(
        map(lambda kv: (kv[0].__str__(), kv[1]), elemLabToIndexMap.items())
    )

    relLabToIndexMap = grProperties.relLabelToIndexDict
    relStringToIndexMap = dict(
        map(lambda kv: (kv[0].__str__(), kv[1]), relLabToIndexMap.items())
    )

    numElems = grProperties.numElem
    numRels = grProperties.numRel

    return GraphIndexProperties(
        elemStringToIndexMap, relStringToIndexMap, numElems, numRels
    )