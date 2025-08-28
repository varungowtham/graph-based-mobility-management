import json

# Load a JSON file to memory
def loadJSONFromFile(filePath: str):
    with open(filePath) as f:
        d = json.load(f)
        d = json.loads(d)
        return d