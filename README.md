# Graph-Based Mobility Management:
This project demonstrates how to build and leverage a Knowledge Graph (KG) as a digital twin that is both human- and machine-readable. The KG is expressed in RDF and constructed according to a predetermined ontology, with RDF triples aligned to its semantics. We then apply Graph Neural Networks (GNNs) for a link prediction task to generate feature-rich embedding vectors. These embeddings serve as a compact, expressive representation that can be used to train downstream, use-case-specific models—ranging from policy learning in telecom networks to sequence models like RNNs.
The KG includes nodes for features such as data rates and RSRQ values, elevating raw real-valued measurements into a structured RDF representation. Once captured in the KG, these values are “lifted” into the model space via GNNs, transforming each node into a unique, feature-rich embedding vector. These embeddings can then be fed into a variety of models depending on the application.

The repository includes:  

1. A KG defined by a clear ontology expressed in RDF turtle format. [Knowledge Graph](./training/graphs/ibn_demo.ttl)
2. A GNN pipeline for link prediction and node embeddings. [GNN Pipeline](./training/):  
    1. Uses Convolutional Encoder [ConvE](https://github.com/TimDettmers/ConvE) model from Tim Dettmers. Model's source code can be found [here](./training/src/model.py)
    2. The resulting ConvE model along with embedding vectors are located in [conve-0.1.pth](./training/models/conve-0.1.pth)
3. A simple baseline model (a stack of linear layers) that learns a policy from embeddings:  
    1. The model's source code can be found [here](./training/src/model.py)
    2. The trained models based on a certain threshold value can be found [here](./training/models/).

This repository offers a practical, end-to-end path from raw domain data to structured knowledge and trainable machine learning representations.

# How to run inference:
1. Create a conda environment. Please note that the environment by default loads PyTorch library for CPU:
> conda env create --file=inference-environment.yml
2. Activate the conda environment:
> conda activate gb-mm
3. Change into the inference folder and execute the model:
> cd inference
> python main.py

# Contact Information

If you have any questions please create an issue or you can reach me via [e-mail](mailto:varun.gowtham@mailbox.org).