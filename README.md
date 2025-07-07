# Thesis Dataset

This repository contains the data and results of my thesis on the use of the Large Language Model (LLM), integrated with a Retrieval-Augmented Generation (RAG), for the detection of architectural smells in microservice-based applications.

## Structure of the repository

- **prompt 1/**: It contains the first prompt used to query the model and, for each project, the RAG system codes and knowledge base.

- **prompt 2/**: It contains the second prompt used to query the model and, for each project, the RAG system codes and knowledge base.

- **result/**: It contains the results obtained from the interaction between the LLM model and each project. Inside:

  - **prompt 1/**: For each project, there are the results generated with prompt 1.
  - **prompt 2/**: For each project, there are the results generated with prompt 2.

## Analysed projects

The open-source projects used for experimentation are:

- **[cinema-microservice](https://github.com/Crizstian/cinema-microservice)**

- **[event-sourcing-examples](https://github.com/cer/event-sourcing-examples)**
