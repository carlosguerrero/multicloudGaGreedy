# Resource optimization of container orchestration: a case study in multi-cloud microservices-based applications

This program has been implemented for the research presented in the article "Resource optimization of container orchestration: a case study in multi-cloud microservices-based applications", accepted for publication in the "Journal of Supercomputing".


This a NSGA-II algorithm implementation in python 2.7, considering the GA settings explained in the article. For more details, please, read the article in https://doi.org/10.1007/s10723-018-9432-8

This program is released under the GPLv3 License.

**Please consider to cite this work as**:

```bash

@article{guerrero_multicloud,
	title = {Resource optimization of container orchestration: a case study in multi-cloud microservices-based applications},
	volume = {74},
	copyright = {All rights reserved},
	issn = {1573-0484},
	doi = {https://doi.org/10.1007/s10723-018-9432-8},
	abstract = {An approach to optimize the deployment of microservices-based applications using containers in multi-cloud architectures is presented. The optimization objectives are three: cloud service cost, network latency among microservices, and time to start a new microservice when a provider becomes unavailable. The decision variables are: the scale level of the microservices; their allocation in the virtual machines; the provider and virtual machine type selection; and the number of virtual machines. The experiments compare the optimization results between a Greedy First-Fit and a Non-dominated Sorting Genetic Algorithm II (NSGA-II). NSGA-II with a two-point crossover operator and three mutation operators obtained an overall improvement of 300\% in regard to the greedy algorithm.},
	journal = {Journal of Supercomputing},
	author = {Guerrero, Carlos and Lera, Isaac and Juiz, Carlos},
	month = jul,
	year = {2018},
	keywords = {Resource management, Genetic algorithm,Multi-objective optimization,Replica placement, MapReduce scheduling, Hadoop},
	pages = {2956–-2983}
}
```

**Execution of the program**:

```bash
    python mainGA.py
```

**Acknowledgment**:

This research was supported by the Spanish Government (Agencia Estatal de Investigación) and the European Commission (Fondo Europeo de Desarrollo Regional) through Grant Number TIN2017-88547-P (AEI/FEDER, UE).
