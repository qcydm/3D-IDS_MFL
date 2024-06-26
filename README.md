#  Doubly Disentangled Few-Shot Intrusion Detection

This repository contains the implementation of a sophisticated intrusion detection system (IDS) tailored for the Internet of Things (IoT) environment. We propose 3D-IDS and MFL, the former is a novel method that aims to tackle the performance inconsistent issues through two-step feature disentanglements and a dynamic graph diffusion scheme, and the latter is a plug-and-play module few-shot intrusion detection module by multi-scale information fusion.The system is designed to identify both known and unknown attacks in encrypted network traffics, providing a frontline defense against increasingly sophisticated cyber threats.

## Key Features

- **Dynamic Graph Diffusion**: Employs a dynamic graph diffusion scheme for spatial-temporal aggregation in evolving data streams.
- **Feature Disentanglement**: Utilizes a two-step feature disentanglement process to address the issue of entangled distributions of flow features.
- **Few-Shot Learning Module**: Integrates a few-shot learning approach for effective detection of novel intrusion types with limited instance data.
- **Memory Model**: Incorporates a memory model to generate representations that highlight attack-specific features.
- **Real-Time Detection**: Capable of performing real-time anomaly detection with high accuracy and speed.

## Getting Started

### Prerequisites

- Python
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- scikit-learn
- tqdm

### Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```
For the torch-geometric library, multiple files need to be downloaded. You can visit the following website to select the appropriate version that matches your environment: (https://data.pyg.org/whl/). After downloading, you can install the specific version with the following command:

```bash
pip install torch_geometric==2.0.4 -i https://pypi.doubanio.com/simple
```

You can choose the version that best suits your needs.

### Usage

To train and evaluate the model, run the following command:

```bash
python main.py
```

This will start the training process, including the pre-training, meta-training, and meta-testing phases.

### Datasets

Benchmark Datasets
The system is trained and tested on several benchmark datasets, which are commonly used in the field of network traffic analysis and intrusion detection. These datasets provide a rich source of labeled data for training and evaluating models.

You can change datasets in main.py:
```bash
    data_all = torch.load(datasets)
```

- CIC-ToN-IoT: A dataset designed for evaluating network-based intrusion detection systems in the context of IoT networks.
- CIC-BoT-IoT: Another IoT-focused dataset that simulates botnet attacks in a network environment.
- EdgeIIoT: A dataset collected from real-world IoT devices and networks, focusing on edge computing scenarios.
- NF-UNSW-NB15-v2: A network flow dataset containing modern normal and attack traffic traces from a university network.
- NF-CSE-CIC-IDS2018-v2: A dataset that combines network flows and system logs to provide a comprehensive view of network traffic.
- NF-UQ-NIDS：A dataset from the University of Queensland for testing NIDS, featuring various network traffic scenarios.
- NF-BoT-IoT-v2：An updated dataset focusing on IoT botnet attacks, providing refined data for network security research.


### Using Your Own Data
You can also use your own data with our system by datasets.py. Please make sure that you keep the following column information：
column information = TemporalData(  
    src=src,  
    dst=dst,  
    src_layer=src_layer,  
    dst_layer=dst_layer,  
    t=t,  
    dt=dt,  
    msg=msg,  
    label=label,  
    attack=attack  
)  
  

### Performance Metrics

The effectiveness of the system is evaluated using the following metrics:

- F1 Score
- Normalized Mutual Information
- Precision and Recall


## Baselines 

we select 11 few-shot learning models to incorporate into 3D-IDS as baselines, including 3 meta-learning based models (i.e., MBase, MTL, TEG), where TEG is designed for graph-structure-based few-shotlearning, 4 augmentation-based models (i.e., CLSA,
ESPT, ICI, KSCL], where CLSA and ESPT are based on contrastive augmentation, ICI and KSCL are based on instance augmentation), 4 metric learning-based models (i.e., BSNet, CMFSL, TAD, PCWPK).


### Baseline Models for 3D-IDS

This project incorporates a series of baseline models for few-shot learning in the context of 3D Intrusion Detection Systems (3D-IDS). Below is a brief description of each model type:

#### Meta-Learning Models
- **MBase**: A fundamental meta-learning model that serves as a basic benchmark for comparison.
- **MTL**: A Multi-Task Learning approach that enhances learning efficiency by leveraging shared features across multiple tasks.
- **TEG**: Tailored for graph-structured data, this model utilizes graph embedding techniques for few-shot learning scenarios.

#### Augmentation-Based Models
- **CLSA**: Employs contrastive augmentation to enhance the model's ability to distinguish between samples.
- **ESPT**: Builds upon contrastive augmentation with different strategies for further performance optimization.
- **ICI**: Utilizes instance augmentation to increase sample diversity and improve model generalization.
- **KSCL**: Another instance augmentation-based model, with unique strategies for few-shot learning.

#### Metric Learning-Based Models
- **BSNet**: Focuses on learning sample distances to bring similar samples closer and dissimilar ones further apart in feature space.
- **CMFSL**: A specialized metric learning model that include innovative feature extraction or similarity measurement techniques.
- **TAD**: Involves adaptive distance learning to better fit the few-shot learning context.
- **PCWPK**: Based on pairwise or contrastive learning approaches to improve accuracy in few-shot scenarios.

#### How to Test Baseline Models
To test any of the baseline models, you can run the following command in your terminal, replacing the method name accordingly:
```bash
python [Method].py
```
For example, to test the ICI model, you would use:
```bash
python ICI.py
```
