# 🌍 Direct Air Capture of CO₂ using MOFs  
**AtomicWizards – CHE1148 Project**  
Rija Ansari · Uyen (Emma) Hua · Miguel Freyermuth  
GNN Model on OPENDAC2025
---

## Project Overview

This project develops machine learning models to predict **CO₂ adsorption energy** in metal–organic frameworks (MOFs) for **Direct Air Capture (DAC)** applications.

MOFs are porous crystalline materials composed of metal nodes and organic linkers with extremely high surface area and tunable chemistry. Their structural flexibility makes them promising candidates for gas capture and separation. However, experimental screening of MOFs is expensive and time-intensive.  

Our goal is to build predictive models that accelerate materials discovery through high-throughput computational screening.

---

## Objective

Predict **DFT-corrected adsorption energy (eV)** of CO₂ in MOFs using structural and chemical information.

We aim to:

- Identify chemical interactions that most strongly influence adsorption
- Achieve prediction errors comparable to experimental uncertainty
- Enable reliable ranking of candidate MOFs for DAC screening


---

##  Dataset

The full Open DAC 2025 dataset (Sriram et al., arXiv 2025).

- ~800,000 entries  
- 5,338 unique MOFs  
- Adsorption data for:
  - CO₂
  - H₂O
  - N₂
  - NO₂  

Each entry contains:

- MOF name and type  
- 3D periodic atomic structure  
- Adsorbate counts  
- Corrected DFT adsorption energy (eV)  

For this project we selected the unique MOFs candidates from the test set to provide a proof-of-concept for this study with limited compute. 

- 195 unique MOFs were shortlisted
- Adsorption data was limited to CO₂

To prevent data leakage, data splits are performed by **unique MOF structure**, not by individual rows.

---

##  Modeling Approach

### Graph Neural Network (Primary Model)

- Input: 3D atomic structure represented as a graph  
  - Nodes → atoms  
  - Edges → bonds / spatial neighbors  
- Learns local coordination and global structure  
- Outputs continuous adsorption energy (regression)  
- Optimized for MAE  

Graph Neural Networks (GNNs) are state-of-the-art for materials property prediction and high-throughput screening.

---

### Baseline Models

For comparison, we also implement:

- Random Forest (using engineered features)  
- XGBoost (using physical, chemical, and geometric descriptors)  


##  Why This Matters

Direct Air Capture is an essential climate mitigation technology. By building reliable ML ranking models, we:

- Reduce experimental costs  
- Accelerate sorbent discovery  
- Enable scalable DAC material screening  
- Support climate-tech innovation  

---

##  References

1. A. Wood, “What are Metal Organic Frameworks (MOFs)?,” Ossila, 2022.  
2. J. Park et al., “How Reproducible Are Isotherm Measurements in Metal–Organic Frameworks?,” Chemistry of Materials, 2017.  
3. A. Sriram et al., “The Open DAC 2025 Dataset for Sorbent Discovery in Direct Air Capture,” arXiv, 2025.  
