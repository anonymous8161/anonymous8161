anonymous#8161 

This repository contains the main experimental code for our tongue analysis project. Please note that the code for GroundingDINO and SAM is not included here.

## Repository Structure

1. **transfg**: 
   This directory contains our reproduced implementation of the TransFG network.

2. **upright_orientation**:
   Here you'll find the code for our tongue upright orientation process.

3. **edge_body_separation**:
   This folder includes the code used to separate the edge region and body region of the uprighted tongue images.

4. **tongue**:
   This is our main experimental code directory, containing:
   - Various baseline networks
   - Our SignNet implementation (tongue/run_tongue.py)
   - Configuration files (tongue/config)
   - Evaluation code (tongue/metrics and tongue; tongue/roc)
   - Miscellaneous code (tongue/others)
