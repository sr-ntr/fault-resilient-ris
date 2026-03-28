# RIS Fault Detection (Deep Learning)

A PyTorch-based implementation of a deep neural network for detecting faulty pixels in a Reconfigurable Intelligent Surface (RIS).

## Overview
- Input: 45 features  
- Output: 100 binary labels (10×10 RIS grid)  
- Task: Multi-label classification (predict failed pixels)  
- Handles class imbalance (~74.5% healthy pixels)

##Reference

Based on RIS fault detection methodology described in the paper "Fault-Resilient RIS Systems: Toward SINR- and Sum-Rate Maximization of IoT Networks Using ML Frameworks"(Section V-A).
