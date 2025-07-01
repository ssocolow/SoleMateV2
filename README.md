<!-- PROJECT LOGO -->
<div align="center">
  <a href="https://github.com/saatvikkher/SoleMate/blob/main/logo.png">
    <img src="static/logo.png" alt="Logo" width="120" height="120">
  </a>
  <h1 align="center">SoleMate</h1>
</div>

Official implementation of *"Improving and Evaluating Machine Learning Methods for Forensic Shoeprint Matching"*.

Try it out at [solematev2.streamlit.app](https://solematev2.streamlit.app/)

<!-- 
### Deliverables
- Extract edges from a shoeprint
- Novel Iterative Closest Point (ICP) implementation for improved alignment
- Calculate Similarity Metrics to assess alignment
 -->
 
## Getting Started

### Prerequisites

Before you begin, make sure you have the following software installed:

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ssocolow/SoleMateV2.git
   ```
2. Install required dependencies
   ```sh
   conda create --name solemate --file requirements.txt
   ```
3. Activate the environment
   ```sh
   conda activate solemate
   ```
## Usage

Create Soles
```python
# border_width=160 removes the border rulers from prints taken from the Everspry Outsole Scanner
Q = Sole("path/to/image", border_width=160) 
K = Sole("path/to/image", border_width=160)

Q.plot()
K.plot()
```
Create a SolePair
```python
pair = SolePair(Q, K, mated=True)
pair.plot()
```

Align a SolePair
```python
sc = SolePairCompare(pair, 
                     icp_downsample_rates=[0.05],
                     shift_up=True,
                     shift_down=True,
                     shift_left=True,
                     shift_right=True,
                     two_way=True) # icp is called here
pair.plot(aligned=True)
```
Generate metrics
```python
sc.min_dist() # Calculate Euclidean Distance metrics
sc.percent_overlap() # Calculate Percent Overlap metrics
sc.pc_metrics() # Calculate Phase-correlation metrics such as peak value, MSE, correlation coefficient
sc.jaccard_index() # Jaccard similarity coefficient
sc.cluster_metrics() # Clustering-based metrics
sc.pc_metrics() # Phase Correlation metrics
```


*Developed and maintained by Simon Angoluan, Divij Jain, Saatvik Kher, Lena Liang, Yufeng Wu, Ashley Zheng, Simon Socolow, and Myer Liebman.*

*We conducted our research in collaboration with the [Center for Statistics and Applications in Forensic Evidence](https://forensicstats.org/).*


