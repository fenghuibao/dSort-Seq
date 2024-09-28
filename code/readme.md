## Deep-learning-assisted Sort-Seq

### Four types of data are needed to calculate the expression mean and noise, which are:

#### 1. Binned distribution

The proportion of each variant across bins, $P_{ik}$: $\sum_k P_{ik} = 1$. 

#### 2. Sorting boundaries

The sorting boundaries of the Sort-Seq experiment, $(b_1, b_2, ...b_{K-1})$. 

#### 3. Overall distribution

The overall fluorescence intensity distribution of the library, obtained via flow cytometry assay of the whole cell library, generally 20 $\times$ coverage is required. 

#### 4. Mixing coefficient

The proportion of each variant in the library, obtained via NGS of the library.

### To execute the dSort-Seq program in the terminal, use the following command:

python dSort-Seq.py -p Mixing_coefficient.csv -f Binned_distribution.csv -d Overall_distribution.csv -b Sorting_boundaries.csv


### Reference
Feng, Huibao, et al. "Deep-learning–assisted Sort-Seq enables high-throughput profiling of gene expression characteristics with high precision." Science Advances 9.45 (2023): eadg5296.
