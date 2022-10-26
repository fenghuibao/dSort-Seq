## The upload file format requirements for the dSort-Seq server (http://www.thu-big.net/dsort-seq/).

### Four files are needed to calculate the expression mean and noise, which are:

#### 1. Binned distribution (.txt, .csv, .xls, .xlsx)

The proportion of each variant across bins, $P_{ik}$: $\sum_k P_{ik} = 1$. Please note that to ensure the privacy of your result, you do not need to provide the sequence of each variant. For an example, see example/Binned_distribution.csv.

#### 2. Sorting boundaries (.txt, .csv, .xls, .xlsx)

The sorting boundaries of the Sort-Seq experiment, $(b_1, b_2, ...b_{K-1})$. For an example, see example/Sorting_boundaries.csv.

#### 3. Overall distribution (.txt, .csv, .xls, .xlsx)

The overall fluorescence intensity distribution of the library, obtained via flow cytometry assay of the whole cell library, generally 20 x coverage is required. For an example, see example/Overall_distribution.csv.

#### 4. Mixing coefficients (.txt, .csv, .xls, .xlsx)

The proportion of each variant in the library, obtained via NGS of the library. For an example, see example/Mixing_coefficients.
