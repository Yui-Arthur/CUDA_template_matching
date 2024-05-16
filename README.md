# Template Matching (PCC / SSD) with CUDA 

## Introduce
#### This project implement the PCC (Pearson Correlation Coefficient) and SSD (Sum of Square Difference) algorithm for CUDA in C

$$ PCC(X,Y)=\frac{\sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)\left(Y_{i}-\bar{Y}\right)}{\sqrt{\sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)^{2}} \sqrt{\sum_{i=1}^{n}\left(Y_{i}-\bar{Y}\right)^{2}}} $$
$$ SSD(X,Y)=\sum_{i=1}^{n}\left(X_{i}-Y_i\right)^2 $$


## Requirement
* ### `nvcc` (NVIDIA CUDA Compiler Driver) must in your system
* ### (optional) `make` for build the project

## Dataset 
### Datasets folder contain several dataset
### Each dataset need have two file, one is the target matrix `T{ID}_{rows}_{columns}`, another is the search kernel `S{ID}_{rows}_{columns}`
```sh
# example
dataset
├── 1
│   ├── S1_3_3.txt
│   └── T1_3750_4320.txt
└── 2
    ├── S2_5_5.txt
    └── T2_7750_1320.txt
```

## Usage
### make (recommend)
```bash
# only compile 
make build # both PCC and SSD
make build PCC=1 SSD=0 # only test PCC
make build PCC=0 SSD=1 # only test SSD

# compile + run PCC and SSD
make run # run with all dataset 
make run TARGET=1 # run with dataset 1

# compile + run PPC or SSD
make PCC # compile + run PCC with all dataset 
make SSD # compile + run SSD with all dataset 
make SSD TARGET=1 # compile + run SSD with dataset 1
```
### nvcc
```bash
# compile
nvcc -DPCC_TEST=1 -DSSD_TEST=1 main.cu -o main.out # test PCC and SSD
# run
./main.out {TARGET} {SEARCH}
# run with dataset 1
./main.out datasets/1/T1_3750_4320.txt datasets/1/S1_3_3.txt 
# run with script
bash test.sh 1 # run with dataset 1
```