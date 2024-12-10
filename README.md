# NanoLoop
A Dual-Modality Framework Leveraging Nanopore Sequencing for Chromatin Interaction Prediction

## Installation
**1. Install FusNet**
```
git clone https://github.com/bioinfomaticsCSU/NanoLoop.git
# Using NanoLoop as the root directory for program execution
cd NanoLoop
```
**2. Create an environment**
```
# create a new enviroment
conda env create -f environment.yml --name NanoLoop
# activate
conda activate NanoLoop
```
**3. Create directory**
```
mkdir logs
mkdir out_dir
```
## Get demo data

**1.Loop**

The loop data of HG001-HG004 has been placed in the data/directory

**2.Methylation**

The WGBS data of HG001-HG004 can be obtained from NCBI (GSE186383)

**3.Nanopore assembly data**

The assembly data of HG001 (NA12878) and HG002 can be obtained from https://zenodo.org/records/5228989

## Usage
**1.Generate negative samples**
```
python preprocess/generate_neg_loop.py \
    --input_path data/HG001/HG001_pos.bedpe \
    --chrom_sizes_path data/hg38.chrom.sizes \
    --anchor_len 5000 \
    --extend_len 500 \
    --sample_proportion 1 \
    --output_path out_dir/HG001_neg.bedpe
```
**2.Generate input data**
```
python generate_input.py \
    --pos_loop data/HG001/HG001_pos.bedpe \
    --neg_loop out_dir/HG001_neg.bedpe \
    --assembly_data data/assembly_data/HG001_assembly.fa \
    --methylation_data data/HG001/GSM5649420_TrueMethylBS_HG001_LAB01_REP01.bedGraph \
    --name HG001_5k_balance \
    --save_dir out_dir/ \
    --test_balance True
```
**3.Training model**
```
python train.py \
    --data_pre out_dir/HG001_5k_balance \
    --model_name HG001_5k_balance \
    --model_dir out_dir \
    --epochs 40
```
**4.Predicting loop**
```
python predict.py \
    --data_pre out_dir/HG001_5k_balance \
    --model_dir out_dir \
    --model_name HG001_5k_balance \
    --result_path out_dir/HG001_5k_balance_test_probs.txt \
    --save_feature False
```
