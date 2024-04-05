# Efficient GPT report comparison generation

## Description

Briefly describe what your project does and its purpose.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed the latest version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
- You have a Linux/Mac machine. This was tested on Mac but should work on other operating systems as well.

## Conda environment installation


Run the following command to install the conda package 

    conda env create -f nlp.yml

Now activate the conda package
    
    conda activate nlp

## Running the code

Use the following command to get the similarity matches between the two texts

    python comp.py --path1 txts/rule7-nscc.txt --path2 txts/rule7-sec.txt --out_path csvs/similar_sentences_rule7.csv 

Include the `--vectorize` option if the code needs to be vectorized. This will use [FAISS](https://github.com/facebookresearch/faiss/) to improve performance of the cosine similarity function.

Now to get the gpt response use the following command

    python generate_comparison.py --ifile csvs/similar_sentences_rule7.csv --ofile csvs/similar_sentences_rule7_generations.csv

To repeat the above process for rule 19 use the `rule19-nscc.txt` and `rule19-sec.txt` files

