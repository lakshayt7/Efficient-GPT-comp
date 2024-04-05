# Efficient GPT report comparison generation

## Description

This repository has code for generating comparison of two texts after finding similar portions using similar search
## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed the latest version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
- You have a Linux/Mac machine. This was tested on Mac but should work on other operating systems as well.

## Conda environment installation


Run the following command to install the conda package 

    conda env create -f nlp.yml

Now activate the conda package
    
    conda activate nlp

## Set up GPT API key

paste the following command at the top of your bashrc file

    export OPENAI_API_KEY='<key-here>'

## Running the code

Use the following command to get the similarity matches between the two texts

    python comp_vector.py --path1 txts/rule7-nscc.txt --path2 txts/rule7-sec.txt --out_path csvs/similar_sentences_rule7.csv 

Include the `--vectorize` option if the code needs to be vectorized. This will use [FAISS](https://github.com/facebookresearch/faiss/) to improve performance of the cosine similarity function. Note that the code may return a segmentation error if their is not enough memory to load the sentence transformer model.

Now to get the gpt response use the following command

    python generate_comparison.py --ifile csvs/similar_sentences_rule7.csv --ofile csvs/similar_sentences_rule7_generations.csv

To repeat the above process for rule 19 use the `rule19-nscc.txt` and `rule19-sec.txt` files

