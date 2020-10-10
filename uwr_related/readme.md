# Updated build instructions

1. Install miniconda

2. Create the conda env
    1. Install from the yaml:
        ```
        conda env create -f environment.yml
        ```
    2. If and only if the above fails, try installing the packages:
        ```
        # Install conda somewhere...
        conda create -n 202010-fairseq-common -c conda-forge -c pytorch -c plotly -c nvidia python=3.7 plotly black click scikit-learn jupyter jupyterlab ipympl dvc plotnine seaborn dtale isort=4.3.21 more-itertools nbdime pyflakes pytest pylint mypy graphviz flake8 dill cython autopep8 nltk mkdocs flask sknw graphviz flake8-docstrings folium pydocstyle tqdm soupsieve docopt lxml jupytext papermill absl-py beautifulsoup4 scipy numba scikit-image scikit-fuzzy pandas-flavor pandas-profiling pytest-cov ptvsd great-expectations requests-cache apsw qgrid texttable pytorch torchvision cudatoolkit=10.1 cudatoolkit-dev=10.1 virtualenv nccl cffi cython dataclasses editdistance hydra-core regex sacrebleu tqdm pandas py-opencv
        ```

3. Activate then env: `conda activate 202010-fairseq-common`
    
4. Install APEX
    ```
    git clone https://github.com/NVIDIA/apex
    export TORCH_CUDA_ARCH_LIST="5.2;6.0;6.1;6.2;7.0;7.5"
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
    --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
    --global-option="--fast_multihead_attn" ./
    cd ..
    ```

5. Clone and install fairseq
    ```
    git clone https://github.com/chorowski-lab/fairseq
    cd fairseq
    pip install --editable ./
    cd ..
    ```

6. Download data. You can also `ln -s /pio/scratch/2/jch/wav2vec/data data`
    ```
    mkdir data
    cd data
    # Fetch part of librispeech
    wget http://www.openslr.org/resources/12/dev-clean.tar.gz
    tar zxvf dev-clean.tar.gz 

    # Fetch scribblelens
    mkdir scribblelens
    cd scribblelens
    url=http://www.openslr.org/resources/84/scribblelens.corpus.v1.2.zip
    wget $url
    7z e scribblelens.corpus.v1.2.zip \
        -o./ scribblelens.corpus.v1/corpora/scribblelens.paths.1.4b.zip
    ```

7. check if training works:
    - Audio:

        From the top of `fairseq` repo call: `bash uwr_related/test_cmd_audio.sh` and verify that the model _starts_ training.
    
    - Scribblelens:
        
        TODO
