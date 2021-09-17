
# install these packages individually or in sequeunce

pip install --editable ./

pip install sacrebleu==1.4.12

conda install -c pytorch faiss-gpu
pip install hydra-core --upgrade
pip install importlib_metadata

# cd ..
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

