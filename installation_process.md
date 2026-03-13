conda create -n acvnet python=3.8
conda activate acvnet
pip install -r requirements.txt

conda deactivate
conda remove -n <环境名称> --all