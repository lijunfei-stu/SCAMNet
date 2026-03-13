conda create -n acvnet python=3.8
conda activate acvnet
pip install -r requirements.txt

conda env create -f environment_ubuntu2404.yml
pip install -r requirements_ubuntu2404.txt 

conda deactivate
conda remove -n <环境名称> --all

执行完成conda env create -f environment_ubuntu2404.yml和pip install -r requirements_ubuntu2404.txt 之后，在运行其他代码时出现一些报错，继续执行下面代码
conda install pillow=11.1.0 -c conda-forge