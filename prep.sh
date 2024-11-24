mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

pip install uv

# Add SSH key to Github
# Clone the cookiecutter repo

git clone git@github.com:StephanHeijl/cookiecutter-ml-scripts.git

cd cookiecutter-ml-scripts
uv pip install -r pyproject.toml

python3 test_cuda.py