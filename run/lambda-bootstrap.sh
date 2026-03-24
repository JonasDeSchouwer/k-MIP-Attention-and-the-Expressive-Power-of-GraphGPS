# Bootstrapping the environment on Lambda Cloud Compute

# Install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc

# Install environment
CONDA_OVERWRITE_FILES=1 conda env create -f $(dirname "$0")/../environment.full.yml -n kmipattn
source ~/miniconda3/bin/activate
conda activate kmipattn

# Set WANDB_API_KEY based on what it can find in jonas-fs/WANDB_API_KEY.txt
# Check if WANDB_API_KEY file exists
if [ ! -f "jonas-fs/access_keys/WANDB_API_KEY.txt" ]; then
    echo "Error: WANDB API key file not found at jonas-fs/access_keys/WANDB_API_KEY.txt"
    exit 1
fi

export WANDB_API_KEY=$(cat jonas-fs/access_keys/WANDB_API_KEY.txt)
