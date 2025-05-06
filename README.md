# Bomb fishing classifier
## Set up env on linux with:
cd ~/bomb_fishing/set_up
conda env create -f linux_env.yml -n conda-bomb-env
conda activate conda-bomb-env

## To run inference, adjust the filepaths at the top of ~/bomb_fishing/code/config.py and run:
cd ~/bomb_fishing/code
python -m scripts.batch_runner


