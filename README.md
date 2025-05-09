# Bomb fishing classifier
## Set up env on linux with:
```
cd ~/bomb_fishing/set_up
conda env create -f linux_env.yml -n conda-bomb-env
conda activate conda-bomb-env
```

## To run inference, adjust the filepaths at the top of ~/bomb_fishing/code/config.py and run:
```
cd ~/bomb_fishing/code
python -m scripts.batch_runner
```

## Create an env for open soundscape
```
conda create -n open_soundscape_env \
  python=3.9 \
  jupyterlab \
  geopandas \
  contextily \
  scipy \
  numpy \
  pandas \
  shapely \
  -c conda-forge

conda activate open_soundscape_env
pip install opensoundscape
conda install ipykernel notebook jupyter -y

# Make kernel visible to jupyter notebooks
python -m ipykernel install --user --name open_soundscape_env --display-name "Python (open_soundscape_env)"
```


