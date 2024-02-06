## Project setup
### Requirements
- Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Setup Environment
1. Create folder `data` and insert data there.
2. Copy `env.sample.yaml` and rename it to the `env.yaml`.
3. Set the variable `PLOTS_PATH` as an absolute path to the plots directory.
4. Set the variable `DATA_PATH` as an absolute path to the data directory.

```sh
conda env create -f environment.lock.yml
conda activate hydro-power-plant-prediction
pip install -e .
```

## Running the pipeline
See [Ploomber user guide](https://docs.ploomber.io/en/latest/user-guide/index.html).

```sh
ploomber build
```

```sh
ploomber plot 
```

```sh
ploomber task name_of_the_task 
```
### Insert task into pipeline
```sh
1. Insert task into pipeline.yaml file.

2. ploomber scaffold

3. ploomber nb --inject
```
