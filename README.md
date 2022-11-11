# JBG060

## Initial setup
Before you install the needed libraries it is first good to create a new anaconda environment in order not to create any conflicts with other libraries and files. If you have istalled Anaconda creating a new anvironment is extremely easy - open anaconda prompt and write:
```
conda create -n my-env
```
Then activate the enviroment:
```
conda activate my-env
```
Now in your new enviroment you can install the needed libraries. Simply paste these comands in the anaconda comand prompt:
```
conda install -c conda-forge statsmodels
pip install -U scikit-learn
pip install dash
pip install pandas
pip install plotly
pip install numpy
pip install matplotlib
```
## Run the code
Again in the anaconda comand prompt navigate to the folder where you have cloned the repo with:
```
cd path/to/folder
```
And then simply type:
```
python main.py
```
## Results
