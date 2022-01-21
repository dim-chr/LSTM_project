# LSTM_project
Time-Series Forecasting: Predicting Stock Prices Using An LSTM Model

### How to execute:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://google.com)<br />

or can execute with a default model by using the command:<br />
Question A: `$python forecast.py –d dir/nasdaq2007_17.csv -n <integer>`<br />
Question B: `$python detect.py –d dir/nasdaq2007_17.csv -n <integer> -mae <double threshold>`<br />
Question C: `$python reduce.py –d dir/nasdaq2007_17.csv -q <queryset> -od <output_dataset_file> -oq <output_query_file>`<br />

you can also specify spesific trained model by using the command:<br />
Question A: `$python forecast.py –d dir/nasdaq2007_17.csv -n <integer> -m /path/model.h5`<br />
Question B: `$python detect.py –d dir/nasdaq2007_17.csv -n <integer> -mae <double threshold> -m /path/model.h5`<br />
Question C: `$python reduce.py –d dir/nasdaq2007_17.csv -q <queryset> -od <output_dataset_file> -oq <output_query_file> -m /path/model.h5`

### Results 

Inside `dir/exports/` you will find: 
- a result.pdf file for each question, analyzing the results for specific arguments.
- the exported plots for each question.