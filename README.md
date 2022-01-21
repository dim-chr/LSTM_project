# LSTM_project
Time-Series Forecasting: Predicting Stock Prices Using An LSTM Model

### How to execute:

You can execute with a default model by using the command:
Question A: `$python forecast.py –d dir/nasdaq2007_17.csv -n <integer>`
Question B: `$python detect.py –d dir/nasdaq2007_17.csv -n <integer> -mae <double threshold>`
Question C: `$python detect.py –d dir/nasdaq2007_17.csv -n <integer> -mae <double threshold>`

or you can specifiy spesific trained model by using the command:
Question A: `$python forecast.py –d dir/nasdaq2007_17.csv -n <integer> -m /path/model.h5`
Question B: `$python detect.py –d dir/nasdaq2007_17.csv -n <integer> -mae <double threshold> -m /path/model.h5`
Question C: `$python detect.py –d dir/nasdaq2007_17.csv -n <integer> -mae <double threshold> -m /path/model.h5`

### Results 

Inside `dir/exports/` you will find: 
- a result.pdf file for each question, analyzing the results with spesific arguments given.
- the exported plots for each question. 