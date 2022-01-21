# LSTM_project
Time-Series Forecasting: Predicting Stock Prices Using An LSTM Model

### Project Structure and Info:

- You can find the python code for each Question inside the  `src/` directory.
- Inside the `dir/` directory you can find the dataset .csv file (input file).
- Inside `dir/exports/` you will find: 
 - a report.pdf file, analyzing the results for all trained models.
 - the exported plots for each question.

### How to train your model:

Question A - For each time series:
[![Open Question A1 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U7BL5D9G0btYpWmswPgVWk59AUj-jMUy?usp=sharing)<br />
Question A - For all time series:
[![Open Question A2 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fZrYgKbhb7ZEABkpXEWWelFMav5sFwJ5?usp=sharing)<br />
Question Β:
[![Open Question B In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1COqg6tsL6WM_Eye2Yv84o0y9hdEkNIEZ?usp=sharing)<br />
Question C:
[![Open Question C In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LBrn2DlUjxAugzAXNAWtZ2Gs_PuOl3po?usp=sharing)<br />

### How to execute:

or can execute with a default model by using the command:<br />
Question A: `$python forecast.py –d nasdaq2007_17.csv -n 10`<br />
Question B: `$python detect.py –d nasdaq2007_17.csv -n 10 -mae 0.09`<br />
Question C: `$python reduce.py –d input_file.csv -q query_file -od out_input.csv -oq out_query.csv`<br />

### Notes:

We have created two variations of Question A. Inside `src/Question A` there are two files, `forecast_all.py` and `forecast.py`. 
- `forecast_all.py` predicts using one model that was trained by using all the time series.
- `forecast.py` predicts using multiple models. Each model was trained by using individual time series.