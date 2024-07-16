Read before using model:
- Requires Python 3.11

- Save 'Adaptive Model' in same file location as dataset

- If entry field is unused, leave '' or None - this is only supported for QuickSearch, focus_metric, low_memory. force_features leave as [] if unused

- File names must include extensions - supports .csv and .xlsx

- use_cols is used to specify certain columns in the dataset. Useful if certain columns are definitely redundant e.g. Customer ID, row number. Written in the format 'C:G, H, I, Z:AI' for example. Columns C through G are used, H and I are used and Z through AI inclusive are used. NOTE: No other columns are used when use_cols is specified

- All input variables i.e. for force_features must be written in the exact same way they are typed in dataset and typed in the format, ['feature1', 'feature2'...,'featuren']

- QuickSearch must be set to 'Y' for faster, less optimised search. Any other entry will default to optimised approach.

- focus_metric field should be left empty if unsure of what different metrics optimise. Only used if QuickSearch is not on. Defaults to f1 for classification and r2 for regression.

- low_memory support for less powerful computers. Caps the number of rows the model uses to 10000 - this may affect model performance.

Tip:
- Any file produced by model is in the same location

- (File name) Modelling.xlsx provides the correlation matrix, the feature importance values and the information for the graphs in tabular form.

- If a certain dataset is updated as time goes on and only one variable is of particular interest, save a copy of the algorithm for this dataset. If only new rows are added, the model does not need to be opened in a coding environment after the first time - each time it is run it accounts for the introduction of new rows. 

- If both algorithms perform poorly in evaluation, introduce new variables or adjust use_cols.

- Datasets in .xlsx file format must be closed for the file to run. .csv files need not be closed.

