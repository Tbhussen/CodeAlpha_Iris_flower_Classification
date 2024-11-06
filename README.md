# CodeAlpha_Iris_flower_Classification

## Brief Discription

Machine Learning Project that classifies flowers into the correct Iris type based on shape features 

## Detailed Discription

The dataset is found on Kaggle under: `Iris.csv`, contains **150** rows and **5** columns.

This project made use of the following python Libraries: `Pandas`, `Seaborn`, `Matplotlib`, and `Numpy`.

### Data Extraction

Using the `read_csv()` function from pandas, the dataset was extracted from the current working directory (CWD).

### Data Cleaning

The `id` column was dropped as it is not needed in this project. We checked for null values, and found none.

### Data Processing and Visualization

After having a look into the distribution of records over the 3 unique species, we looked at a discription of the data showing for each column:

- A `count` of the no. of records

- `mean`, average value

- `std`, standard deviation of the values

- `min`, minimum value

- `25%`, `50%`, and `75%` showing the quarter, half, and three-quarter values respectively.

- `max`, maximum value

Then, we used a `pairplot()` from seaborn library to visualize the correlation between each 2 dimentions. We further showed the correlation using the correlation matrix.

### Correlation Matrix

The `SepalWidthCm` column seemed to have the greatest correlation with the `Species` column, we suspect that this column might have gotten the greatest weight in the KNN model used after.

### Model Building 

We started by separating the features and the labels, then each were put in training or test set based on a **30%** test set scheme. The results might not be reproducible as we didn't specify a random_state parameter. As the following labelling is a categorical one, we used K-Nearest-Neighbor(KNN). We specified that there are **3** labels that the model should be able to differentiate between: _Iris-setosa_, _Iris-versicolor_, and _Iris-virginica_. The last run of the model gave a **100.0%** accuracy score. The `heatmap` reflecting the cofusion matrix, represents the distribution of correct labels achieved by the KNN model.

_Project_Done_
