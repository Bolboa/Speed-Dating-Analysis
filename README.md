# Speed-Dating-Analysis
## Description
The aim of this project was to get the best match accuracy of a participant in a speed dating event. We want to be able to predict the likeliness that a participant will find a match by using learning models on the data set. The data set is unclean so a lot of preprocessing was necessary in order to achieve the highest accuracy possible in the models used.

## Preprocessing
Very little information is given concerning the data set, all we know is what each column represents, but nothing about the data type of each value or how many missing values there are. The first step was to ensure that all columns hold the same data type.

* **Step 1 - Map Similar String Values to a Unique Integer**
  * There were two columns that contained solely string values, these needed to be completely encoded. However before doing so, all *NaN* values were first converted to string format so that empty values will be encoded as well. This is better than replacing them with a mean value since each integer is a representation of a string, and not the result of some calculation.
  * The *from* column contained only strings. Scikit Learn's *Label Encoder* was implemented to convert all similar strings to a unique number after ensuring that all whitespace was removed and each string was converted to uppercase. 
  * The *zipcode* column, athough it contained integer values, the integer values were enormous because each zipcode was interpreted as a number. To save some unescessary computation, each value was converted to string format and then the *Label Encoder* was used to encode similar zipcodes. This works since many users were coming from roughly the same place, so there were a significant amount of repeating values.
  * There were two columns, *career* and *career_c*, the former contains strings and the latter represents each string's numerical equivalent. Clearly, *career* is redundant, however, *career_c* contained some values that were empty but that were not empty at the same position in the *career* column. I encoded these strings by filtering by certain keywords and grouping the careers that belong in the same category together. After encoding these strings, the empty values were replaced with the necessary encoding.

* **Step 2 - Clustering Model**
  * It is pretty obvious from looking at the data that some columns must be redundant, whereas other ones must be strongly correlated to the target column we are trying to predict. I decided to look at some of the columns I believe to be strongly correlated to the target column to prove that this was true. I decided to graph the match rate of each participant in relation to their age and gender. I theorized that the youngest of females had higher match rates than the yougest of males, since women generally are more accepting to dating someone older than them.
  
  ![Alt text](/img/data_graph.png)

  * From the graph, it appears that I am correct thus far. *1* represents male and *0* represents femal. It seems that 19 and 20 year old women have a higher match rate than their male counterparts. I took this idea one step further and decided to factor in the age of the participant's partner as well.
