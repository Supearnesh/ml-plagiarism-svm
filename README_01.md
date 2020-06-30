# Plagiarism Text Data

In this project, you will be tasked with building a plagiarism detector that examines a text file and performs binary classification; labeling that file as either plagiarized or not, depending on how similar the text file is when compared to a provided source text. 

The first step in working with any dataset is loading the data in and noting what information is included in the dataset. This is an important step in eventually working with this data, and knowing what kinds of features you have to work with as you transform and group the data!

So, this notebook is all about exploring the data and noting patterns about the features you are given and the distribution of data. 

> There are not any exercises or questions in this notebook, it is only meant for exploration. This notebook will not be required in your final project submission.

---

## Read in the Data

The cell below will download the necessary data and extract the files into the folder `data/`.

This data is a slightly modified version of a dataset created by Paul Clough (Information Studies) and Mark Stevenson (Computer Science), at the University of Sheffield. You can read all about the data collection and corpus, at [their university webpage](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html). 

> **Citation for data**: Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press. [Download]


```python
!wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip
!unzip data
```

    --2020-06-28 22:00:02--  https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.216.28.22
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.216.28.22|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 113826 (111K) [application/zip]
    Saving to: ‘data.zip’
    
    data.zip            100%[===================>] 111.16K  --.-KB/s    in 0.004s  
    
    2020-06-28 22:00:02 (27.4 MB/s) - ‘data.zip’ saved [113826/113826]
    
    Archive:  data.zip
       creating: data/
      inflating: data/.DS_Store          
       creating: __MACOSX/
       creating: __MACOSX/data/
      inflating: __MACOSX/data/._.DS_Store  
      inflating: data/file_information.csv  
      inflating: __MACOSX/data/._file_information.csv  
      inflating: data/g0pA_taska.txt     
      inflating: __MACOSX/data/._g0pA_taska.txt  
      inflating: data/g0pA_taskb.txt     
      inflating: __MACOSX/data/._g0pA_taskb.txt  
      inflating: data/g0pA_taskc.txt     
      inflating: __MACOSX/data/._g0pA_taskc.txt  
      inflating: data/g0pA_taskd.txt     
      inflating: __MACOSX/data/._g0pA_taskd.txt  
      inflating: data/g0pA_taske.txt     
      inflating: __MACOSX/data/._g0pA_taske.txt  
      inflating: data/g0pB_taska.txt     
      inflating: __MACOSX/data/._g0pB_taska.txt  
      inflating: data/g0pB_taskb.txt     
      inflating: __MACOSX/data/._g0pB_taskb.txt  
      inflating: data/g0pB_taskc.txt     
      inflating: __MACOSX/data/._g0pB_taskc.txt  
      inflating: data/g0pB_taskd.txt     
      inflating: __MACOSX/data/._g0pB_taskd.txt  
      inflating: data/g0pB_taske.txt     
      inflating: __MACOSX/data/._g0pB_taske.txt  
      inflating: data/g0pC_taska.txt     
      inflating: __MACOSX/data/._g0pC_taska.txt  
      inflating: data/g0pC_taskb.txt     
      inflating: __MACOSX/data/._g0pC_taskb.txt  
      inflating: data/g0pC_taskc.txt     
      inflating: __MACOSX/data/._g0pC_taskc.txt  
      inflating: data/g0pC_taskd.txt     
      inflating: __MACOSX/data/._g0pC_taskd.txt  
      inflating: data/g0pC_taske.txt     
      inflating: __MACOSX/data/._g0pC_taske.txt  
      inflating: data/g0pD_taska.txt     
      inflating: __MACOSX/data/._g0pD_taska.txt  
      inflating: data/g0pD_taskb.txt     
      inflating: __MACOSX/data/._g0pD_taskb.txt  
      inflating: data/g0pD_taskc.txt     
      inflating: __MACOSX/data/._g0pD_taskc.txt  
      inflating: data/g0pD_taskd.txt     
      inflating: __MACOSX/data/._g0pD_taskd.txt  
      inflating: data/g0pD_taske.txt     
      inflating: __MACOSX/data/._g0pD_taske.txt  
      inflating: data/g0pE_taska.txt     
      inflating: __MACOSX/data/._g0pE_taska.txt  
      inflating: data/g0pE_taskb.txt     
      inflating: __MACOSX/data/._g0pE_taskb.txt  
      inflating: data/g0pE_taskc.txt     
      inflating: __MACOSX/data/._g0pE_taskc.txt  
      inflating: data/g0pE_taskd.txt     
      inflating: __MACOSX/data/._g0pE_taskd.txt  
      inflating: data/g0pE_taske.txt     
      inflating: __MACOSX/data/._g0pE_taske.txt  
      inflating: data/g1pA_taska.txt     
      inflating: __MACOSX/data/._g1pA_taska.txt  
      inflating: data/g1pA_taskb.txt     
      inflating: __MACOSX/data/._g1pA_taskb.txt  
      inflating: data/g1pA_taskc.txt     
      inflating: __MACOSX/data/._g1pA_taskc.txt  
      inflating: data/g1pA_taskd.txt     
      inflating: __MACOSX/data/._g1pA_taskd.txt  
      inflating: data/g1pA_taske.txt     
      inflating: __MACOSX/data/._g1pA_taske.txt  
      inflating: data/g1pB_taska.txt     
      inflating: __MACOSX/data/._g1pB_taska.txt  
      inflating: data/g1pB_taskb.txt     
      inflating: __MACOSX/data/._g1pB_taskb.txt  
      inflating: data/g1pB_taskc.txt     
      inflating: __MACOSX/data/._g1pB_taskc.txt  
      inflating: data/g1pB_taskd.txt     
      inflating: __MACOSX/data/._g1pB_taskd.txt  
      inflating: data/g1pB_taske.txt     
      inflating: __MACOSX/data/._g1pB_taske.txt  
      inflating: data/g1pD_taska.txt     
      inflating: __MACOSX/data/._g1pD_taska.txt  
      inflating: data/g1pD_taskb.txt     
      inflating: __MACOSX/data/._g1pD_taskb.txt  
      inflating: data/g1pD_taskc.txt     
      inflating: __MACOSX/data/._g1pD_taskc.txt  
      inflating: data/g1pD_taskd.txt     
      inflating: __MACOSX/data/._g1pD_taskd.txt  
      inflating: data/g1pD_taske.txt     
      inflating: __MACOSX/data/._g1pD_taske.txt  
      inflating: data/g2pA_taska.txt     
      inflating: __MACOSX/data/._g2pA_taska.txt  
      inflating: data/g2pA_taskb.txt     
      inflating: __MACOSX/data/._g2pA_taskb.txt  
      inflating: data/g2pA_taskc.txt     
      inflating: __MACOSX/data/._g2pA_taskc.txt  
      inflating: data/g2pA_taskd.txt     
      inflating: __MACOSX/data/._g2pA_taskd.txt  
      inflating: data/g2pA_taske.txt     
      inflating: __MACOSX/data/._g2pA_taske.txt  
      inflating: data/g2pB_taska.txt     
      inflating: __MACOSX/data/._g2pB_taska.txt  
      inflating: data/g2pB_taskb.txt     
      inflating: __MACOSX/data/._g2pB_taskb.txt  
      inflating: data/g2pB_taskc.txt     
      inflating: __MACOSX/data/._g2pB_taskc.txt  
      inflating: data/g2pB_taskd.txt     
      inflating: __MACOSX/data/._g2pB_taskd.txt  
      inflating: data/g2pB_taske.txt     
      inflating: __MACOSX/data/._g2pB_taske.txt  
      inflating: data/g2pC_taska.txt     
      inflating: __MACOSX/data/._g2pC_taska.txt  
      inflating: data/g2pC_taskb.txt     
      inflating: __MACOSX/data/._g2pC_taskb.txt  
      inflating: data/g2pC_taskc.txt     
      inflating: __MACOSX/data/._g2pC_taskc.txt  
      inflating: data/g2pC_taskd.txt     
      inflating: __MACOSX/data/._g2pC_taskd.txt  
      inflating: data/g2pC_taske.txt     
      inflating: __MACOSX/data/._g2pC_taske.txt  
      inflating: data/g2pE_taska.txt     
      inflating: __MACOSX/data/._g2pE_taska.txt  
      inflating: data/g2pE_taskb.txt     
      inflating: __MACOSX/data/._g2pE_taskb.txt  
      inflating: data/g2pE_taskc.txt     
      inflating: __MACOSX/data/._g2pE_taskc.txt  
      inflating: data/g2pE_taskd.txt     
      inflating: __MACOSX/data/._g2pE_taskd.txt  
      inflating: data/g2pE_taske.txt     
      inflating: __MACOSX/data/._g2pE_taske.txt  
      inflating: data/g3pA_taska.txt     
      inflating: __MACOSX/data/._g3pA_taska.txt  
      inflating: data/g3pA_taskb.txt     
      inflating: __MACOSX/data/._g3pA_taskb.txt  
      inflating: data/g3pA_taskc.txt     
      inflating: __MACOSX/data/._g3pA_taskc.txt  
      inflating: data/g3pA_taskd.txt     
      inflating: __MACOSX/data/._g3pA_taskd.txt  
      inflating: data/g3pA_taske.txt     
      inflating: __MACOSX/data/._g3pA_taske.txt  
      inflating: data/g3pB_taska.txt     
      inflating: __MACOSX/data/._g3pB_taska.txt  
      inflating: data/g3pB_taskb.txt     
      inflating: __MACOSX/data/._g3pB_taskb.txt  
      inflating: data/g3pB_taskc.txt     
      inflating: __MACOSX/data/._g3pB_taskc.txt  
      inflating: data/g3pB_taskd.txt     
      inflating: __MACOSX/data/._g3pB_taskd.txt  
      inflating: data/g3pB_taske.txt     
      inflating: __MACOSX/data/._g3pB_taske.txt  
      inflating: data/g3pC_taska.txt     
      inflating: __MACOSX/data/._g3pC_taska.txt  
      inflating: data/g3pC_taskb.txt     
      inflating: __MACOSX/data/._g3pC_taskb.txt  
      inflating: data/g3pC_taskc.txt     
      inflating: __MACOSX/data/._g3pC_taskc.txt  
      inflating: data/g3pC_taskd.txt     
      inflating: __MACOSX/data/._g3pC_taskd.txt  
      inflating: data/g3pC_taske.txt     
      inflating: __MACOSX/data/._g3pC_taske.txt  
      inflating: data/g4pB_taska.txt     
      inflating: __MACOSX/data/._g4pB_taska.txt  
      inflating: data/g4pB_taskb.txt     
      inflating: __MACOSX/data/._g4pB_taskb.txt  
      inflating: data/g4pB_taskc.txt     
      inflating: __MACOSX/data/._g4pB_taskc.txt  
      inflating: data/g4pB_taskd.txt     
      inflating: __MACOSX/data/._g4pB_taskd.txt  
      inflating: data/g4pB_taske.txt     
      inflating: __MACOSX/data/._g4pB_taske.txt  
      inflating: data/g4pC_taska.txt     
      inflating: __MACOSX/data/._g4pC_taska.txt  
      inflating: data/g4pC_taskb.txt     
      inflating: __MACOSX/data/._g4pC_taskb.txt  
      inflating: data/g4pC_taskc.txt     
      inflating: __MACOSX/data/._g4pC_taskc.txt  
      inflating: data/g4pC_taskd.txt     
      inflating: __MACOSX/data/._g4pC_taskd.txt  
      inflating: data/g4pC_taske.txt     
      inflating: __MACOSX/data/._g4pC_taske.txt  
      inflating: data/g4pD_taska.txt     
      inflating: __MACOSX/data/._g4pD_taska.txt  
      inflating: data/g4pD_taskb.txt     
      inflating: __MACOSX/data/._g4pD_taskb.txt  
      inflating: data/g4pD_taskc.txt     
      inflating: __MACOSX/data/._g4pD_taskc.txt  
      inflating: data/g4pD_taskd.txt     
      inflating: __MACOSX/data/._g4pD_taskd.txt  
      inflating: data/g4pD_taske.txt     
      inflating: __MACOSX/data/._g4pD_taske.txt  
      inflating: data/g4pE_taska.txt     
      inflating: __MACOSX/data/._g4pE_taska.txt  
      inflating: data/g4pE_taskb.txt     
      inflating: __MACOSX/data/._g4pE_taskb.txt  
      inflating: data/g4pE_taskc.txt     
      inflating: __MACOSX/data/._g4pE_taskc.txt  
      inflating: data/g4pE_taskd.txt     
      inflating: __MACOSX/data/._g4pE_taskd.txt  
      inflating: data/g4pE_taske.txt     
      inflating: __MACOSX/data/._g4pE_taske.txt  
      inflating: data/orig_taska.txt     
      inflating: __MACOSX/data/._orig_taska.txt  
      inflating: data/orig_taskb.txt     
      inflating: data/orig_taskc.txt     
      inflating: __MACOSX/data/._orig_taskc.txt  
      inflating: data/orig_taskd.txt     
      inflating: __MACOSX/data/._orig_taskd.txt  
      inflating: data/orig_taske.txt     
      inflating: __MACOSX/data/._orig_taske.txt  
      inflating: data/test_info.csv      
      inflating: __MACOSX/data/._test_info.csv  
      inflating: __MACOSX/._data         



```python
# import libraries
import pandas as pd
import numpy as np
import os
```

This plagiarism dataset is made of multiple text files; each of these files has characteristics that are is summarized in a `.csv` file named `file_information.csv`, which we can read in using `pandas`.


```python
csv_file = 'data/file_information.csv'
plagiarism_df = pd.read_csv(csv_file)

# print out the first few rows of data info
plagiarism_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>File</th>
      <th>Task</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>g0pA_taska.txt</td>
      <td>a</td>
      <td>non</td>
    </tr>
    <tr>
      <th>1</th>
      <td>g0pA_taskb.txt</td>
      <td>b</td>
      <td>cut</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g0pA_taskc.txt</td>
      <td>c</td>
      <td>light</td>
    </tr>
    <tr>
      <th>3</th>
      <td>g0pA_taskd.txt</td>
      <td>d</td>
      <td>heavy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>g0pA_taske.txt</td>
      <td>e</td>
      <td>non</td>
    </tr>
    <tr>
      <th>5</th>
      <td>g0pB_taska.txt</td>
      <td>a</td>
      <td>non</td>
    </tr>
    <tr>
      <th>6</th>
      <td>g0pB_taskb.txt</td>
      <td>b</td>
      <td>non</td>
    </tr>
    <tr>
      <th>7</th>
      <td>g0pB_taskc.txt</td>
      <td>c</td>
      <td>cut</td>
    </tr>
    <tr>
      <th>8</th>
      <td>g0pB_taskd.txt</td>
      <td>d</td>
      <td>light</td>
    </tr>
    <tr>
      <th>9</th>
      <td>g0pB_taske.txt</td>
      <td>e</td>
      <td>heavy</td>
    </tr>
  </tbody>
</table>
</div>



## Types of Plagiarism

Each text file is associated with one **Task** (task A-E) and one **Category** of plagiarism, which you can see in the above DataFrame.

###  Five task types, A-E

Each text file contains an answer to one short question; these questions are labeled as tasks A-E.
* Each task, A-E, is about a topic that might be included in the Computer Science curriculum that was created by the authors of this dataset. 
    * For example, Task A asks the question: "What is inheritance in object oriented programming?"

### Four categories of plagiarism 

Each text file has an associated plagiarism label/category:

1. `cut`: An answer is plagiarized; it is copy-pasted directly from the relevant Wikipedia source text.
2. `light`: An answer is plagiarized; it is based on the Wikipedia source text and includes some copying and paraphrasing.
3. `heavy`: An answer is plagiarized; it is based on the Wikipedia source text but expressed using different words and structure. Since this doesn't copy directly from a source text, this will likely be the most challenging kind of plagiarism to detect.
4. `non`: An answer is not plagiarized; the Wikipedia source text is not used to create this answer.
5. `orig`: This is a specific category for the original, Wikipedia source text. We will use these files only for comparison purposes.

> So, out of the submitted files, the only category that does not contain any plagiarism is `non`.

In the next cell, print out some statistics about the data.


```python
# print out some stats about the data
print('Number of files: ', plagiarism_df.shape[0])  # .shape[0] gives the rows 
# .unique() gives unique items in a specified column
print('Number of unique tasks/question types (A-E): ', (plagiarism_df['Task'].unique()))
print('Unique plagiarism categories: ', (plagiarism_df['Category'].unique()))
```

    Number of files:  100
    Number of unique tasks/question types (A-E):  ['a' 'b' 'c' 'd' 'e']
    Unique plagiarism categories:  ['non' 'cut' 'light' 'heavy' 'orig']


You should see the number of text files in the dataset as well as some characteristics about the `Task` and `Category` columns. **Note that the file count of 100 *includes* the 5 _original_ wikipedia files for tasks A-E.** If you take a look at the files in the `data` directory, you'll notice that the original, source texts start with the filename `orig_` as opposed to `g` for "group." 

> So, in total there are 100 files, 95 of which are answers (submitted by people) and 5 of which are the original, Wikipedia source texts.

Your end goal will be to use this information to classify any given answer text into one of two categories, plagiarized or not-plagiarized.

### Distribution of Data

Next, let's look at the distribution of data. In this course, we've talked about traits like class imbalance that can inform how you develop an algorithm. So, here, we'll ask: **How evenly is our data distributed among different tasks and plagiarism levels?**

Below, you should notice two things:
* Our dataset is quite small, especially with respect to examples of varying plagiarism levels.
* The data is distributed fairly evenly across task and plagiarism types.


```python
# Show counts by different tasks and amounts of plagiarism

# group and count by task
counts_per_task=plagiarism_df.groupby(['Task']).size().reset_index(name="Counts")
print("\nTask:")
display(counts_per_task)

# group by plagiarism level
counts_per_category=plagiarism_df.groupby(['Category']).size().reset_index(name="Counts")
print("\nPlagiarism Levels:")
display(counts_per_category)

# group by task AND plagiarism level
counts_task_and_plagiarism=plagiarism_df.groupby(['Task', 'Category']).size().reset_index(name="Counts")
print("\nTask & Plagiarism Level Combos :")
display(counts_task_and_plagiarism)
```

    
    Task:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Task</th>
      <th>Counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>


    
    Plagiarism Levels:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Category</th>
      <th>Counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cut</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>heavy</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>light</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>non</td>
      <td>38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>orig</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


    
    Task & Plagiarism Level Combos :



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Task</th>
      <th>Category</th>
      <th>Counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>cut</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>heavy</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>light</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>non</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>orig</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b</td>
      <td>cut</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b</td>
      <td>heavy</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b</td>
      <td>light</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>b</td>
      <td>non</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>b</td>
      <td>orig</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>c</td>
      <td>cut</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>c</td>
      <td>heavy</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>c</td>
      <td>light</td>
      <td>4</td>
    </tr>
    <tr>
      <th>13</th>
      <td>c</td>
      <td>non</td>
      <td>7</td>
    </tr>
    <tr>
      <th>14</th>
      <td>c</td>
      <td>orig</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>d</td>
      <td>cut</td>
      <td>4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>d</td>
      <td>heavy</td>
      <td>4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>d</td>
      <td>light</td>
      <td>5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>d</td>
      <td>non</td>
      <td>6</td>
    </tr>
    <tr>
      <th>19</th>
      <td>d</td>
      <td>orig</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>e</td>
      <td>cut</td>
      <td>5</td>
    </tr>
    <tr>
      <th>21</th>
      <td>e</td>
      <td>heavy</td>
      <td>3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>e</td>
      <td>light</td>
      <td>4</td>
    </tr>
    <tr>
      <th>23</th>
      <td>e</td>
      <td>non</td>
      <td>7</td>
    </tr>
    <tr>
      <th>24</th>
      <td>e</td>
      <td>orig</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


It may also be helpful to look at this last DataFrame, graphically.

Below, you can see that the counts follow a pattern broken down by task. Each task has one source text (original) and the highest number on `non` plagiarized cases.


```python
import matplotlib.pyplot as plt
% matplotlib inline

# counts
group = ['Task', 'Category']
counts = plagiarism_df.groupby(group).size().reset_index(name="Counts")

plt.figure(figsize=(8,5))
plt.bar(range(len(counts)), counts['Counts'], color = 'blue')
```




    <BarContainer object of 25 artists>




![png](output_12_1.png)


## Up Next

This notebook is just about data loading and exploration, and you do not need to include it in your final project submission. 

In the next few notebooks, you'll use this data to train a complete plagiarism classifier. You'll be tasked with extracting meaningful features from the text data, reading in answers to different tasks and comparing them to the original Wikipedia source text. You'll engineer similarity features that will help identify cases of plagiarism. Then, you'll use these features to train and deploy a classification model in a SageMaker notebook instance. 
