# rp_deeplearning_assignment
RP assignment for Deep Learning Fundamentals

# Question 1
- Run Question1_DLF.ipynb in jupyter notebook
- Set 'verbose=0' or 'verbose' in the notebook to turn on verbosity by commenting/uncommenting the lines.

# Question 2
- There are 3 files in the directory.  1) Question2_DLF.ipynb, 2) Question2_Test_DLF.ipynb 3) Question2_Test_DLF.py
- Question2_DLF.ipynb is to train the model.  The dataset is loaded from 'data' directory.  
  The model generated is saved, together with the scaler in 'model' directory.
- Like question 1 above, you can set 'verbose=0' or 'verbose=1' in the script.

- Question2_Test_DLF.ipynb is to test the model using test files 1) datatest.txt or 2) datatest2.txt.  
  Question2_Test_DLF.ipynb has been converted to a python script and you can run it in a terminal, 
  with the necessary python modules installed.
- Like question 1 above, you can set 'verbose=0' or 'verbose=1' in the script.  
- You can test with either one of the two test files by commenting and uncommenting the filename lines in the script.
- An example to run it as a python script:
            python Question2_Test_DLF.py -h

            usage: Question #2 Testing [-h] [-v VERBOSE] [-t TEST]

            optional arguments:
               -h, --help            show this help message and exit
               -v VERBOSE, --verbose VERBOSE
                                     turn on or off verbose mode (default: 1)
               -t TEST, --test TEST  (0) for datatest.txt and (1) datatest1.txt
  Example: python Question2_Test_DLF.py -v 0 -t 1 OR
           python Question2_Test_DLF.py -v 1 -t 2
