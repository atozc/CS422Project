Class: CS 422
Project: Project 1
Author: Crystal Atoz 

CONTENTS OF THIS FILE 
-----------------------
* Description
* Program Running Instruction 
* Write Up 

DESCRIPTION
-----------------------
This project creates decision trees and random forests using python data storage methods. The 
decision trees are created by having functions that train, test, and predict. The random 
forests are built and tested. Although, it is noted that is not included in this 
project, as I had trouble implementing. The decision trees use data from .csv test files.

PROGRAM RUNNING INSTRUCTIONS
-----------------------
Compile and execute test file(s) with data_storage.py 
and decision_trees.py in same folder.

WRITE UP 
-----------------------
1. Did you implement your decision tree functions iteratively or recursively? Which data structure
did you choose and were you happy with that choice? If you were unhappy with that choice which other data
structure would have built the model with?

    I implemented my decision tree functions both iteratively and recursively. For example, I created many separate 
    functions to build the tree and iteratively went through all the possibilities when creating a branch to 
    branch on. In regards to the data structure used, I used a list to traverse the dictionary and then a binary tree 
    to branch on. To be honest, I am relatively happy with my choice. 

2. Why might the individual trees have such variance in their accuracy? How would you reduce
this variance and potentially improve accuracy?

    Individual trees have variance in their accuracy due to their nature. They can easily change by small changes 
    in the input variables. To reduce the varaince and potentially improve accuracy, I would add more data. 

3. Why is it beneficial for the random forest to use an odd number of individual trees?

    It is beneficial for the random forest to use an odd number of individual trees because it can aid in avoiding 
    overfitting the data and could possibly improve performance of the random forest. 

4. Overall, if you are still feeling uncomfortable working with python, what aspect of the
 coding language do you feel you are struggling with the most? If you do feel comfortable, 
 what part of python do you feel you should continue practicing?

    I struggled quite a bit on this project. I am quite uncomfortable with python. I had to 
    try to understand different data structures and how some modules worked like numpy. On the other 
    hand, I struggled with remembering recursion and some data structures. I would like to continue 
    practicing python and brush up on some code knowledge. This includes the numpy module, tree data structures, 
    recursion, arrays, lists, amongst others. I also found it difficult to figure out where to start.  
    I understood how the trees work, just had a hard time translating that to code.






