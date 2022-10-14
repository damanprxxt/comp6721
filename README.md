# Group No NS18
# Face mask detection CNN Model part 1


# Files included are
files included are
The project Report
age_pickle files and this contains all the path for young and adult dataset.
gender_pickle files and this contains all the path for female and male dataset.
seperate dataset for male and female based on the gender for bias calculation
seprate dataset for young and adult based on the gender for bias calculation
biasCalculater.py for calculating the bias.
faceMaskDetector.py for the main code from where we can control the codes
kfold_clustering.py for implementing the k_fold.
pickleFileGenerator.py for generating new pickle file for datasets.
the resuslts after correcting the bias for gender and age 


# steps to run
after downloading the dataset in a folder
run the pickle file generator to generate the pickle files which are required to run the project
after running the generator go to the facemaskdetector.py and in that there are commented pieces of code under main
and the functionalities are defined for each of the function
every function has some parameters which can be changed accordingly
the default parameters are already given
the code keeps records of all the runs and if user wants to run the last recorded run just uncomment the related command under main
