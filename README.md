# LitCovid_NLP
Classification of Covid-19 documents: Comparing Supervised and Unsupervised Machine learning techniques with transformers

The year 2020 changed our lives forever. Covid-19, an unknown term until then, became the new reality of our lives. With very little information available about this
virus, there was an urgent need for knowledge that would help in dealing with this deadly pandemic. And experts from various fields did respond to this need. Various
documents regarding Covid-19 transmission, prevention, diagnosis etc were being published with great urgency. But with such an outpour of data, the need of the
hour was an effective automated document classification technique that could accurately classify any given Covid-19 related document, the research for which is still in progress. Through this project, I wanted to compare the efficiency and accuracy of document classification by supervised and unsupervised machine learning models with well-known NLP transformers. This is done by comparing the accuracy and microF1 scores of supervised and unsupervised machine learning techniques with the latest transformer models. My project is based on the research findings by B. J. Gutierrez et al.(B. J. Gutierrez, J. Zeng, D. Zhang, P. Zhang, and Y. Su. Document classification for covid-19 literature, 2020) who have compared various machine learning, neural network and artificial intelligence models for document classification of Covid-19 literature, back in 2020.

To try out this project, please clone this repo. Available folder contents: LitCovid folder, which has a requirements text file, a data
subfolder and a script subfolder.
The data subfolder has a cord19 test tsv file, a litcovid category order text file and a blank FullLitCovid folder. 
The script folder has:
1. to load litcovid data.txt file.
2. traditional models ipynb file.
3. Supervised ML methodology ipynb file.
4. UnSupervised ML Methodology ipynb file.
5. litcovid dataset from xml py file.
6. utils py file.
7. Bert LitCovid ipynb file.
8. BioBert LitCovid ipynb file.
Kindly ensure the order of contents of the folder is followed, to ensure successful execution. Please note that the venv is installed on windows 64bit machine via cmd.
Ensure python(3.6 or higher) virtual environment, ‘venv’, is available and the installables mentioned in requirements.txt file are installed on the venv, before executing the python file.Also, ensure that nltk downloader, wget and gzip is installed in the python venv. Anaconda would need to be installed to run the Jupyter notebook for .ipynb file execution. Google Colab with at least 1 GPU and high-speed RAM would be required for BERT model execution.

For the Data pre-processing aspect, we would be focusing on getting the required data and running the python code on the Python3 venv (windows cmd). In cmd, go the location where venv is installed (cd Documents\nlp \env\Scripts) and execute .\activate. Next, run the commands mentioned in open the "to load litcovid data.txt" file. Next move to the script file and execute the litcovid dataset from xml python code.
The output of this code is a folder FullLitCovid under data folder, containing the val, test and train tsv files and LitCovid.val,LitCovid.test and LitCovid.train csv files. The csv files capture the entire content of LitCovid dataset in csv format whereas the tsv files capture only the category classification and text content of the LitCovid dataset. The tsv files would be the input to our next step: Executing supervised and unsupervised ML models.

Jupyter notebook would be required for this part as sklearnex, an intel extension to speed up sci-kit executions is available only in Anaconda. On Jupyter notebook, go to the scripts folder and execute Supervised ML methodology.ipynb file. You should be able to execute and see the metrics such as Accuracy and micro F1 score for train, val, test and eval datasets on decision tree (gini and entropy) models. Next, open UnSupervised ML Methodology.ipynb file under the script folder on Jupyter notebook.You should be able to execute and see the metrics such as Accuracy and micro F1 score for train, val, test and eval datasets on doc2vec (dbow and dm) models.

A prerequisite for the BERT model execution is the availability of GPUs and high-speed RAMs. For this project, the BERT models have been pre-trained and fine-tuned on Google colab. To implement and check the accuracy and micro F1 scores of the BERT models, just upload the Bert Base LitCovid and BioBert LitCovid ipynb files on Google colab and execute them. The results would be available as and when the individual cells of the notebook are executed. Please note that you would need to set the hardware as GPU and Runtime shape as High–Speed RAM in the Notebook Settings under Edit option.

Note: In order to see the improved accuracy of SVM and LR, please execute the traditional models ipynb file taken from B. J. Gutierrez et al.(B. J. Gutierrez, J. Zeng, D. Zhang, P. Zhang, and Y. Su. Document classification for covid-19 literature, 2020), in the jupyter notebook.
