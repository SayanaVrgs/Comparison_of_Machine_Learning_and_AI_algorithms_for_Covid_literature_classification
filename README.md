# Comparison of Machine Learning and AI algorithms for Covid literature classification
<img align="right" alt="doc class" width="300" height="200" src="https://www.todaysoftmag.com/images/articles/tsm64/a41.jpg">
The objective of this project is to compare the efficiency and reliability (via accuracy and microF1 scores) of COVID-related document classification by supervised and unsupervised machine learning models with well-known NLP transformers. My project is based on the research findings by B. J. Gutierrez et al.(B. J. Gutierrez, J. Zeng, D. Zhang, P. Zhang, and Y. Su. Document classification for COVID-19 literature, 2020) who have compared various machine learning, neural network and artificial intelligence models for document classification of Covid-19 literature, back in 2020.

## Dataset used
<img align="right" alt="dataset" width="400" height="200" src="https://github.com/SayanaVrgs/Comparison_of_Machine_Learning_and_AI_algorithms_for_Covid_literature_classification/blob/master/dataset.jpg">
The LitCovid dataset is a collection of recently published PubMed articles that are directly related to the 2019 novel Coronavirus. The dataset contains upwards of 375,727 articles and approximately 3,000 new articles are added every week, even now, 2 years after the pandemic started. The articles uploaded have only abstract or only text or both abstract and text. These articles were then cross-verified with their respective classification (from the 8 topic labels: Prevention, Treatment, Diagnosis, Mechanism, Case Report, Transmission, Forecasting, and General) based on their pmid.

The COVID-19 Open Research Dataset (CORD-19)[10] was one of the earliest datasets released to facilitate cooperation between the computing community and the many relevant actors of the COVID-19 pandemic. It consists of approximately 60,000 papers related to COVID-19 and similar coronaviruses such as SARS and MERS since the SARS epidemic of 2002. Due to their differences in scope, this dataset shares only around 1,200 articles with the LitCovid dataset. 

Ideally, when faced with a pandemic of this intensity, experts rely not only on data on the current virus but also on historical data of past epidemics and similar virus outbreaks. Hence, it becomes important for models trained on the LitCovid dataset to successfully categorize articles about related epidemics as well. This is the reason why the CORD-19 dataset was chosen as it also contains information about viruses and epidemics similar to COVID-19.

## ML/AI Models used
| Supervised Machine Learning Algorithm  | Unsupervised Machine Learning Algorithm | Transformers |
| ------------- | ------------- | ------------- |
| Decision Trees (gini and entropy), SVM, LR  | DOc2Vec (DBOW and Distributed Memory | BERT and BioBERT |

## Results
<img align="right" alt="result" width="550" height="300" src="https://github.com/SayanaVrgs/Comparison_of_Machine_Learning_and_AI_algorithms_for_Covid_literature_classification/blob/master/data2.png">

On comparing certain instances where predictions and actual classifications differ in BERT models, it is seen that the models tend to correlation certain categories, namely prevention and treatment or diagnosis and treatment etc. It could also be that the number of epochs for training is low, and the accuracy might increase at higher epochs. Or, the abstract might have certain key words and the text content might have different key words due to which the models would have difficulty in correctly classifying the document.

On evaluating the models on CORD-19 dataset, it is seen that the accuracy and F1 scores actually drop. This massive drop in performance from a minor change in domain indicates that the models have trouble ignoring the overarching COVID-19 topic and isolating relevant signals from each category.

## Conclusion
Overall, we observe that pre-trained (supervised) and fine-tuned (unsupervised) models like BERT are much more data-efficient than other models and that BioBERT is the most efficient, demonstrating the importance of domain-specific pre-training. Additionally, if an accuracy of 71% can be considered as good enough, then traditional machine learning methods like LR and SVM can also be utilized provided the data is sufficiently pre-processed.

## Hands-On
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
Kindly ensure the order of contents of the folder is followed, to ensure successful execution. Ensure that nltk downloader, wget and gzip is installed in the python version that you use. Anaconda would need to be installed to run the Jupyter notebook for .ipynb file execution. Google Colab with at least 1 GPU and high-speed RAM would be required for BERT model execution.

First, run the commands mentioned in "to load litcovid data.txt" file. Next, move to the script file and execute the litcovid dataset from xml python code.
The output of this code is a folder FullLitCovid under data folder, containing the val, test and train tsv files and LitCovid.val, LitCovid.test and LitCovid.train csv files. The csv files capture the entire content of LitCovid dataset in csv format whereas the tsv files capture only the category classification and text content of the LitCovid dataset. The tsv files would be the input to our next step: Executing supervised and unsupervised ML models.

Jupyter notebook would be required for this part as sklearnex, an intel extension to speed up sci-kit executions is available only in Anaconda. On Jupyter notebook, go to the scripts folder and execute Supervised ML methodology.ipynb file and UnSupervised ML Methodology.ipynb file. You should be able to execute and see the metrics such as Accuracy and micro F1 score for train, val, test and eval datasets on decision tree (gini and entropy) models and doc2vec (dbow and dm) models respectively. The BERT models have been pre-trained and fine-tuned on Google Colab. To implement and check the accuracy and micro F1 scores of the BERT models, just upload the Bert Base LitCovid and BioBert LitCovid ipynb files on Google colab and execute them.

<sub>Note: In order to see the improved accuracy of SVM and LR, please execute the traditional models ipynb file taken from B. J. Gutierrez et al.(B. J. Gutierrez, J. Zeng, D. Zhang, P. Zhang, and Y. Su. Document classification for covid-19 literature, 2020), in the jupyter notebook.</sub>
