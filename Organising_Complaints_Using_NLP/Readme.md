### Consumer Complaints Data

#### Objective
In this project we will work on complaints textual data against financial companies to extract Company Names and complaints topic using advanced text mining algorithms. 

    1. We will use NER tagging to extract Organisation Names from complaints text
    2. We will use Topic Modelling to extract what complaints was regarding

#### Dataset Info
The dataset comprises of Consumer Complaints on Financial products and we’ll see how to classify consumer complaints text into these categories: Debt collection, Consumer Loan, Mortgage, Credit card, Credit reporting, Student loan, Bank account or service, Payday loan, Money transfers, Other financial service, Prepaid card.

The dataset consist of 670598 rows (complaints) and 18 columns.

For our work we are only interested in complaint Id, Company, Product, Issue and Complaint Narrative.Complaints Narrative will be used for text mining and other columns for evaluating the results.
### Basic EDA on datasets:
Insights 1 : Complaints Narrative is available for only 114K complaints and histogram for text size indicates most complaints have length < 1000. It also indicates that when text is avilable its complete

![image](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Organising_Complaints_Using_NLP/Output/complaint_size.png)

Insights 2: Complaints corresponds to 12 different product across various categories from Debt to Virtual Currency.For deatils of compalints by product refer below

![image](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Organising_Complaints_Using_NLP/Output/complaints_product.png)
![image](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Organising_Complaints_Using_NLP/Output/Disputed_CrossTab_Compalints.png)

Insights 3: Companies with maximum number of complaints are as follows
![image](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Organising_Complaints_Using_NLP/Output/top_25_companies_by_freq.png)

### WordCloud for top 5 prdocuts by frequency
  ##### Debt Collection
![debt](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Organising_Complaints_Using_NLP/Output/Debt%20collection.png)
  ##### Mortgage
![mortgage](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Organising_Complaints_Using_NLP/Output/Mortgage.png)
  ##### Credit Card
![Credit_card](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Organising_Complaints_Using_NLP/Output/Credit%20card.png)
  ##### Credit Reporting
![credit_reporting](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Organising_Complaints_Using_NLP/Output/Credit%20reporting.png)
  #####  Bank Account or service
![bank_Account](https://github.com/Ashwinikumar1/NLP-DL/blob/master/Organising_Complaints_Using_NLP/Output/Bank%20account%20or%20service.png)

Wordcloud for other products can be found at this [location](https://github.com/Ashwinikumar1/NLP-DL/tree/master/Organising_Complaints_Using_NLP/Output)

#### Methodology for extracting organisation Names :
We will use Name Entity Tagging provided by spacy in its vanilla english language model.From the tagging we will keep all words which has been assigned an 'ORG' tags. As it is a pretrained model the tagging is quite fast and scalable

After applying the pretrained model, we create a new columns which has list of distinct words recongnized as company name:

| Model Output Preview |
|:-------|
['Applied', 'USPMO', 'USPMO', 'Capital One', 'USPMO', 'Capital One', 'Judgement']
[]
['Charter One BankRBS Citizens Bank', 'Financial and Medical', 'Charter One Bank', 'Master Card', 'Charter One Bank', 'Charter One Bank', 'the Charter One Bank']
['noone']
['Collection Consultants', 'Collection Consultants', 'HIPPA', 'HIPPA']

####  Issues with Current Model Ouput :

    1. In 40K claims model is not able to identify Orgs, on manual QA it seems the compalints dont have company name 
    2. As it is free text, each organisation can be written in multiple ways for example Capital One is written as following
        ["Cap One","Capital One Bank","Cap One Corp",'the Capital One Bank', 'Capital One' ] etc
    3. We need to create a method of normalizing the company names to the required list
    4. We need to find a way to keep company names and remove other orgs in case of mutiple orgs in same complaint

###### Problem 1 - Company names are not normalised :
1. Create a matching score between every company names in the stored dictionary. We can use fuzzy wuzzy matching for the same
2. Based on fuzzy wuzzy matching scoring algorithm metric, create cluster. As we are not sure of how many clsuters be used we will use a Affinity clustering algorithm
3. Once the cluster is created, assign names based which has highest similarity score to everyone else in the cluster

###### Problem 2 - Few Noise terms which do not repspond to company names but are recognised as Companies:
1. Hypothesis : The freq of these noise terms would be very much less than the freqeuncy of comapny names
2. We should do this on cluster frequency rather than the individual words as individual frequency can be biased and not reliable due to different names used

### Topic Models to extract complaint topic using text mining
As it is free text, before applying topic model we need to do data preprocessing to remove junk words,symbols and also reduce the size of vocabulary

The data processing pipline we have created uses SPacy and does following things:
1. Converts the token to lower and remove pronouns
2. Remove Stopwords from general englisgh words as wells user defined list
3. Remove all teh puntuations as well as digits 0-9
4. We covert the textual data to sparse matrix using TF-IDF vectoriser, we could have used count vectoriser also. But in my experience TF-IDF works better for topic models. Also we can tune parametrs like min_df, max_df and number of features

#### Model Tried

#### 1. Non Negative Maxtrix Factorzation from Sklearn

Topic #0: situation understand way tell time feel explain finally good like

Topic #1: report credit report credit account debt report credit account credit remove validate collection

Topic #2: xx xx xx xx xx xx xx xx xx xx report credit report credit date report credit account credit

Topic #3: account bank card debit bank account branch check check account banking credit card

Topic #4: mortgage loan payment refinance lender mortgage payment current servicer home appraisal

Topic #5: card credit card credit limit payment charge pay use cancel purchase

Topic #6: payment late pay account late payment monthly monthly payment miss month set

Topic #7: wells fargo wells fargo pay credit interest branch report high banking

Topic #8: loan pay default bank school interest private high student dollar

Topic #9: bank america bank america mortgage boa mortgage payment process foreclosure report letter

Topic #10: tell pay money phone come talk check manager thing need

#### 2. Latent Dirichlet Allocation from Sklearn 

Topic #0: bill insurance medical pay collection pay bill owe company receive service

Topic #1: score credit credit score report card limit drop credit card credit report history

Topic #2: modification loan mortgage home loan modification payment appraisal document tell time

Topic #3: debt letter court send state receive send letter judgment owe verification

Topic #4: equifax report dispute credit account information credit report item investigation file

Topic #5: sale foreclosure property mortgage home loan short short sale bank foreclose

Topic #6: theft identity identity theft report validation account debt credit victim information

Topic #7: open citi account account open open account settlement card close credit card credit

Topic #8: contract inc agreement sign signature term agree copy support service

Topic #9: bankruptcy discharge year ago ago year file report list credit account

Topic #10: principal payment apply monthly monthly payment twice loan interest additional pay

By comparing the results, we see that topic generated by Latent Dirchlet Allocation are much better but still can be improved

### Next Steps
1. Data needs a lot of cleaning for example "XX","XXX",organisation names
2. After looking at Topics we realise that a lot of keywords are important but do not give anyinformation about complaints. We should create a custom stop word list & clean the required data
3. We may want to try clustering topics into K means for naming and better understanding and grouping
4. Use LDA visualisation to understand topic in much more details









