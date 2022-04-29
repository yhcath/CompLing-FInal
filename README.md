





Use the following commands to install required libraries and packages:

	pip install -r requirements.txt

	python

	nltk.download('omw-1.4')

	nltk.download('sentiwordnet')


---------------------------------



Steps to run the program:

0. Sample files to run _sentiment_analysis.py_ with are already included in the repo: _semeval2013_train.tsv_, _sentiment_randomized.tsv_. Unless you want to pull your own datasets, skip to Step 5. 

1. You need an environment variable TWITTER_BEARER_TOKEN to pull historical tweets from Twitter API. The token can be obtained through requesting elevated or research developer access to Twitter. 

2. Use _pull_historical_tweets.py_ to pull self-collected tweets. This stores the data in _sentiment_nonrandomized.tsv_.
   
   Example command line input:
   
   		python pull_historical_tweets.py

3. In order to pull SemEval2013 data, follow the instructions at https://github.com/aritter/twitter_download. The dist files can be pulled from https://alt.qcri.org/semeval2017/task4/?id=download-the-full-training-data-for-semeval-2017-task-4. 

4. Use _randomize.py_ to shuffle self-collected tweets for partition purposes.
  
   Example command line input:
   
   		python randomize.py
	

5. Use _sentiment_analysis.py_ to run the program. Example results are stored in _results_new.txt_ and _results_old_(trained_on_semeval_test).txt_ for your convenience, due to the fact that running this program takes up a significant amount of time. 
   
   Example command line input:
   
   		python sentiment_analysis.py --training-set semeval2013-train.tsv --data-set sentiment_randomized.tsv


Note: Before submitting the code, I realized I ran the _semeval2013-test.tsv_ in place of _semeval2013-train.tsv_. The results may be different from what's reported in the poster for the out-of-domain training model. I included _semeval2013-test.tsv_ in the repo as well, in case anyone wants to generate results reported in the poster. To do that, run:

	python sentiment_analysis.py --training-set semeval2013-test.tsv --data-set sentiment_randomized.tsv
	
	
	
