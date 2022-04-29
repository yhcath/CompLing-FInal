import argparse
import pandas
import re
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import string
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn





def clean_data(data_set):
    # takes in a pandas dataframe object
    tweets = []
    sentiments = []
    counts = {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    }
    for tweet in data_set.values:
        if tweet[1] == "rep":
            continue
        if tweet[2] == "Not Available":
            continue
        else:
            tweets.append(tweet[2])
            sentiment = tweet[1]
            if sentiment == "-1":
                sentiments.append("negative")
                counts["negative"] += 1
            elif sentiment == "0":
                sentiments.append("neutral")
                counts["neutral"] += 1
            elif sentiment == "1":
                sentiments.append("positive")
                counts["positive"] += 1
            else:
                sentiments.append(sentiment)
                counts[sentiment] += 1

    print(counts)
    return tweets, sentiments

def generate_tokens(input_list):
    result = []

    tokenizer = TweetTokenizer()
    for input in input_list:
        # remove any mentions
        clean_input = re.sub("(@[A-Za-z0-9_]+)", "", input)
        # remove any links
        clean_input = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', clean_input)
        # remove all punctuation other than !,
        clean_input = re.sub("[?.,:;]", "", clean_input)
        # Remove hashtags, but keep the body of the hashtag
        clean_input = re.sub("#", "", clean_input)
        clean_tokens = tokenizer.tokenize(clean_input.lower())

        result.append(clean_tokens)

    return result

def lemmatize_and_remove_stopwords(tokens_list):
    result = []

    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    for tokens in tokens_list:
        cleaned_tokens = []

        for token, tag in pos_tag(tokens):
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token not in stop_words:
                cleaned_tokens.append(token.lower())
        result.append(cleaned_tokens)

    return result

def split_train_test(X, y):
    print("Splitting")
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    split = .9

    counts_train = {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    }

    counts_test = {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    }

    for i in range(len(X)):
        if i < len(X) * split:
            X_train.append(X[i])
            y_train.append(y[i])
            counts_train[y[i]] += 1
        else:
            X_test.append(X[i])
            y_test.append(y[i])
            counts_test[y[i]] += 1

    print("Train")
    print(counts_train)
    print("Test")
    print(counts_test)

    return X_train, y_train, X_test, y_test

def run_model(name, nb, training_X, training_y, testing_X, testing_y):
    print(f"Running model {name}")

    model = nb.fit(training_X, training_y)

    pred = model.predict(testing_X)
    accuracy = accuracy_score(testing_y, pred)
    precision = precision_score(testing_y, pred, average='macro', zero_division=0)
    recall = recall_score(testing_y, pred, average='macro', zero_division=0)
    f1 = f1_score(testing_y, pred, average='macro', zero_division=0)
    confusion = confusion_matrix(testing_y, pred)

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion-matrix": confusion
    }

def get_unigram_features(token_list_list):
    result = []

    for token_list in token_list_list:
        for token in token_list:
            if token not in result:
                result.append(token)

    return result

def build_unigram_features(token_list_list, feature_list):
    result = []

    for token_list in token_list_list:
        feature_row = [0]*len(feature_list)
        for token in token_list:
            if token not in feature_list:
                # Shouldn't happen for the training set, but might happen for the testing set
                continue
            else:
                feature_row[feature_list.index(token)] = 1

        result.append(feature_row)

    return result

def build_unigram2_features(token_list_list, feature_list):
    result = []

    for token_list in token_list_list:
        feature_row = [0]*len(feature_list)
        for token in token_list:
            if token not in feature_list:
                # Shouldn't happen for the training set, but might happen for the testing set
                continue
            else:
                feature_row[feature_list.index(token)] += 1

        result.append(feature_row)

    return result

def get_bigram_features(token_list_list):
    result = []

    for token_list in token_list_list:
        for i in range(len(token_list) - 1):
            token_one = token_list[i]
            token_two = token_list[i+1]
            bigram = f"{token_one} {token_two}"
            if bigram not in result:
                result.append(bigram)

    return result

def build_bigram_features(token_list_list, feature_list):
    result = []

    for token_list in token_list_list:
        feature_row = [0]*len(feature_list)
        for i in range(len(token_list) - 1):
            token_one = token_list[i]
            token_two = token_list[i + 1]
            bigram = f"{token_one} {token_two}"

            if bigram not in feature_list:
                # Shouldn't happen for the training set, but might happen for the testing set
                continue
            else:
                feature_row[feature_list.index(bigram)] = 1

        result.append(feature_row)

    return result

def build_bigram2_features(token_list_list, feature_list):
    result = []

    for token_list in token_list_list:
        feature_row = [0]*len(feature_list)
        for i in range(len(token_list) - 1):
            token_one = token_list[i]
            token_two = token_list[i + 1]
            bigram = f"{token_one} {token_two}"

            if bigram not in feature_list:
                # Shouldn't happen for the training set, but might happen for the testing set
                continue
            else:
                feature_row[feature_list.index(bigram)] += 1

        result.append(feature_row)

    return result

def get_trigram_features(token_list_list):
    result = []

    for token_list in token_list_list:
        for i in range(len(token_list) - 2):
            token_one = token_list[i]
            token_two = token_list[i+1]
            token_three = token_list[i+2]
            trigram = f"{token_one} {token_two} {token_three}"
            if trigram not in result:
                result.append(trigram)

    return result

def build_trigram_features(token_list_list, feature_list):
    result = []

    for token_list in token_list_list:
        feature_row = [0]*len(feature_list)
        for i in range(len(token_list) - 2):
            token_one = token_list[i]
            token_two = token_list[i+1]
            token_three = token_list[i+2]
            trigram = f"{token_one} {token_two} {token_three}"

            if trigram not in feature_list:
                # Shouldn't happen for the training set, but might happen for the testing set
                continue
            else:
                feature_row[feature_list.index(trigram)] = 1

        result.append(feature_row)

    return result

def build_trigram2_features(token_list_list, feature_list):
    result = []

    for token_list in token_list_list:
        feature_row = [0]*len(feature_list)
        for i in range(len(token_list) - 2):
            token_one = token_list[i]
            token_two = token_list[i+1]
            token_three = token_list[i+2]
            trigram = f"{token_one} {token_two} {token_three}"

            if trigram not in feature_list:
                # Shouldn't happen for the training set, but might happen for the testing set
                continue
            else:
                feature_row[feature_list.index(trigram)] += 1

        result.append(feature_row)

    return result

# https://www.kaggle.com/code/yommnamohamed/sentiment-analysis-using-sentiwordnet/notebook
def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def sentiwordnet_analysis(tokens_list):
    result = []

    for tokens in tokens_list:
        word_scores = []
        for token, tag in pos_tag(tokens):
            wn_tag = penn_to_wn(tag)

            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            synsets = wn.synsets(token, pos=wn_tag)
            if not synsets:
                continue

            # Take the most commpn
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())

            word_scores.append([swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()])

        tokens_score = [0] * 3
        for score in word_scores:
            # Want to make sure everything gets added
            try:
                tokens_score[0] += score[0]
            except:
                "Nothing happens"
            try:
                tokens_score[1] += score[1]
            except:
                "Nothing happens"
            try:
                tokens_score[2] += score[2]
            except:
                "Nothing happens"
        result.append(tokens_score)

    return result

def sentiwordnet2_analysis(tokens_list):
    result = []

    for tokens in tokens_list:
        pos_total = 0
        neg_total = 0
        obj_total = 0

        positive = 0
        negative = 0
        neutral = 0
        for token, tag in pos_tag(tokens):
            wn_tag = penn_to_wn(tag)

            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            synsets = wn.synsets(token, pos=wn_tag)
            if not synsets:
                continue

            # Take the most commpn
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())

            pos = swn_synset.pos_score()
            neg = swn_synset.neg_score()
            obj = swn_synset.obj_score()

            pos_total += pos
            neg_total += neg
            obj_total += obj

            if (pos > neg and pos > obj):
                positive += 1
            if (neg > pos and neg > obj):
                negative += 1
            else:
                neutral += 1

        polarity = positive/(negative + 1)
        subjectivity = (positive+negative)/(neutral + 1)
        result.append([pos_total, neg_total, obj_total, positive, negative, neutral, polarity, subjectivity])

    return result

def zip_X(X_list):
    result = []

    if len(X_list) == 0:
        return result
    # ensure that they are the same length
    length = len(X_list[0])
    for X in X_list:
        list_len = len(X)
        if list_len != length:
            print("X's lengths don't match")
            return result

    for j in range(length):
        line = []
        for i in range(len(X_list)):
            line += X_list[i][j]
        result.append(line)

    return result

def run_all_variants(confirmed_Xs, possible_sets, training_y, testing_y):
    if len(possible_sets) != 0:
        # First run without adding anything from this set
        result = run_all_variants(confirmed_Xs, possible_sets[1:], training_y, testing_y)
        # Next run by choosing any one from the next set
        next_set = possible_sets[0]
        for X in next_set:
            result += run_all_variants(confirmed_Xs + [X], possible_sets[1:], training_y, testing_y)

        return result
    else:
        if len(confirmed_Xs) == 0:
            # No sets to run
            return []
        else:
            # First set up our models
            available_models = {
                "Bernoulli": BernoulliNB(),
                "Multinomial": MultinomialNB(),
                "Gaussian": GaussianNB()
            }

            # Get all possible models that can be run, and set up the training and testing sets, get the printable name
            training_Xs = []
            testing_Xs = []
            possible_models = confirmed_Xs[0]["models"] # set
            basic_name = ""
            for X in confirmed_Xs:
                models = X["models"]
                possible_models = possible_models.intersection(models)
                training_Xs.append(X["training"])
                testing_Xs.append(X["testing"])
                basic_name += f"{X['name']}/"
            # Now that we have all of the models, we can zip up the final features and run them
            training_X = zip_X(training_Xs)
            testing_X = zip_X(testing_Xs)
            basic_name = basic_name[:-1] # remove the last slash

            result = []

            for model in possible_models:
                name = f"{model} {basic_name}"
                m = available_models[model]
                result.append(run_model(name, m, training_X, training_y, testing_X, testing_y))

            return result

def results_sort_helper(result):
    return result["accuracy"]

def clean_results(results):
    print("name,accuracy,f1,precision,recall")

    results.sort(key=results_sort_helper, reverse=True)

    best_confusion_matrix = results[0]["confusion-matrix"]

    for result in results:
        name = result["name"]
        accuracy = result["accuracy"]
        f1 = result["f1"]
        precision = result["precision"]
        recall = result["recall"]
        print(f"{name},{accuracy:.4f},{f1:.4f},{precision:.4f},{recall:.4f}")

    print("Best confusion matrix")

    print(best_confusion_matrix)

def main():
    parser = argparse.ArgumentParser(description='Run sentiment analysis on a dataset')
    parser.add_argument('--training-set', type=str, default='semeval2013-train.tsv')
    #parser.add_argument('--testing-set', type=str, default='semeval2013-test.tsv')
    parser.add_argument('--data-set', type=str, default='sentiment_randomized.tsv')
    args = parser.parse_args()

    print("Reading CSVs...")
    training_data = pandas.read_csv(args.training_set, sep='\t')
    #testing_data = pandas.read_csv(args.testing_set, sep='\t')
    eileen_gu_data = pandas.read_csv(args.data_set, sep='\t')

    print("Cleaning Data...")
    print("Train")
    training_inputs, training_y = clean_data(training_data)
    print("Test")
    #testing_inputs, testing_y = clean_data(testing_data)
    print("Gu")
    eileen_gu_inputs, eileen_gu_y = clean_data(eileen_gu_data)

    print("Getting tokens...")
    training_tokens = generate_tokens(training_inputs)
    training_cleaned_tokens = lemmatize_and_remove_stopwords(training_tokens)

    #testing_tokens = generate_tokens(testing_inputs)
    #testing_cleaned_tokens = lemmatize_and_remove_stopwords(testing_tokens)

    eileen_gu_tokens = generate_tokens(eileen_gu_inputs)

    eileen_gu_cleaned_tokens = lemmatize_and_remove_stopwords(eileen_gu_tokens)
    eileen_gu_training_tokens, eileen_gu_training_y, eileen_gu_testing_tokens, eileen_gu_testing_y = split_train_test(
        eileen_gu_tokens, eileen_gu_y)

    eileen_gu_cleaned_training_tokens = lemmatize_and_remove_stopwords(eileen_gu_training_tokens)
    eileen_gu_cleaned_testing_tokens = lemmatize_and_remove_stopwords(eileen_gu_testing_tokens)

    print("Getting Unigram Features...")

    unigram_features = get_unigram_features(training_tokens)

    #unigram_training_X = build_unigram_features(training_tokens, unigram_features)
    #unigram_testing_X = build_unigram_features(testing_tokens, unigram_features)

    unigram2_training_X = build_unigram2_features(training_tokens, unigram_features)
    #unigram2_testing_X = build_unigram2_features(testing_tokens, unigram_features)

    # Eileen Gu data
    eileen_gu_unigram_features = get_unigram_features(eileen_gu_training_tokens)

    #eileen_gu_unigram_training_X = build_unigram_features(eileen_gu_training_tokens, eileen_gu_unigram_features)
    #eileen_gu_unigram_testing_X = build_unigram_features(eileen_gu_testing_tokens, eileen_gu_unigram_features)

    eileen_gu_unigram2_training_X = build_unigram2_features(eileen_gu_training_tokens, eileen_gu_unigram_features)
    eileen_gu_unigram2_testing_X = build_unigram2_features(eileen_gu_testing_tokens, eileen_gu_unigram_features)

    # Mix (train semeval, test eileen_gu)

    #mixed_unigram_testing_X = build_unigram_features(eileen_gu_tokens, unigram_features)
    mixed_unigram2_testing_X = build_unigram2_features(eileen_gu_tokens, unigram_features)

    print("Getting Bigram Features...")

    bigram_features = get_bigram_features(training_tokens)
    #bigram_training_X = build_bigram_features(training_tokens, bigram_features)
    #bigram_testing_X = build_bigram_features(testing_tokens, bigram_features)

    bigram2_training_X = build_bigram2_features(training_tokens, bigram_features)
    #bigram2_testing_X = build_bigram2_features(testing_tokens, bigram_features)

    # Eileen Gu data
    eileen_gu_bigram_features = get_bigram_features(eileen_gu_training_tokens)

    #eileen_gu_bigram_training_X = build_bigram_features(eileen_gu_training_tokens, eileen_gu_bigram_features)
    #eileen_gu_bigram_testing_X = build_bigram_features(eileen_gu_testing_tokens, eileen_gu_bigram_features)

    eileen_gu_bigram2_training_X = build_bigram2_features(eileen_gu_training_tokens, eileen_gu_bigram_features)
    eileen_gu_bigram2_testing_X = build_bigram2_features(eileen_gu_testing_tokens, eileen_gu_bigram_features)

    # Mix (train semeval, test eileen_gu)

    mixed_bigram_testing_X = build_bigram_features(eileen_gu_tokens, bigram_features)
    mixed_bigram2_testing_X = build_bigram2_features(eileen_gu_tokens, bigram_features)

    print("Getting Trigram Features...")

    trigram_features = get_trigram_features(training_tokens)

    #trigram_training_X = build_trigram_features(training_tokens, trigram_features)
    #trigram_testing_X = build_trigram_features(testing_tokens, trigram_features)

    trigram2_training_X = build_trigram2_features(training_tokens, trigram_features)
    #trigram2_testing_X = build_trigram2_features(testing_tokens, trigram_features)

    # Eileen Gu data
    eileen_gu_trigram_features = get_trigram_features(eileen_gu_training_tokens)

    #eileen_gu_trigram_training_X = build_trigram_features(eileen_gu_training_tokens, eileen_gu_trigram_features)
    #eileen_gu_trigram_testing_X = build_trigram_features(eileen_gu_testing_tokens, eileen_gu_trigram_features)

    eileen_gu_trigram2_training_X = build_trigram2_features(eileen_gu_training_tokens, eileen_gu_trigram_features)
    eileen_gu_trigram2_testing_X = build_trigram2_features(eileen_gu_testing_tokens, eileen_gu_trigram_features)

    # Mix (train semeval, test eileen_gu)

    #mixed_trigram_testing_X = build_trigram_features(eileen_gu_tokens, trigram_features)
    mixed_trigram2_testing_X = build_trigram2_features(eileen_gu_tokens, trigram_features)

    print("Getting Sentiwordnet Features...")

    #sentiwordnet_training_X = sentiwordnet_analysis(training_cleaned_tokens)
    #sentiwordnet_testing_X = sentiwordnet_analysis(testing_cleaned_tokens)

    sentiwordnet2_training_X = sentiwordnet2_analysis(training_cleaned_tokens)
    #sentiwordnet2_testing_X = sentiwordnet2_analysis(testing_cleaned_tokens)

    # Now that we have all three

    # Eileen Gu data

    #eileen_gu_sentiwordnet_training_X = sentiwordnet_analysis(eileen_gu_cleaned_training_tokens)
    #eileen_gu_sentiwordnet_testing_X = sentiwordnet_analysis(eileen_gu_cleaned_testing_tokens)

    eileen_gu_sentiwordnet2_training_X = sentiwordnet2_analysis(eileen_gu_cleaned_training_tokens)
    eileen_gu_sentiwordnet2_testing_X = sentiwordnet2_analysis(eileen_gu_cleaned_testing_tokens)

    # Mix (train semeval, test eileen_gu)

    #mixed_sentiwordnet_testing_X = sentiwordnet_analysis(eileen_gu_cleaned_tokens)
    mixed_sentiwordnet2_testing_X = sentiwordnet2_analysis(eileen_gu_cleaned_tokens)
    '''
    print("Running Semeval models...")

    possible_sets = [
        [
            # {
            #    "name": "Unigram",
            #    "training": unigram_training_X,
            #    "testing": unigram_testing_X,
            #    "models": {"Bernoulli", "Multinomial"}
            # },
            {
                "name": "Unigram*",
                "training": unigram2_training_X,
                "testing": unigram2_testing_X,
                "models": {"Multinomial"}  # Not binary anymore
            }
        ],
        [
            # {
            #    "name": "Bigram",
            #    "training": bigram_training_X,
            #    "testing": bigram_testing_X,
            #    "models": {"Bernoulli", "Multinomial"}
            # },
            {
                "name": "Bigram*",
                "training": bigram2_training_X,
                "testing": bigram2_testing_X,
                "models": {"Multinomial"}
            }
        ],
        [
            # {
            #    "name": "Trigram",
            #    "training": trigram_training_X,
            #    "testing": trigram_testing_X,
            #    "models": {"Bernoulli", "Multinomial"}
            # },
            {
                "name": "Trigram*",
                "training": trigram2_training_X,
                "testing": trigram2_testing_X,
                "models": {"Multinomial"}
            }
        ],
        [
            # {
            #    "name": "Sentiwordnet",
            #    "training": sentiwordnet_training_X,
            #    "testing": sentiwordnet_testing_X,
            #    "models": {"Multinomial"}
            # },
            {
                "name": "Sentiwordnet*",
                "training": sentiwordnet2_training_X,
                "testing": sentiwordnet2_testing_X,
                "models": {"Multinomial"}
            }
        ]
    ]

    # results = run_all_variants([], possible_sets, training_y, testing_y)
    '''
    print("Running EileenGu models...")

    possible_eileen_gu_sets = [
        [
            # {
            #    "name": "Unigram",
            #    "training": eileen_gu_unigram_training_X,
            #    "testing": eileen_gu_unigram_testing_X,
            #    "models": {"Bernoulli", "Multinomial"}
            # },
            {
                "name": "Unigram*",
                "training": eileen_gu_unigram2_training_X,
                "testing": eileen_gu_unigram2_testing_X,
                "models": {"Multinomial"}  # Not binary anymore
            }
        ],
        [
            # {
            #    "name": "Bigram",
            #    "training": eileen_gu_bigram_training_X,
            #    "testing": eileen_gu_bigram_testing_X,
            #    "models": {"Bernoulli", "Multinomial"}
            # },
            {
                "name": "Bigram*",
                "training": eileen_gu_bigram2_training_X,
                "testing": eileen_gu_bigram2_testing_X,
                "models": {"Multinomial"}
            }
        ],
        [
            # {
            #    "name": "Trigram",
            #    "training": eileen_gu_trigram_training_X,
            #    "testing": eileen_gu_trigram_testing_X,
            #    "models": {"Bernoulli", "Multinomial"}
            # },
            {
                "name": "Trigram*",
                "training": eileen_gu_trigram2_training_X,
                "testing": eileen_gu_trigram2_testing_X,
                "models": {"Multinomial"}
            }
        ],
        [
            # {
            #    "name": "Sentiwordnet",
            #    "training": eileen_gu_sentiwordnet_training_X,
            #    "testing": eileen_gu_sentiwordnet_testing_X,
            #    "models": {"Multinomial"}
            # },
            {
                "name": "Sentiwordnet*",
                "training": eileen_gu_sentiwordnet2_training_X,
                "testing": eileen_gu_sentiwordnet2_testing_X,
                "models": {"Multinomial"}
            }
        ]
    ]

    eileen_gu_results = run_all_variants([], possible_eileen_gu_sets, eileen_gu_training_y, eileen_gu_testing_y)

    print("Running Semeval train/ Eileen Gu test models...")

    mixed_possible_sets = [
        [
            # {
            #    "name": "Unigram",
            #    "training": unigram_training_X,
            #    "testing": mixed_unigram_testing_X,
            #    "models": {"Bernoulli", "Multinomial"}
            # },
            {
                "name": "Unigram*",
                "training": unigram2_training_X,
                "testing": mixed_unigram2_testing_X,
                "models": {"Multinomial"}  # Not binary anymore
            }
        ],
        [
            # {
            #    "name": "Bigram",
            #    "training": bigram_training_X,
            #    "testing": mixed_bigram_testing_X,
            #    "models": {"Bernoulli", "Multinomial"}
            # },
            {
                "name": "Bigram*",
                "training": bigram2_training_X,
                "testing": mixed_bigram2_testing_X,
                "models": {"Multinomial"}
            }
        ],
        [
            # {
            #    "name": "Trigram",
            #    "training": trigram_training_X,
            #    "testing": mixed_trigram_testing_X,
            #    "models": {"Bernoulli", "Multinomial"}
            # },
            {
                "name": "Trigram*",
                "training": trigram2_training_X,
                "testing": mixed_trigram2_testing_X,
                "models": {"Multinomial"}
            }
        ],
        [
            # {
            #    "name": "Sentiwordnet",
            #    "training": sentiwordnet_training_X,
            #    "testing": mixed_sentiwordnet_testing_X,
            #    "models": {"Multinomial"}
            # },
            {
                "name": "Sentiwordnet*",
                "training": sentiwordnet2_training_X,
                "testing": mixed_sentiwordnet2_testing_X,
                "models": {"Multinomial"}
            }
        ]
    ]

    mixed_results = run_all_variants([], mixed_possible_sets, training_y, eileen_gu_y)

    print("Printing results...")

    # clean_results(results)
    clean_results(eileen_gu_results)
    clean_results(mixed_results)

    print("Done!")


if __name__ == "__main__":
    # calling main function
    main()
