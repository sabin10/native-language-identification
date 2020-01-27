'''
Logistic Regression
'''

import numpy as np
import re
import os
import time
import math
from collections import defaultdict
from sklearn import svm
from nltk.util import ngrams
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

FIRST_N_WORDS = 1000
FIRST_N_KEYS = 700

FIRST_N_1GRAM = 2000
FIRST_N_2GRAM = 500
FIRST_N_3GRAM = 0

'''
READ TRAIN LABELS
'''
dir_path = 'trainData/'
labels = np.loadtxt(os.path.join(dir_path, 'labels_train.txt'))


def accuracy(y, p):
    return 100 * (y == p).astype('int').mean()

def files_in_folder(mypath):
    fisiere = []
    for f in os.listdir(mypath):
        if os.path.isfile(os.path.join(mypath, f)):
            fisiere.append(os.path.join(mypath, f))
    return sorted(fisiere)

def extrage_fisier_fara_extensie(cale_catre_fisier):
    nume_fisier = os.path.basename(cale_catre_fisier)
    nume_fisier_fara_extensie = nume_fisier.replace('.txt', '')
    return nume_fisier_fara_extensie

'''
class_dots_arr[i] = number of dots for i-th class
'''
class_dots_arr = np.zeros(11)
class_commas_arr = np.zeros(11)
class_lines_arr = np.zeros(11)
class_apostrophes_arr = np.zeros(11)


def citeste_texte_din_director(cale):
    date_text = []
    iduri_text = []
    two_gram_list = []
    three_gram_list =[]


    '''
    arr_dots[i] = number of dots for i-th train text
    '''
    arr_dots = np.zeros(2983)
    arr_commas = np.zeros(2983)
    arr_lines = np.zeros(2983)
    arr_apostrophes = np.zeros(2983)

    index = 0
    for fis in files_in_folder(cale):
        id_fis = extrage_fisier_fara_extensie(fis)
        iduri_text.append(id_fis)
        with open(fis, 'r', encoding='utf-8') as fin:
            text = fin.read()

        num_dots = text.count('.')
        num_commas = text.count(',')
        num_lines = text.count('\n')
        num_apostrophes = text.count("'")

        class_dots_arr[int(labels[index])] += num_dots
        class_commas_arr[int(labels[index])] += num_commas
        class_lines_arr[int(labels[index])] += num_lines
        class_apostrophes_arr[int(labels[index])] += num_apostrophes

        arr_dots[index] = num_dots
        arr_commas[index] = num_commas
        arr_lines[index] = num_lines
        arr_apostrophes[index] = num_apostrophes

        text_fara_punct = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
        cuvinte_text = text_fara_punct.split()
        # bow preprocess -> lowercase
        cuvinte_text = [x.lower() for x in cuvinte_text]
        two_gram_list += generate_ngram(text_fara_punct, 2)
        three_gram_list += generate_ngram(text_fara_punct, 3)
        date_text.append(cuvinte_text)
        index += 1
    return (iduri_text, date_text, two_gram_list, three_gram_list, arr_dots, arr_commas, arr_lines, arr_apostrophes)

def generate_ngram(s, n):
    s = s.lower()
    # s = re.sub("[-.,;:!?\"\'\/()_*=`]", "", s)
    tokens = [token for token in s.split(" ") if token != ""]
    output = list(ngrams(tokens, n))
    return output


'''
:return @array(0,10) for percentage of uses in a specific label
'''
def transform_vector_frequency_into_percentage(freq):
    percentage_vector = np.zeros(11)
    total_sum = sum(freq)

    for i, num in enumerate(freq):
        percentage_vector[i] = (num*100) / total_sum
    return percentage_vector

'''
return True if take one_gram under considration
'''
def is_n_gram_considerable(freq_vector, is_one_gram):
    # 3.5 and 2.0
    limit = 3.5 if is_one_gram else 2.0
    percentage_vector = transform_vector_frequency_into_percentage(freq_vector)
    for percentage in percentage_vector:
        if percentage < 9.09 - limit or percentage > 9.09 + limit:
            return True
    return False



'''
READ TRAIN DATA
'''
train_data_path = os.path.join(dir_path, 'trainExamples')
iduri_train, data_train, two_gram_list, three_gram_list, arr_dots_train, arr_commas_train, arr_lines_train, arr_apostrophes_train = citeste_texte_din_director(train_data_path)

# '''
# READ TEST DATA
# '''
# ids_test, data_test, _, _, arr_dots_test, arr_commas_test, arr_lines_test, arr_apostrophes_test = citeste_texte_din_director('testData')


dict_one_gram_freq_vector = {}
dict_two_gram_freq_vector = {}
def write_dicts_n_gram_freq_vector():
    for i, doc in enumerate(data_train):
        label = int(labels[i])

        for i in range(len(doc)):
            if doc[i] in dict_one_gram_freq_vector:
                freq_vector = dict_one_gram_freq_vector[doc[i]]
                freq_vector[label] += 1
                dict_one_gram_freq_vector[doc[i]] = freq_vector
            else:
                freq_vector = np.zeros(11)
                freq_vector[label] += 1
                dict_one_gram_freq_vector[doc[i]] = freq_vector

            if i < len(doc)-1:
                two_gram_key = doc[i] + doc[i+1]
                if two_gram_key in dict_two_gram_freq_vector:
                    freq_vector = dict_two_gram_freq_vector[two_gram_key]
                    freq_vector[label] += 1
                    dict_two_gram_freq_vector[two_gram_key] = freq_vector
                else:
                    freq_vector = np.zeros(11)
                    freq_vector[label] += 1
                    dict_two_gram_freq_vector[two_gram_key] = freq_vector
    return

write_dicts_n_gram_freq_vector()

set_banned_one_gram = set()
set_banned_two_gram = set()

def complete_sets_banned_n_gram():
    for key, value_freq in dict_one_gram_freq_vector.items():
        if (is_n_gram_considerable(value_freq, is_one_gram=True) is False):
            set_banned_one_gram.add(key)

    for key, value_freq in dict_two_gram_freq_vector.items():
        if (is_n_gram_considerable(value_freq, is_one_gram=False) is False):
            set_banned_two_gram.add(key)
    return

complete_sets_banned_n_gram()

dict_one_gram = defaultdict(int)
dict_two_gram = defaultdict(int)
dict_three_gram = defaultdict(int)

for doc in data_train:
    for i in range(len(doc)):
        key_one_gram = doc[i]
        # decide if consider curr_one_gram
        if key_one_gram not in set_banned_one_gram:
            dict_one_gram[key_one_gram] += 1
        if i < len(doc) - 1:
            key_two_gram = doc[i] + doc[i+1]
            # sometimes happens 1gram = 2gram
            if key_two_gram in dict_one_gram.keys():
                continue

            if key_two_gram not in set_banned_two_gram:
                dict_two_gram[key_two_gram] += 1

        if i < len(doc) - 2:
            key_three_gram = doc[i] + doc[i+1] + doc[i+2]
            dict_three_gram[key_three_gram] += 1


'''
list of tuples
'''
one_gram_frequence = list(dict_one_gram.items())
one_gram_frequence = sorted(one_gram_frequence, key=lambda kv: kv[1], reverse=True)

two_gram_frequence = list(dict_two_gram.items())
two_gram_frequence = sorted(two_gram_frequence, key=lambda kv: kv[1], reverse=True)

three_gram_frequence = list(dict_three_gram.items())
three_gram_frequence = sorted(three_gram_frequence, key=lambda kv: kv[1], reverse=True)

# select only first N for both one and two gram
selected_one_gram = one_gram_frequence[0:FIRST_N_1GRAM]
selected_two_gram = two_gram_frequence[0:FIRST_N_2GRAM]
selected_three_gram = three_gram_frequence[0:FIRST_N_3GRAM]

list_of_selected_two_gram = []
for two_gram, frequency in selected_two_gram:
    list_of_selected_two_gram.append(two_gram)

list_of_selected_one_gram = []
for one_gram, frequency in selected_one_gram:
    list_of_selected_one_gram.append(one_gram)

list_of_selected_three_gram = []
for three_gram, frequency in selected_three_gram:
    list_of_selected_three_gram.append(three_gram)



'''
return Bag-Of-Words @dict for each document in the data

for test change choose_dict_one_gram_freq

length_list_data = len(whole data in list of lists)
list_text = only the current list from list_data
selected_list_one_gram = list with only selected to be considerated one_grams
selected_list_two_gram = list with only selected to be considerated two_grams
'''
def get_bow_for_one_text(length_list_data,
                         list_text,
                         selected_list_one_gram,
                         selected_list_two_gram,
                         selected_list_three_gram,
                         choose_dict_one_gram_freq=dict_one_gram,
                         choose_dict_two_gram_freq=dict_two_gram):
    dict = {}
    for key in selected_list_one_gram:
        dict[key] = 0
    for key in selected_list_two_gram:
        dict[key] = 0
    for key in selected_list_three_gram:
        dict[key] = 0

    for i in range(len(list_text)):
        one_gram_key = list_text[i]
        if one_gram_key in selected_list_one_gram:
            dict[one_gram_key] += 1

        if i < len(list_text)-1:
            two_gram_key = list_text[i] + list_text[i+1]
            if two_gram_key in selected_list_two_gram:
                dict[two_gram_key] += 1

        if i < len(list_text) - 2:
            three_gram_key = list_text[i] + list_text[i+1] + list_text[i+2]
            if three_gram_key in selected_list_three_gram:
                dict[three_gram_key] += 1

    return dict




def compute_bow_for_all_data(list_data, selected_list_one_gram, selected_list_two_gram, arr_dots, arr_commas, arr_lines, arr_apostrophes):

    # + 4 cand adaug punctuatii
    bow = np.zeros((len(list_data), len(selected_one_gram)+len(selected_two_gram)+4 + len(selected_three_gram)))
    for index, list_text in enumerate(list_data):
        text_bow = get_bow_for_one_text(length_list_data=len(list_data),
                                        list_text=list_text,
                                        selected_list_one_gram=list_of_selected_one_gram,
                                        selected_list_two_gram=list_of_selected_two_gram,
                                        selected_list_three_gram=list_of_selected_three_gram)
        v = np.array(list(text_bow.values()))


        # v = v / np.sqrt(np.sum(v ** 2))
        # v = v * 100
        v = np.insert(v, 0, arr_apostrophes[index])
        v = np.insert(v, 0, arr_lines[index])
        v = np.insert(v, 0, arr_commas[index])
        v = np.insert(v, 0, arr_dots[index])
        v = v / np.sqrt(np.sum(v ** 2))
        v = v * 100
        # print("v = ", index, v)
        bow[index] = v
    return bow

all_data_train_bow = compute_bow_for_all_data(list_data=data_train,
                                              selected_list_one_gram=list_of_selected_one_gram,
                                              selected_list_two_gram=list_of_selected_two_gram,
                                              arr_dots=arr_dots_train,
                                              arr_commas=arr_commas_train,
                                              arr_lines=arr_lines_train,
                                              arr_apostrophes=arr_apostrophes_train)




nr_ex_train = 2000
nr_ex_valid = 500
nr_ex_test = len(data_train) - (nr_ex_train + nr_ex_valid)

# 0 -> nr_ex_train
indices_train = np.arange(0, nr_ex_train)
# nr_ex_train -> nr_ex_train+nr_ex_valid
indices_valid = np.arange(nr_ex_train, nr_ex_train + nr_ex_valid)
# nr_ex_train + nr_ex_valid -> final
indices_test = np.arange(nr_ex_train + nr_ex_valid, len(data_train))

indices_train_valid = np.concatenate([indices_train, indices_valid])

'''
submission logistic regression
'''

model = LogisticRegression()
print("Start time")
start_time = time.time()
model.fit(all_data_train_bow[indices_train, :], labels[indices_train])
print("End time")
elapsed_time = time.time() - start_time
print("elapsed time ", elapsed_time)
predicted_classes = model.predict(all_data_train_bow[indices_valid, :])
accuracy = accuracy_score(labels[indices_valid], predicted_classes)
print('The accuracy score using scikit-learn is {}'.format(accuracy))