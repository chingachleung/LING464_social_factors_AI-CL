import spacy
import math
from nltk import word_tokenize
import string
import itertools
from collections import Counter
from collections import defaultdict
from operator import itemgetter

class ColloPMI:
    """
    create an object that accepts a text_file, a threshold number the capture words with
    minimal word frequency, and default_data_set = 'en_core_web_lg'
    """
    def __init__(self,text_file,threshold,lanuage_data_set='en_core_web_lg'):
        self.model = spacy.load(lanuage_data_set)
        self.threshold = threshold
        self.text_file = text_file
        self.doc_list = self.__preprocess()
        self.frequent_word_dict = self.__create_frequent_word_dict()
        self.frequent_doc_list = self.__doc_list_with_frequent_only()

    def __preprocess(self):
        """
        pre-process the text file by removing \n,stop-words, punctuations, and turning words into their lemmas,tokenize
        :param
        :return: a list that contains lists of words correspond to the premises/hypotheses in each line
        """
        stop_words = self.model.Defaults.stop_words
        doc_list = []
        with open(self.text_file,'r') as file:
            for line in file:
                #strip off the \n at the end of the sentences
                line = line.strip() # remove the \n at the end of the line
                
                #remove punctuations
                line = line.translate(str.maketrans('','',string.punctuation))
                
                #tokenize the line
                line = [w.lower() for w in word_tokenize(line)]
                
                #remove stop words
                line = [w for w in line if w not in stop_words]
                
                #combine to become string for spacy doc process : only accept strings
                line = " ".join(w for w in line)

                no_stop_doc = []
                doc = self.model(line)
                for token in doc:
                     #only use lemmas to avoid variations
                     if len(token) > 1:
                        no_stop_doc.append(token.lemma_)
                doc_list.append(no_stop_doc)
                
        return doc_list


    def __create_frequent_word_dict(self):
        """
        turns a list of word-lists into q frequency_dict with words' frequency less than threshold removed
        :param document_list
        :return: frequent_word list and frequent_word_dict
        """
        # combining lists of words into one list
        concatenated_doc_list = list(itertools.chain.from_iterable(self.doc_list))
        
        # calculate all word frequency
        all_word_frequency_dict = Counter(concatenated_doc_list)
        only_frequent_word_list = [w for w in concatenated_doc_list if all_word_frequency_dict[w] >= self.threshold]
        frequent_word_dict = Counter(only_frequent_word_list)
        return frequent_word_dict

    def __doc_list_with_frequent_only(self):
        """
        returns a list of word_lists (which represent each document) whose words >= number threshold
        :param doc_list:
        :param frequent_word_dict:
        :return: a list of docs whose words are >= threshold number
        """
        frequent_doc_list = []
        for doc in self.doc_list:
            #print(f"printing the doc that has been preprocessd: {doc}")
            frequent_word_doc = [w for w in doc if w in self.frequent_word_dict.keys()]
            #print(f"printing doc that only have frequent words: {frequent_word_doc}")
            if len(frequent_word_doc) > 1:
                frequent_doc_list.append(frequent_word_doc)
        return frequent_doc_list

    def pmi_probability(self,label, top_count=5):
        """

        :param top_count: defaullt num value for how many of the top frequently existing combinations to return
        :return: a dictionary with tuples(2 words) as keys, PMI as value
        """
        label = label.lower()
        collo_dict = defaultdict(int)  # stores number of times two words appear together
        N = len(self.frequent_doc_list)
        label_count = self.frequent_word_dict[label]# the count should be more than you check on txt since i use lemma
        pmi_dict = {}
        
        for doc in self.frequent_doc_list:
            if label not in doc:
                continue
            unique_doc = list(set(doc)) # remove duplicates of words
            for word in unique_doc:
                if word != label:
                    collo_dict[label, word] += 1
                    
        for word_combination, f in collo_dict.items():
            together_count = f
            word_count = self.frequent_word_dict[word_combination[1]]
            pmi = math.log2((N * together_count) / (word_count * label_count))
            pmi_dict[word_combination] = pmi

        highest_pmi = dict(sorted(pmi_dict.items(), key=itemgetter(1), reverse=True)[:top_count])
        return pmi_dict, highest_pmi
