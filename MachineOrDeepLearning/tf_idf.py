"""
the file is to show how the tf-idf is calculated.
the main steps are as following:
    1. calculate tf, or word_frequency
    2. calculate idf
        a. caculate df
        b. inverse to get idf
    3. use tf and idf to get tf-idf
"""
import os
import numpy as np


def get_files_in_a_root(root):
    """
    a sample of get all the files in the root folder.
    """
    files_all = []
    for root, dirs, files in os.walk(root, topdown=True):  # use os.walk
        files_all += files
    return files_all  # only basename

def tf(files):
    """
    files: a file list
    return: a word count 
    """
    word_count = {}
    for file_in in files:
        f = open(file_in, "r", encoding="utf-8")
        lines = f.readlines()
        f.close()

        lines = [line.strip() for line in lines]
        
        for line in lines:
            words = line.split()  # suppose line is tokenized
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1  # dict.get
    return word_count


def TfIdf(path, tf):
    """
    path: a 2-level folder
    tf: word count dict, caculated already.
    
    mainly cal idf, which including 2 params:
    1. docuemnt count
    2. key_including document count
    """

    
    count = 0  # count files
    fold_list = os.listdir(path)
    
    # init tf_idf
    df, idf, tf_idf= {}, {}, {}
    for key in tf.keys():
        df[key] = 1
        tf_idf[key] = 0
    
    # iterate to get document_count and df[key]
    for fold in fold_list:  # get count and idf count
        file_list = os.listdir(path + '/' + fold)
        count += len(file_list)  # count
        
        for file in file_list:
            with open(path + '/' + fold + '/' + file) as f:
                text = f.read()
                for key in tf.keys():
                    if key in text:  # if key in document
                        df[key] += 1  # df count
    # inverse 
    for key, value in df.items():
        idf[key] = np.log(count / (value + 1))  # log(file_count/idf_count)
    
    # caculate tf_idf
    for key, value in tf.items():
        tf_idf[key] = tf[key] * idf[key]
    
    return tf_idf

if __name__ == "__main__":
    root = "/ai/data"
    files = get_files_in_a_root(root)
    print(files)
