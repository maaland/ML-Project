import tempfile
import tarfile
import urllib2
import pickle
import os
import random
import re
import codecs

import numpy as np # for numpy arrays

CACHE_FILE = "dataset.pkl"
REGEX_EMAIL = re.compile(r"[\+\w\.-]+@[\w\.-]+")
REGEX_QUOTE = re.compile(r"(writes in|writes:|wrote:|says:|said:|^>|^\|)")

def do_truncate_label(doc, n):
    new_label = ".".join(doc[1].split(".")[:n])
    return (doc[0], new_label, doc[2])

def do_remove_header(doc):
    _, _, after = doc[2].partition("\n\n")
    return (doc[0], doc[1], after)

def do_remove_emails(doc):
    replaced = REGEX_EMAIL.sub("EMAIL", doc[2])
    return (doc[0], doc[1], replaced)

def do_remove_quotes(doc):
    # loop through all lines and filter out anything that
    # matches the QUOTE_REGEX
    lines = [line for line in doc[2].split("\n") if not REGEX_QUOTE.search(line)]
    return (doc[0], doc[1], "\n".join(lines))

def get(subset="all", categories=None, truncate_label=0, preprocess=True, verbose=False):
    """
    query the dataset, returns y, X where X is an array of documents (text)
    and y is an array of labels (usenet groups) corresponding to those
    documents.
    """
    docs = None

    # try to grab from cache
    if os.path.isfile(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            try:
                docs = pickle.load(f)
            except:
                pass

    if not docs:
        # if cache didn't exist, download the dataset
        if verbose:
            print "Downloading dataset"
        docs = download()
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(docs, f)

    if truncate_label > 0:
        docs = [do_truncate_label(doc, truncate_label) for doc in docs]
        
    all_categories = set([doc[1] for doc in docs])

    # parameter juggling
    if subset == "all":
        subset = ("train", "test")
    else:
        subset = (subset)

    if not isinstance(categories, (np.ndarray, list, int)):
        categories = all_categories
    elif isinstance(categories, int):
        categories = random.sample(all_categories, categories)

    # filtering
    docs = [doc for doc in docs if doc[0] in subset and doc[1] in categories]

    # preprocessing
    if preprocess:
        # remove headers
        if preprocess == True or "headers" in preprocess:
            if verbose:
                print "- Preprocessing: Removing headers"
            docs = [do_remove_header(doc) for doc in docs]

        # replace email addresses
        if preprocess == True or "emails" in preprocess:
            if verbose:
                print "- Preprocessing: Removing emails"
            docs = [do_remove_emails(doc) for doc in docs]

        # remove quotes
        if preprocess == True or "quotes" in preprocess:
            if verbose:
                print "- Preprocessing: Removing quotes"
            docs = [do_remove_quotes(doc) for doc in docs]

    if verbose:
        print "labels:", list(set([doc[0] for doc in docs]))
        print "size:", len(docs)
        print "categories looked at:", sorted(list(categories))
        print "categories found:", sorted(list(set([doc[1] for doc in docs])))

    y = [doc[1] for doc in docs]
    X = [doc[2] for doc in docs]

    return np.array(y), X

def download():
    """
    download fetches the tar.gz file of the dataset, extracts
    it and then returns an array of (subset, group, data) triplets
    where subset is either train or test, group is the usenet group
    name and data contains the actual documents
    """

    url = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
    data = []

    # use a tempfile to store the tar.gz file in
    with tempfile.TemporaryFile() as tmp:
        res = urllib2.urlopen(url)
        tmp.write(res.read())
        tmp.seek(0)

        # open the file as a tar and extract all documents
        with tarfile.open(fileobj=tmp) as tar:
            for f in tar.getmembers():
                # is f a regular file?
                if not f.isreg():
                    continue
                
                (root, group, _) = f.name.split("/")
                subset = "train" if "train" in root else "test"
                doc = tar.extractfile(f).read()
                # only keep documents that the vectorizers can decode
                try:
                    doc = codecs.decode(doc, "ascii")
                except:
                    continue
                data += [(subset, group, doc)]
    return data

if __name__ == "__main__":
    get(subset="all", categories=2, verbose=True)
    print 80*"-"
    get(subset="all", categories=5, truncate_label=1, verbose=True)
    print 80*"-"
    get(subset="all", truncate_label=2, verbose=True)
    print 80*"-"
    get(subset="all", truncate_label=2, preprocess=True, verbose=True)

