import nltk # also import stopwords data package via downloader if missing
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Initialize dictionary of directory & filename
    d = dict()

    # Iterate thru files in directory to append to dictionary
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), encoding="utf-8") as ofile:
            d[file] = ofile.read()

    return d


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # Tokenize all words in the document and lowercase them
    tokenized_w = nltk.tokenize.word_tokenize(document.lower())

    # Filter out punctuations and stopwords  
    filtered_w = [x for x in tokenized_w if x not in string.punctuation and x not in nltk.corpus.stopwords.words("english")]

    return filtered_w

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    # Initialize words dictionary
    words = dict()

    # Iterate thru documents to count unique words in given documents
    for f in documents:
        unique_words = set()

        for w in documents[f]:
            if w not in unique_words:
                unique_words.add(w)
                try:
                    words[w] += 1
                except KeyError:
                    words[w] = 1

    return {w: math.log(len(documents) / words[w]) for w in words}


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Initialize dictionary of tf-idf of query
    scores = dict()

    # Iterate thru given files and compare each word with given query 
    for filename, filecontent in files.items():
        file_score = 0
        for w in query: # if word is in the query calculate 
            if w in filecontent:
                file_score += filecontent.count(w) * idfs[w]
        if file_score != 0:
            scores[filename] = file_score

    #Sort according to tf-idf
    sorted_scores = [k for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    return sorted_scores[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    # Initialize scores dictionary
    scores = dict()

    # Iterate thru given sentences check if the query matches
    for sentence, sentwords in sentences.items():
        score = 0
        for w in query:
            if w in sentwords:
                score += idfs[w]

        if score != 0:
            density = sum([sentwords.count(x) for x in query]) / len(sentwords)
            scores[sentence] = (score, density)

    #Sort according to tf-idf
    sorted_scores = [k for k, v in sorted(scores.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)]

    return sorted_scores[:n]


if __name__ == "__main__":
    main()