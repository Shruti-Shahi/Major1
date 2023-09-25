import json
import os
import stanza 
from nltk.parse.stanford import StanfordParser
from nltk.tree import *
from six.moves import urllib
import zipfile
import sys
import time
import ssl
from extract import video_to_transcript
import stanza
import pprint 
from flask import Flask,request,render_template,send_from_directory,jsonify

ssl._create_default_https_context = ssl._create_unverified_context

app =Flask(__name__,static_folder='static', static_url_path='')

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
os.environ['CLASSPATH'] = os.path.join(BASE_DIR, 'stanford-parser-full-2018-10-17')
os.environ['STANFORD_MODELS'] = os.path.join(BASE_DIR,
                                             'stanford-parser-full-2018-10-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
os.environ['NLTK_DATA'] = '/usr/local/share/nltk_data/'


def is_parser_jar_file_present():
    stanford_parser_path = os.environ.get('CLASSPATH') + ".jar"
    return os.path.exists(stanford_parser_path)


def reporthook(count, block, size):
    global start
    if count == 0:
        start = time.perf_counter()
        return
    duration = time.perf_counter() - start
    progress_size = int(count * block)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block * 100 / size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download_parser_jar_file():
    stanford_parser_path = os.environ.get('CLASSPATH') + ".jar"
    url = "https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip"
    urllib.request.urlretrieve(url, stanford_parser_path, reporthook)


def extract_parser_jar_file():
    stanford_parser_path = os.environ.get('CLASSPATH') + ".jar"
    try:
        with zipfile.ZipFile(stanford_parser_path) as z:
            z.extractall(path=BASE_DIR)
    except Exception:
        os.remove(stanford_parser_path)
        download_parser_jar_file()
        extract_parser_jar_file()


def extract_models_jar_file():
    stanford_models_path = os.path.join(os.environ.get('CLASSPATH'), 'stanford-parser-3.9.2-models.jar')
    stanford_models_dir = os.environ.get('CLASSPATH')
    with zipfile.ZipFile(stanford_models_path) as z:
        z.extractall(path=stanford_models_dir)


def download_required_packages():
    if not os.path.exists(os.environ.get('CLASSPATH')):
        if is_parser_jar_file_present():
            pass
        else:
            download_parser_jar_file()
        extract_parser_jar_file()

    if not os.path.exists(os.environ.get('STANFORD_MODELS')):
        extract_models_jar_file()


en_nlp = stanza.Pipeline('en', processors={'tokenize': 'spacy'})
stop_words = {"am", "are", "is", "was", "were", "be", "being", "been", "have", "has", "had", "does", "did", "could",
              "should", "would", "can", "shall", "will", "may", "might", "must", "let", "a", "the", "an", "The", "A",
              "An"}


sent = []
sent_extra = []
word = []
word_extra = []


def convert_to_sentence_list(text):
    for sentence in text.sentences:
        sent.append(sentence.text)
        sent_extra.append(sentence)


def convert_to_word_list(sentences):
    temp = []
    temp_extra = []
    for sentence in sentences:
        for w in sentence.words:
            temp.append(w.text)
            temp_extra.append(w)
        word.append(temp.copy())
        word_extra.append(temp_extra.copy())
        temp.clear()
        temp_extra.clear()


# stop words are removed
def filter_words(input_word):
    temp = []
    finalword = []
    for words in input_word:
        temp.clear()
        for w in words:
            if w not in stop_words:
                temp.append(w)
        finalword.append(temp.copy())
    for words in word_extra:
        for i, w in enumerate(words):
            if words[i].text in stop_words:
                del words[i]
                break
    return finalword


def remove_punct(input_word):
    for words, words_detailed in zip(input_word, word_extra):
        for i, (w, word_detailed) in enumerate(zip(words, words_detailed)):
            if word_detailed.upos == 'PUNCT':
                del words_detailed[i]
                words.remove(word_detailed.text)
                break


def lemmatize(final_word_list):
    for words, final in zip(word_extra, final_word_list):
        for i, (w, fin) in enumerate(zip(words, final)):
            if fin in w.text:
                if len(fin) == 1:
                    final[i] = fin
                else:
                    final[i] = w.lemma
    for w in final_word_list:
        print("final_words", w)


# creates a dictionary that tells which subtree is traversed
def label_parse_subtrees(parent_tree):
    tree_traversal_flag = {}
    for sub_tree in parent_tree.subtrees():
        tree_traversal_flag[sub_tree.treeposition()] = 0
    return tree_traversal_flag


def handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    if tree_traversal_flag[sub_tree.treeposition()] == 0 and tree_traversal_flag[sub_tree.parent().treeposition()] == 0:
        tree_traversal_flag[sub_tree.treeposition()] = 1
        modified_parse_tree.insert(i, sub_tree)
        i = i + 1
    return i, modified_parse_tree


def handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    for child_sub_tree in sub_tree.subtrees():
        if child_sub_tree.label() == "NP" or child_sub_tree.label() == 'PRP':
            if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[
                child_sub_tree.parent().treeposition()] == 0:
                tree_traversal_flag[child_sub_tree.treeposition()] = 1
                modified_parse_tree.insert(i, child_sub_tree)
                i = i + 1
    return i, modified_parse_tree


def modify_tree_structure(parent_tree):
    tree_traversal_flag = label_parse_subtrees(parent_tree)
    modified_parse_tree = Tree('ROOT', [])
    i = 0
    for sub_tree in parent_tree.subtrees():
        if sub_tree.label() == "NP":
            i, modified_parse_tree = handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
        if sub_tree.label() == "VP" or sub_tree.label() == "PRP":
            i, modified_parse_tree = handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
    for sub_tree in parent_tree.subtrees():
        for child_sub_tree in sub_tree.subtrees():
            if len(child_sub_tree.leaves()) == 1:
                if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[
                    child_sub_tree.parent().treeposition()] == 0:
                    tree_traversal_flag[child_sub_tree.treeposition()] = 1
                    modified_parse_tree.insert(i, child_sub_tree)
                    i = i + 1
    return modified_parse_tree


def reorder_eng_to_isl(input_string):
    download_required_packages()
    count = 0
    for w in input_string:
        if len(w) == 1:
            count += 1
    if count == len(input_string):
        return input_string
    parser = StanfordParser()
    possible_parse_tree_list = [tree for tree in parser.parse(input_string)]
    print("testing: ", possible_parse_tree_list)
    parse_tree = possible_parse_tree_list[0]
    parent_tree = ParentedTree.convert(parse_tree)
    modified_parse_tree = modify_tree_structure(parent_tree)
    parsed_sent = modified_parse_tree.leaves()
    return parsed_sent


final_words = []
final_words_extra = []


def pre_process(text):
    remove_punct(word)
    final_words.extend(filter_words(word))
    lemmatize(final_words)


# checks for words in words.txt which correspond to the words having sigml files
def final_output(input_word):
    final_string = ""
    valid_words = open("words.txt", 'r').read()
    valid_words = valid_words.split('\n')
    fin_words = []
    for w in input_word:
        w = w.lower()
        if w not in valid_words:
            for letter in w:
                fin_words.append(letter)
        else:
            fin_words.append(w)

    return fin_words


final_sent = []


def convert_to_final():
    for words in final_words:
        final_sent.append(final_output(words))


def take_input(text):
    test_input = text.strip().replace("\n", "").replace("\t", "")
    test_input2 = ""
    if len(test_input) == 1:
        test_input2 = test_input
    else:
        for w in test_input.split("."):
            test_input2 += w.capitalize() + " ."

    some_text = en_nlp(test_input2)
    convert(some_text)


def convert(some_text):
    convert_to_sentence_list(some_text)
    convert_to_word_list(sent_extra)
    print(word)
    pre_process(some_text)
    for i, words in enumerate(word):
        word[i] = reorder_eng_to_isl(words)
    convert_to_final()
    remove_punct(final_sent)
    print_lists()


#def print_lists():
#   # print(word)
#   # print(final_words)
#    print(final_sent)
#    print("\n Final Output:")
#    for texts in final_sent:
#        for text in texts:
#            print(text, end=" ")

def print_lists():
	print("--------------------Word List------------------------");
	pprint.pprint(word_list)
	print("--------------------Final Words------------------------");
	pprint.pprint(final_words);
	print("---------------Final sentence with letters--------------");
	pprint.pprint(final_output_in_sent)

def clear_all():
    sent.clear()
    sent_extra.clear()
    word.clear()
    word_extra.clear()
    final_words.clear()
    final_words_extra.clear()
    final_sent.clear()
    final_words.clear()


# path is defined for txt file where transcript of video is stored
#file_path = "file.txt"
#url = input("Enter video url: ")
#video_to_transcript(url)
#if os.path.isfile(file_path):
#    text_file = open(file_path, "r")
#    data = text_file.read()
#    take_input(data)
#    text_file.close()

# https://www.youtube.com/watch?v=evYLLupqqL4



# dict for sending data to front end in json
final_words_dict = {};

@app.route('/',methods=['GET'])
def index():
	clear_all();
	return render_template('index.html')


@app.route('/',methods=['GET','POST'])
def flask_test():
	clear_all();
	text = request.form.get('text') #gets the text data from input field of front end
	print("text is", text)
	if(text==""):
		return "";
	take_input(text)

	# fills the json 
	for words in final_output_in_sent:
		for i,word in enumerate(words,start=1):
			final_words_dict[i]=word;

	print("---------------Final words dict--------------");
	print(final_words_dict)

	return final_words_dict;


# serve sigml files for animation
@app.route('/static/<path:path>')
def serve_signfiles(path):
	print("here");
	return send_from_directory('static',path)


if __name__=="__main__":
	app.run(debug=True)
