# Facebook_word_prediction
Using Kneser-Ney 4-gram, LSTM based RNN and GPT-2 fine tuned modeling to compare results on different Datasets to the GBoard word predictor. 

Federated Learning and AI chips allow circumvention of privacy issues and more personally tuned Language Models(LM). We explore its advantages to test accuracy gains with GBoard next word prediction and user to user LM for Code-Switching. We test several SOA models, including n-grams and various smoothing techniques, especially Kneser-Ney, LSTM RNN's, and OpenAi's GPT-2 on Personal but small Facebook Messenger datasets. We give an overview of the LM field while testing ways to improve GBoard prediction on a personal user level.

## Getting The Data

The thing you will want is the Facebook data itself. we get this by following the instructions here: https://www.zapptales.com/en/download-facebook-messenger-chat-history-how-to/

After you have the data you can open the file in a web browser, and hit CTRL+A and copy and paste this into a text file. Now we can perform some preprocessing on the data and parse out timestamps and usernames, as well as isolate which user it is you want the data of. 

In FB_preprocessing.py:
```
on line 10: f = open(filename, "r"), Put the FB text file you want to preprocess here

on line 12 User = "name", specify the user you want the messages of here

and on line 106 specify the output file. 

Make these changes and run the file, this will give you the facebook data ready for language model training. 
```
### N-Gram Modeling

To model the Facebook data with various N-gram technique refer to FB_Modeling.py
To model the preprocessed data:

```
Line 150: with io.open('filename.txt, .... Put the filename here that you want to be trained

Line 158: N =? Set the order of max order of n-grams you want trained
```

This will train a Kneser-Ney interpolated model. Different models are avaliable at NLTK.LM.Models
```
On line 193: Specify the output file, this will pickle the file and store the model for easy future access
```

### Predicting the next word, and scoring a test file

In FB_nextword.py you can find the next word givena  conditional text, and perform this same function on a set of mesages, with the layout as you will find in /Extrinsic_test_outputs/Steven_ext_messages.txt. 

To perform a next word prediction that will output three choices:

```
Load your model on line 26, load you pickled model

then use the function next_word(model, text)
input loaded_model for the first parameter and your messages you want the model to predict the next word for in "text"
```
You can use the premade models in KN-4-gram_models
To score the accuracy of this model with a test file such as /Extrinsic_test_outputs/Steven_ext_messages.txt. 
```
Use the function document_word_test(document, model, output_file)
load the document, load model_loaded, and the output_file or dir.

if you use print(document_test_word()) it will output the score to the console.
```

## Training the LSTM RNN

This same LSTM based RNN for word level text generation can be found here: https://github.com/campdav/text-rnn-keras

To train this model you can use LSTM_RNN/Train_word_RNN_tf.py:
```
Input you textfile you want raining done on on line 29

on lines 30 and 57 and 95 you set the vocab and model names that you want saved

lines 30: with open('Keras_word_based_NN/words_vocab.pkl', 'rb') 
line 57 make sure the vocab name matches

on line 95 set the output model name:
Line 95: model.save('Keras_word_based_NN/model_name.h5')
```
We could not fit any premade model on github. Training the model can take a while, about 40 epochs is 10 hours worth of training.


### Next word generation with the LSTM, scoring a test doc
As with the n-gram model yuou will likely want to generate text, predict the next word, and score a doc.
These can be done as follows in the file LSTM_RNN/Keras_NN_word_generate.py:

Generating text:
```
Load the model on line 43 and the vocab on line 35

then use print(generate("text")) to generate some text, the lenght of this generation is specified on line 28
```
Generating next word prediction (Three choices):
```
Use the function: print(three_choices("text"))
```
Similar to the N-gram document scoring we score the test doc using the LSTM RNN with:
```
print(document_word_test('test.txt', 'output.txt'))

This will give an output score to the console, for next word prediciton. And output the predictions for later analysis
```
## GPT-2 File training

Information on how to train the GPT-2 to fine-tune it on a document can be found here: https://huggingface.co/transformers/examples.html#language-model-fine-tuning

It will take about an hour to fine tune for 1 epoch, this gave us perplexities of 100-98

We include in this github parts of two finetuned models, they are however missing pytorch_bin.bin due to github file size limitations. we recommend making these models on your own as per the tutorial linked above, use the ones given here as a reference for it being done correctly.

## GPT-2 Next word prediction and text generation
Just as in the other two methods for N-gram and LSTM
The script can be found on GPT2_next_word_test.py:

To generate next word:
```
Load the model for GPT-2 on line 23 and its tokenizor on line 17

then use Next_word("text") to predict the next word
```
Scoring a test document:
```
print(document_word_test("test_doc.txt", 'output_doc.txt'))

as before printing this function will give you a score to the console, adn will output a file for analysis on these prediciotns. GPT-2 only can predict 1 word at a time for right now, not three choices as with the other cases.
```

## Authors

* **Steven Alsheimer** 


## Acknowledgments

*campdav for the LSTM RNN

