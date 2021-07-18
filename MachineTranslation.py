import tensorflow
from tensorflow.keras.layers import LSTM, Input, TimeDistributed, Dense, Embedding, Dropout, Concatenate, Activation, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from DataLoader import *
import numpy as np
import string
import json
import matplotlib.pyplot as plt

class MachineTranslation:
    def __init__(self,):
        self.path_to_data = 'data/ita_eng.txt'
        self.X_train_encoder  = [] # contains the sentences in Italian
        self.X_train_decoder  = [] # contains the sentences in English, where we have added <sos> at the beginning of each sentence
        self.Y_train_decoder  = [] # contains the target English phrases (i.e., without the <sos> token at the beginning of each sentence)
        self.X_valid_encoder  = []
        self.X_valid_decoder  = []
        self.Y_valid_decoder  = []
        self.X_test_encoder   = []
        self.X_test_decoder   = []
        self.Y_test_decoder   = []
        self.italian_vocab_size   = 0
        self.english_vocab_size   = 0
        self.italian_sentences = []
        self.english_sentences = []
        self.model = None
        self.eng_text_tokenizer = None
        self.ita_text_tokenizer = None
        self.encoder_inference = None
        self.decoder_inference = None
        self.max_italian_len = 0
        self.max_english_len = 0

    # Predict new sentence from dataset
    def predict_sentence_target(self, num_sentece_dataset):
        print("The english sentence is: {}".format(self.english_sentences[306001 + num_sentece_dataset]))
        print("The italian sentence is: {}".format(self.italian_sentences[306001 + num_sentece_dataset]))

        print(self.logits_to_sentence(
                                    self.model.predict([self.X_test_encoder[num_sentece_dataset:num_sentece_dataset+1],
                                                        self.X_test_decoder[num_sentece_dataset:num_sentece_dataset+1]],
                                                        batch_size=128)[0],
                                    self.eng_text_tokenizer))


    def logits_to_sentence(self, logits, tokenizer):
        index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
        index_to_words[0] = '<empty>'
        index_to_words[1] = '<end>'
        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

    # Load dataset
    def dataset_load(self):
        raw_data = DataLoader.load(self.path_to_data)
        self.pre_processing_dataset(raw_data)

    # Model load
    def model_load(self, name_model):
        self.model = load_model(name_model)

    # Split dataset in training/validation/testing set
    def split_dataset(self, ):
        X_train_encoder_ = self.X_train_encoder[:200000]
        X_train_decoder_ = self.X_train_decoder[:200000]
        Y_train_decoder_ = self.Y_train_decoder[:200000]

        self.X_valid_encoder = self.X_train_encoder[200001:250000]
        self.X_valid_decoder = self.X_train_decoder[200001:250000]
        self.Y_valid_decoder = self.Y_train_decoder[200001:250000]

        self.X_test_encoder = self.X_train_encoder[250001:]
        self.X_test_decoder = self.X_train_decoder[250001:]
        self.Y_test_decoder = self.Y_train_decoder[250001:]

        self.X_train_encoder = X_train_encoder_
        self.X_train_decoder = X_train_decoder_
        self.Y_train_decoder = Y_train_decoder_


    def train(self, model_name):

        #filepath='model.{epoch:02d}-{val_loss:.2f}.h5'
        mcp_save = ModelCheckpoint(filepath=model_name,
                                   save_best_only=True,
                                   monitor='val_loss',
                                   mode='min',
                                   verbose=1)

        history = self.model.fit([self.X_train_encoder, self.X_train_decoder], self.Y_train_decoder,
                        epochs=25,
                        validation_data=([self.X_valid_encoder, self.X_valid_decoder], self.Y_valid_decoder),
                        verbose=2,
                        batch_size=128,
                        callbacks=mcp_save)

        self.model.save("epoca_finale_"+model_name)

        history_dict = history.history
        # Save it under the form of a json file
        json.dump(history_dict, open('history.json', 'w'))

        metrics_train = self.model.evaluate(x=[self.X_train_encoder, self.X_train_decoder], y=[self.Y_train_decoder], batch_size=128, verbose=0)
        metrics_valid = self.model.evaluate(x=[self.X_valid_encoder, self.X_valid_decoder], y=[self.Y_valid_decoder], batch_size=128, verbose=0)
        return metrics_train, metrics_valid


    # Build encoder/decoder model for training phase
    def build(self, attetion_mode):
        return_seq_lstm_enc = False
        # Encoder
        encoder_input = Input(shape=[None], dtype=tensorflow.int32)
        encoder_embedding = Embedding(input_dim=self.italian_vocab_size+1,
                                      output_dim=256,
                                      mask_zero=True)(encoder_input)

        # If we train the model with attention mechanism, we must return the LSTM output for each step
        if attetion_mode:
            return_seq_lstm_enc = True

        encoder_lstm_output, enc_state_h, enc_state_c = LSTM(256, #dropout=0.3,
                                                             #recurrent_dropout=0.3,
                                                             return_state=True,
                                                             return_sequences=return_seq_lstm_enc)(encoder_embedding)
        encoder_state = [enc_state_h, enc_state_c]  # save the state of the last step of the encoder which will be the initial state of the decoder.
        # Decoder
        decoder_input = Input(shape=[None], dtype=tensorflow.int32)
        decoder_embedding = Embedding(input_dim=self.english_vocab_size+1,
                                      output_dim=256,
                                      mask_zero=True)(decoder_input)
        decoder_lstm_output, dec_state_h, dec_state_c = LSTM(256, #dropout=0.3,
                                                             #recurrent_dropout=0.3,
                                                             return_sequences=True,
                                                             return_state=True)(decoder_embedding, initial_state=encoder_state)
        # If we train the model with attention mechanism,
        # we use the dot method to create the alignment vector definid in Luong Attention
        if attetion_mode:
            dot_prod = Dot(axes=(2, 2))([decoder_lstm_output, encoder_lstm_output])
            attention = Activation('softmax', name='attention')
            attention_vec = attention(dot_prod)
            context = Dot(axes=(2, 1))([attention_vec, encoder_lstm_output])
            conc_out = Concatenate()([context, decoder_lstm_output])

            decoder_lstm_output_dropout = Dropout(0.4)(conc_out)
        else:
            decoder_lstm_output_dropout = Dropout(0.4)(decoder_lstm_output)

        decoder_output = TimeDistributed(Dense(self.english_vocab_size, activation="softmax"))(decoder_lstm_output_dropout)

        model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

        self.model = model

    # Predict new sentence
    def predict_new_sentence(self, sentence, attention_mode):
        sentence = self.pre_processing_sentece(sentence)
        return self.inference(sentence, attention_mode)

    # Input : cleaned and tokenized sentence.
    # We use teacher forcing mechanism for predict one word at time of decoder ooutput
    def inference(self, sentence, attention_mode):
      #tensorflow.keras.utils.plot_model(self.decoder_inference, to_file='lol.png', show_shapes=True)

      # Input the sequence to encoder_model and get the final timestep encoder_states(Hidden and cell state)
      # If attention mechanism is active, the model must return the encoder output also
      if attention_mode:
          encoder_output, state_h, state_c = self.encoder_inference.predict(sentence)
      else:
          state_h, state_c = self.encoder_inference.predict(sentence)

      # Define target word
      target_word = np.zeros((1,1))
      # <start>:1 , <end>:2
      target_word[0,0] = 1

      stop_condition=False
      # Define output sentence string
      sent=''
      step_size=0

      index_to_words = {idx: word for word, idx in self.eng_text_tokenizer.word_index.items()}
      while not stop_condition:

          # We are giving a target_word which represents <start> and encoder_states to the decoder_model
          # If attention mechanism is active, we give as input the encoder output also
          if attention_mode:
              output, state_h, state_c = self.decoder_inference.predict([target_word, encoder_output, [state_h, state_c]])
          else:
              output, state_h, state_c = self.decoder_inference.predict([target_word, state_h, state_c])

          # As the target word length is 1. We will only have one time step
          encoder_state_value = [state_h, state_c]
          # Find the word which the decoder predicted with max_probability
          output = np.argmax(output,-1)
          # The output is a integer sequence, to get back the word. We use our lookup table reverse_dict
          sent = sent+' '+str(index_to_words.get(int(output))) #+eng_index_word[int(output)]
          step_size+=1
          # If the max_length of the sequence is reached or the model predicted 2 (<end>) stop the model
          if step_size>self.max_english_len or output==2:
            stop_condition = True
          # Define next decoder input
          target_word=output.reshape(1,1)

      return sent

    # Pre processing sentece for inference phase
    def pre_processing_sentece(self, sentence):
        sentence = self.clean_sentence(sentence)
        # Tokenize words
        sentence_tokenized = self.ita_text_tokenizer.texts_to_sequences([sentence])
        sentence_tokenized = pad_sequences(sentence_tokenized, self.max_italian_len, padding="post")
        return sentence_tokenized

    # Build (and restore loaded) model that use attention mechanism for inference time
    def build_inference_attention(self,):
        # encoder
        encoder_input = self.model.input[0]
        encoder_lstm_output, encoder_state_h, encoder_state_c = self.model.layers[4].output
        encoder_lstm_states = [encoder_state_h, encoder_state_c]
        encoder_model = Model(encoder_input, # input encoder model
                              [encoder_lstm_output, encoder_state_h, encoder_state_c]) # output encoder model

        # decoder
        decoder_input = self.model.input[1]
        embeded_decoder = self.model.layers[3]
        embeded_decoder = embeded_decoder(decoder_input)
        decoder_state_h = Input(shape=(256), name="input_3")
        decoder_state_c = Input(shape=(256), name="input_4")
        decoder_state_inputs = [decoder_state_h, decoder_state_c]
        decoder_lstm = self.model.layers[5]
        decoder_output_lstm, state_h, state_c = decoder_lstm(embeded_decoder, initial_state=decoder_state_inputs)
        decoder_states = [state_h,state_c]

        # Attention mechanism in decoder
        encoder_out_as_input = Input(shape=(None, 256), name="input_5")
        dot_layer = self.model.layers[6]
        activation_dot_layer = self.model.layers[7]
        attention = dot_layer([decoder_output_lstm, encoder_out_as_input])
        attention = activation_dot_layer(attention)
        dot_layer2 = self.model.layers[8]
        context  = dot_layer2([attention, encoder_out_as_input])
        conc_out = self.model.layers[9]
        conc_out = conc_out([context, decoder_output_lstm])

        # Decoder output
        dropout_out = self.model.layers[10]
        dropout_out = dropout_out(conc_out)
        decoder_dense = self.model.layers[11]
        decoder_outputs = decoder_dense(dropout_out)

        decoder_model = Model([decoder_input, encoder_out_as_input, decoder_state_inputs], # input decoder model
                              [decoder_outputs]+decoder_states) # output decoder model

        return encoder_model, decoder_model

    # Build (and restore loaded) model that NOT use attention mechanism for inference time
    def build_inference_no_attention(self,):
        # encoder
        encoder_input = self.model.input[0]
        _, state_h, state_c = self.model.layers[4].output
        encoder_states = [state_h, state_c]
        encoder_model = Model(encoder_input, # input encoder model
                              encoder_states)# output encoder model

        # decoder
        decoder_input = self.model.input[1]
        embeded_decoder = self.model.layers[3]
        embeded_decoder = embeded_decoder(decoder_input)
        decoder_state_h = Input(shape=(256), name="input_3")
        decoder_state_c = Input(shape=(256), name="input_4")
        decoder_state_inputs = [decoder_state_h,decoder_state_c]
        decoder_lstm = self.model.layers[5]
        decoder_outputs, state_h, state_c = decoder_lstm(embeded_decoder, initial_state=decoder_state_inputs)
        decoder_states = [state_h,state_c]
        # decoder outputs
        dropout_out = self.model.layers[6]
        dropout_out = dropout_out(decoder_outputs)
        decoder_dense = self.model.layers[7]
        decoder_outputs = decoder_dense(dropout_out)

        decoder_model = Model([decoder_input]+decoder_state_inputs, # input decoder model
                              [decoder_outputs]+decoder_states) # output decoder model

        return encoder_model, decoder_model

    # Wrapper method that recall method for to define attention/no attention model
    # for inference phase
    def build_inference(self,attention_mode):
        for layer in self.model.layers:
            print(layer.name)

        if attention_mode:
            encoder_model, decoder_model = self.build_inference_attention()
        else:
            encoder_model, decoder_model = self.build_inference_no_attention()

        self.encoder_inference = encoder_model
        self.decoder_inference = decoder_model

    # Input : raw_data sentences dataset.
    # It processes the dataset sentences  and defines the tokenized dataset
    # for the training phase
    def pre_processing_dataset(self, raw_data):
        pairs = self.parsing_raw_data(raw_data)

        # Split italian and english sentences in two list and remove punctuation from it.
        self.italian_sentences = [self.clean_sentence(pair[1]) for pair in pairs]
        english_sentences = [self.clean_sentence(pair[0]) for pair in pairs]


        X_train_decoder = []
        Y_train_decoder = []
        # Add <start> token at start of every input decoder sentence and
        # add <end> token at end of every target decoder sentence.
        for sentence in english_sentences:
            X_train_decoder.append("<start> " + sentence)
            Y_train_decoder.append(sentence + " <end>")
            self.english_sentences.append("<start> " + sentence + " <end>")

        # Tokenize words
        # Tokenize encoder input
        ita_text_tokenized, self.ita_text_tokenizer = self.tokenize(self.italian_sentences)
        # Tokenize decoder input
        _, self.eng_text_tokenizer = self.tokenize(self.english_sentences)
        eng_in_text_tokenized = self.eng_text_tokenizer.texts_to_sequences(X_train_decoder)
        eng_out_text_tokenized = self.eng_text_tokenizer.texts_to_sequences(Y_train_decoder)

        # Let's add 0 padding to the sentences, to make sure they are all the same length.
        # That is, we must be sure that all Italian sentences have the same length as the
        # longest Italian sentence and that all English sentences have the same length
        # as the longest English sentence
        X_train_encoder = pad_sequences(ita_text_tokenized, padding="post")
        X_train_decoder = pad_sequences(eng_in_text_tokenized, padding="post")
        Y_train_decoder = pad_sequences(eng_out_text_tokenized, padding="post")

        # Let's check the length of the vocabulary
        # Let's add one unit to size for 0 padding
        self.italian_vocab_size = len(self.ita_text_tokenizer.word_index) + 1
        self.english_vocab_size = len(self.eng_text_tokenizer.word_index) + 1

        # get the lenght of max italian/english sentence
        self.max_italian_len = X_train_encoder[0].shape[0] #
        self.max_english_len = X_train_decoder[0].shape[0] #

        # Save dataset in class.
        self.X_train_encoder = X_train_encoder
        self.X_train_decoder = X_train_decoder
        self.Y_train_decoder = Y_train_decoder


    def plot_training_accuracy(self, history, savePlot=False, title = "fig"):
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'validation loss'], loc='upper left')
        if savePlot:
            plt.savefig(title+'_loss.png')
        plt.show()

        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        if savePlot:
            plt.savefig(title+'_acc.png')
        plt.show()

    def plot_training_accuracy_together(self, history_no_att, history_si_att, savePlot=False, title = "fig"):
        plt.plot(history_no_att['loss'])
        plt.plot(history_si_att['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['no attention loss', 'yes attention loss'], loc='upper left')
        if savePlot:
            plt.savefig(title+'_loss.png')
        plt.show()

        plt.plot(history_no_att['val_accuracy'])
        plt.plot(history_si_att['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['no attention val acc', 'si attention val acc'], loc='upper left')
        if savePlot:
            plt.savefig(title+'_acc.png')
        plt.show()


    # Data Parsing : returns a list where each element contains two strings ->
    # the sentence in Italian and the corresponding sentence / target translated into English
    def parsing_raw_data(self, raw_data):
        raw_data = raw_data.split('\n')
        pairs = [sentence.split('\t') for sentence in raw_data]
        #for x in pairs: x.pop()
        return pairs[:300000]

    # Lowercase and remove punctuation in sentences
    def clean_sentence(self, sentence):
        # Add a space ' ' befor the ? word
        sentence = sentence.replace('?', ' ?')
        # Lower case the sentence
        lower_case_sent = sentence.lower()
        # Strip punctuation
        string_punctuation = string.punctuation
        string_punctuation = string_punctuation.replace('?','')
        # Clean the sentence
        clean_sentence = lower_case_sent.translate(str.maketrans('', '', string_punctuation))
        return clean_sentence

    # Tokenize dataset.
    def tokenize(self, sentences):
        # Create tokenizer
        text_tokenizer = Tokenizer(filters='')

        # Fit texts
        text_tokenizer.fit_on_texts(sentences)
        return text_tokenizer.texts_to_sequences(sentences), text_tokenizer
