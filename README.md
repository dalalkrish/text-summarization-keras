# Text Summarization With Keras

This text summarizer is built on BBC news articles from categories
such as Business, Politics and Sports. 

Pickle files of the articles along with their respective heading is provided
here.

Pre trained 6B GloVe vectors from Standfor University are used to speed up the 
training of the network.

The network uses Encoder-Decoder Architecture with 'Attention' layer to focus
on important words used for summarization.
