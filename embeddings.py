import numpy as np
import torch
from torch import nn
import dataset
import os
import typing

# Set up environment variables so that embedding models downloaded from the internet are
# cached locally in the ./embedding_models folder; see:
# https://radimrehurek.com/gensim/downloader.html#:~:text=data%20using%20the-,GENSIM_DATA_DIR,-environment%20variable.%20For
os.environ["GENSIM_DATA_DIR"] = os.path.join(os.path.curdir, "embedding_models")

import gensim
import gensim.downloader


class word_embedder:
    def __init__(self, embedding_model: str = None) -> None:
        self._embedding_model = None
        self._embedding_vector_length = None
        
        if embedding_model:
            self.load_embedding_model(embedding_model)
        
        
    def load_embedding_model(self, embedding_model: str) -> None:
        # Check to see if the specified embedding_model is valid; see:
        # https://radimrehurek.com/gensim/downloader.html#gensim.downloader.info
        available_models = gensim.downloader.info(name_only=True)["models"]
        if not embedding_model in available_models:
            raise ValueError(f'Invalid embedding_model speficied: "{embedding_model}". The following models are available from other research groups: {available_models}')
        
        # Load the specified embedding_model
        self._embedding_model = gensim.downloader.load(embedding_model)
        
        # Get the length of the embedding model's output vectors by looking at the vector for the
        # first word of the model's vector embedding list
        self._embedding_vector_length = self._embedding_model[0].shape[0]
    
    
    def _split_text(text: str) -> typing.Iterable[str]:
        return gensim.utils.tokenize(text, lowercase=True, deacc=False)
    
    
    def _embed_text_vectors(self, text: str, oov_feature = True) -> typing.Iterable[np.ndarray]:
        # Empty input check
        if (not text) or (len(text) == 0):
            return np.empty([0, self._embedding_vector_length + oov_feature])
        
        for word in word_embedder._split_text(text):
            # oov_feature is True, add an extra feature to in-vocab data and return a one-hot oov
            # vector to out-of-vocab data.
            if oov_feature:
                try:
                    # Add a zero out-of-vocab flag if the embedding could be found
                    yield np.concatenate([self._embedding_model[word], np.zeros(1)], axis=0).reshape([1, -1])
                except KeyError:
                    # Add a one out-of-vocab flag if the embedding couldn't be found
                    yield np.concatenate([np.zeros(self._embedding_vector_length), np.ones(1)], axis=0).reshape([1, -1])
            
            # oov_feature is False, simply skip over any out-of-vocab data; no need to concatenate
            # another feature dimension.
            else:
                try:
                    yield self._embedding_model[word].reshape([1, -1])
                except KeyError:
                    continue
    
    
    def _embed_text_tensor(self, text: str, oov_feature = True) -> np.ndarray:
        return np.concatenate(list(self._embed_text_vectors(text, oov_feature)), axis=0)
        
    
    def embed_review(self, review: dataset.review, oov_feature = True, title_body_feature = True) -> np.ndarray:
        # Embed the title and the body
        title_embedding = self._embed_text_tensor(review.title, oov_feature)
        body_embedding = self._embed_text_tensor(review.body, oov_feature)
        
        # Add an extra feature to denote title vs. body, if desired
        if title_body_feature:
            # Add zeros as a feature for the title
            title_embedding = np.concatenate([title_embedding, np.zeros(title_embedding.shape[0]).reshape([-1, 1])], axis=1)
            # And ones as a feature for the body
            body_embedding = np.concatenate([body_embedding, np.ones(body_embedding.shape[0]).reshape([-1, 1])], axis=1)
        
        # Combine the review title and body together
        review_embedding = np.concatenate([title_embedding, body_embedding], axis=0)
        return review_embedding