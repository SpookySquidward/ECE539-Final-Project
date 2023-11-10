import numpy as np
import torch
import dataset
import os
import typing
from tqdm import tqdm
from random import shuffle

# Set up environment variables so that embedding models downloaded from the internet are
# cached locally in the ./embedding_models folder; see:
# https://radimrehurek.com/gensim/downloader.html#:~:text=data%20using%20the-,GENSIM_DATA_DIR,-environment%20variable.%20For
os.environ["GENSIM_DATA_DIR"] = os.path.join(os.path.curdir, "embedding_models")

import gensim
import gensim.downloader


class review_embedder:
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
        return gensim.utils.tokenize(text, lowercase=True, deacc=True)
    
    
    def _embed_text_vectors(self, text: str, oov_feature = True) -> typing.Iterable[np.ndarray]:
        # Empty input check
        if (not text) or (len(text) == 0):
            return np.empty([0, self._embedding_vector_length + oov_feature])
        
        for word in review_embedder._split_text(text):
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
        embedded_text = list(self._embed_text_vectors(text, oov_feature))
        if len(embedded_text) > 0:
            return  np.concatenate(embedded_text, axis=0)
        else:
            # Blank or non-alpha text, return an empty tensor
            return np.empty([0, self._embedding_vector_length + oov_feature])
        
    
    def embed_review_features(self, review: dataset.review, oov_feature = True, title_body_feature = True, dtype=torch.float32) -> torch.Tensor:
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
        review_feature_embedding = np.concatenate([title_embedding, body_embedding], axis=0)
        
        # Cast to tensor
        review_feature_embedding = torch.from_numpy(review_feature_embedding).type(dtype)
        
        # Return the embedded review features
        return review_feature_embedding
    

    def embed_dataset_features_and_labels(self, reviews: typing.Iterable[dataset.review],
                                          review_label_mapping: dict[str, torch.Tensor],
                                          oov_feature = True, title_body_feature = True) -> typing.Tuple[list[torch.tensor], list[torch.tensor]]:
        
        # Verify the mappings are all the same shape
        mapping_keys = review_label_mapping.keys()
        try:
            # This will throw a RuntimeError when the review label mappings don't have identical shapes
            torch.stack(list(review_label_mapping[mapping_key] for mapping_key in mapping_keys))
        except RuntimeError as E:
            raise ValueError(f"Torch RuntimeError {E}; make sure that all tensors in the review_label_mapping dict have the same shape!")
        
        
        embedded_features = []
        one_hot_labels = []
        
        with tqdm(reviews, "Embedding features", position=0, leave=True) as treviews:
            for review in treviews:
                features = self.embed_review_features(review, oov_feature, title_body_feature)
                embedded_features.append(features)
                
                # Label mapping needs to be reshaped to have an extra dimension to emulate a batch size of 1
                one_hot_label = review_label_mapping[review.label].reshape([1, -1])
                one_hot_labels.append(one_hot_label)
        
        return embedded_features, one_hot_labels
            

class embedded_review_random_sampler(typing.Iterable):
    def __init__(self, embedded_features: list[torch.tensor], one_hot_labels: list[torch.tensor]) -> None:
        # Check for valid inputs
        if len(embedded_features) != len(one_hot_labels):
            raise ValueError(f"Mismatched feature and label list lengths of {len(embedded_features)} and {len(one_hot_labels)}, respectively")
        
        # Store inputs
        self._embedded_features = embedded_features
        self._one_hot_labels = one_hot_labels
        
        # Indexes to sample from
        self._sample_indexes = np.arange(len(self), dtype=np.int32)
    
    
    def __len__(self) -> int:
        return len(self._embedded_features)
    
    
    def __iter__(self) -> typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]:
        # Prepare to iterate through the data
        np.random.shuffle(self._sample_indexes)
        self._sample_read_location = 0
        return self


    def __next__(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # End iteration if all data hsa been read
        if self._sample_read_location >= len(self):
            raise StopIteration
        
        # Get one sample
        read_location = self._sample_indexes[self._sample_read_location]
        sample_features = self._embedded_features[read_location]
        sample_label = self._one_hot_labels[read_location]
        
        # Advance the read location
        self._sample_read_location += 1
        
        # Return the requested sample
        return sample_features, sample_label
        

# TODO finish me!
# class review_embedder_sampler(typing.Iterable):
#     def __init__(self, reviews: typing.List[dataset.review], embedder: review_embedder, review_label_mapping: dict[str, torch.Tensor], oov_feature = True, title_body_feature = True, chunk_size: int = 500):
#         # Store parameters
#         self._reviews = []
#         self._embedder = embedder
#         self._chunk_size = chunk_size
    
    
#     def __len__(self) -> int:
#         return len(self._reviews)
    
    
#     def __iter__(self) -> typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]:
#         # Prepare to iterate through the data
#         shuffle(self._reviews)
#         self._current_chunk = []
#         self._sample_read_location = 0
#         self._load_next_chunk()
#         return self
    
    
#     def _load_next_chunk(self) -> bool:
#         if self._sample_read_location >= len(self):
#             # Out of data to read, exit early
#             return False
        
#         self._current_chunk = self._reviews[self._sample_read_location : self._sample_read_location + self._chunk_size]


# TODO this sampler could make repeated calls from an embedded_review_random_sampler to construct
# torch.nn.utils.rnn.PackedSequence feature objects; see:
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
# https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence
# Note: the existing implementation is not functional!

# class embedded_review_random_batch_sampler(typing.Iterable):
#     def __init__(self, data: list[typing.Tuple[torch.Tensor, typing.Any]], batch_size: int) -> None:
#         self.data = data
#         self.batch_size = batch_size
    
    
#     def __iter__(self) -> typing.Iterator[list[typing.Tuple[torch.Tensor, typing.Any]]]:
#         # Shuffle the data and prepare to read a new epoch
#         random.shuffle(self.data)
#         self.read_location_current_batch = 0
#         return self
    
    
#     def __next__(self) -> typing.Tuple[torch.nn.utils.rnn.PackedSequence[torch.Tensor], torch.Tensor]:
#         # End iteration if all data has been read
#         if self.read_location_current_batch >= len(self.data):
#             raise StopIteration
        
#         # Read a batch of data
#         batch_data = self.data[self.read_location_current_batch : self.read_location_current_batch + self.batch_size]
#         batch_data.sort(key=(lambda x : x[0].shape[0]), reverse=True)
        
#         # Update the read position
#         self.read_location_current_batch += self.batch_size
        
#         # Split the data into feature and label lists
#         batch_data_features = list(data[0] for data in batch_data)
#         batch_data_labels = ...
        
#         return batch_data