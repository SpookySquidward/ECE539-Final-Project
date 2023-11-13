import numpy as np
import torch
import dataset
import os
import typing
from tqdm import tqdm
from random import shuffle
from math import ceil

# Set up environment variables so that embedding models downloaded from the internet are
# cached locally in the ./embedding_models folder; see:
# https://radimrehurek.com/gensim/downloader.html#:~:text=data%20using%20the-,GENSIM_DATA_DIR,-environment%20variable.%20For
os.environ["GENSIM_DATA_DIR"] = os.path.join(os.path.curdir, "embedding_models")

import gensim
import gensim.downloader


class review_embedder:
    def __init__(self, embedding_model: str = None) -> None:
        self._embedding_model = embedding_model
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
                                          oov_feature = True, title_body_feature = True) -> list[typing.Tuple[torch.tensor, torch.tensor]]:
        
        # Verify the mappings are all the same shape
        mapping_keys = review_label_mapping.keys()
        try:
            # This will throw a RuntimeError when the review label mappings don't have identical shapes
            torch.stack(list(review_label_mapping[mapping_key] for mapping_key in mapping_keys))
        except RuntimeError as E:
            raise ValueError(f"Torch RuntimeError {E}; make sure that all tensors in the review_label_mapping dict have the same shape!")
        
        
        embedded_reviews = []
        
        with tqdm(reviews, "Embedding features", position=1, unit="reviews", leave=False) as treviews:
            for review in treviews:
                features = self.embed_review_features(review, oov_feature, title_body_feature)
                embedded_features.append(features)
                
                # Label mapping needs to be reshaped to have an extra dimension to emulate a batch size of 1
                one_hot_label = review_label_mapping[review.label].reshape([1, -1])
                
                embedded_reviews.append((features, one_hot_label))
        
        return embedded_reviews
            

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
        

class review_embedder_sampler(typing.Iterable[typing.Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, reviews: typing.List[dataset.review], embedder: review_embedder, review_label_mapping: dict[str, torch.Tensor], oov_feature = True, title_body_feature = True, chunk_size: int = 5000):
        # Store parameters
        self._reviews = reviews
        self._embedder = embedder
        self._review_label_mapping = review_label_mapping
        self._oov_feature = oov_feature
        self._title_body_feature = title_body_feature
        self._chunk_size = chunk_size
    
    
    def __len__(self) -> int:
        return len(self._reviews)
    
    
    def __iter__(self) -> typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]:
        # Shuffle the reviews before iterating over them
        shuffle(self._reviews)
        
        # Set read location for reviews
        self._review_read_location = 0
        
        # Initialize the first chunk of data
        self._load_next_chunk()
        
        return self
    
    
    def _load_next_chunk(self) -> bool:
        # Exit early if no more data exists to read
        if self._review_read_location >= len(self):
            return False
        
        # Fetch the next chunk of data
        next_chunk_reviews = self._reviews[self._review_read_location : self._review_read_location + self._chunk_size]
        next_chunk_embedded_reviews = self._embedder.embed_dataset_features_and_labels(next_chunk_reviews, self._review_label_mapping, self._oov_feature, self._title_body_feature)
        self._current_chunk_embedded_reviews = next_chunk_embedded_reviews
        self._chunk_read_location = 0
        
        # Update read location for next load call
        self._review_read_location += self._chunk_size
        
        # Chunk was successfully loaded, return True
        return True
        
    
    def __next__(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # Check to see if another chunk of data needs to be loaded
        if self._chunk_read_location >= len(self._current_chunk_embedded_reviews):
            chunk_load_success = self._load_next_chunk()
            # Exit if no new reviews were found to embed
            if not chunk_load_success:
                raise StopIteration
        
        # Get the next sample
        sample = self._current_chunk_embedded_reviews[self._chunk_read_location]
        
        # Update the chunk read position
        self._chunk_read_location += 1
        
        # Return the requested sample
        return sample


class batched_review_embedder_sampler(typing.Iterable[typing.Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor]]):
    def __init__(self, reviews: typing.List[dataset.review], embedder: review_embedder, review_label_mapping: dict[str, torch.Tensor], batch_size: int = 100, oov_feature = True, title_body_feature = True, chunk_size: int = 5000):
        # Initialize the actual embedder
        self._sampler = review_embedder_sampler(reviews, embedder, review_label_mapping, oov_feature, title_body_feature, chunk_size)
        
        # Store parameter
        self._batch_size = batch_size


    def __len__(self) -> int:
        return ceil(len(self._sampler) / self._batch_size)
    
    
    def __iter__(self) -> typing.Iterator[typing.Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor]]:
        # Reset the sampler
        iter(self._sampler)
        
        return self
    
    
    def __next__(self) -> typing.Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor]:
        # Fetch batch_size samples
        batch_samples = []
        for sample_i in range(self._batch_size):
            try:
                # Fetch one sample
                batch_samples.append(next(self._sampler))
                
            except StopIteration:
                # Out of samples! Finish the current batch if there are any samples in it, or exit if not
                if len(batch_samples) > 0:
                    break
                else:
                    raise StopIteration
        
        # In order to pack the samples into a PackedSequence, they first need to be sorted by length; see:
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence
        batch_samples.sort(key = (lambda sample: sample[0].shape[0]), reverse=True)
        
        # Unzip the samples into feature and label lists
        batch_features = list(sample[0] for sample in batch_samples)
        batch_labels = list(sample[1] for sample in batch_samples)
        
        # Pack the features
        batch_features_packed = torch.nn.utils.rnn.pack_sequence(batch_features)
        
        # And stack the labels
        batch_labels_stacked = torch.concatenate(batch_labels, dim=0)
        
        return batch_features_packed, batch_labels_stacked
        
