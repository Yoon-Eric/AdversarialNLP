import numpy as np
import math
import textattack
import transformers
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers.pytorch_model_wrapper import PyTorchModelWrapper
from textattack.models.helpers import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer
import random
from collections import Counter
from nltk.stem import WordNetLemmatizer

# for random masking
class MyWrapperRandomMask(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer):
        assert isinstance(
            model, (transformers.PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        # self.tfidf_vocab = tfidf_vocab

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """

        """
        Option 1 Make a custom preprocessing function for text_input_list
        """
#############################       
        percent_to_mask = 30
        for i in range(len(text_input_list)):
            words = text_input_list[i].split()
            num_words = len(words)
            num_to_mask = int(num_words * percent_to_mask / 100)
            indices_to_mask = random.sample(range(num_words), num_to_mask)
            text_input_list[i] = (" ".join(["<mask>" if i in indices_to_mask else word for i, word in enumerate(words)]))
#############################
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

       
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
    

# for TF-IDF based Masking
class MyWrapperImportanceBasedMask(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, importance_dict):
        assert isinstance(
            model, (transformers.PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.importance_dict = importance_dict

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """

        """
        Option 1 Put preprocessing function for text_input_list before tokenizing
        """
#############################       
        percent_to_mask = 30
        for i in range(len(text_input_list)):
            # truncate n% of words using tfidf
            words = text_input_list[i].split()

            # Sort the words in descending order of importance
            sorted_words = sorted(words, key=lambda w: self.importance_dict.get(w, 0), reverse=True)

            # Replace the n most important words with "<mask>"
            n_words_to_mask = int(len(sorted_words) * (int(percent_to_mask) / 100))
            masked_words = ["<mask>" if w in sorted_words[:n_words_to_mask] else w for w in words]

            # Join the remaining words back into a sentence
            remaining_sentence = " ".join(masked_words)

            text_input_list[i] = remaining_sentence
#############################

        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
    
# Function for building TF-IDF vocab
def tfidf_vocab(sentences):
    # lemmatizer = WordNetLemmatizer()

    # Step 1: Calculate the term frequency (TF) for each word in each sentence
    tf_dict = {}
    sentence_count = len(sentences)
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        word_count = len(words)
        word_freq = Counter(words)
        for word in word_freq:
            tf = word_freq[word] / word_count
            if word in tf_dict:
                if i in tf_dict[word]:
                    tf_dict[word][i] += tf
                else:
                    tf_dict[word][i] = tf
            else:
                tf_dict[word] = {i: tf}
    
    # Step 2: Calculate the inverse document frequency (IDF) for each word
    idf_dict = {}
    for word in tf_dict:
        idf_dict[word] = math.log(sentence_count / len(tf_dict[word]))
    
    # Step 3: Calculate the TF-IDF score for each word in each sentence
    tfidf_dict = {}
    for word in tf_dict:
        for sentence_index in tf_dict[word]:
            tf = tf_dict[word][sentence_index]
            idf = idf_dict[word]
            tfidf = tf * idf
            if sentence_index in tfidf_dict:
                tfidf_dict[sentence_index][word] = tfidf
            else:
                tfidf_dict[sentence_index] = {word: tfidf}
    
    # Step 4: Sum up TF-IDF scores for each unique word
    tfidf_corpus = {}
    for i in range(len(tfidf_dict)):
        for word in tfidf_dict[i]:
            if word in tfidf_corpus:
                tfidf_corpus[word] += tfidf_dict[i][word]
            else:
                tfidf_corpus[word] = tfidf_dict[i][word]
 
    return tfidf_corpus

# function for removing stop words from the vocab
def remove_stop_words(word_importance_dict):
    # add more 
    stop_words = ["the", "a", "an", "and", "in", "on", "at", "to", "of", "are", "is", "am", "were"]
    # Create a new dictionary that excludes the stop words
    return {k: v for k, v in word_importance_dict.items() if k not in stop_words}



# Random masking on vectorized tokens
class MyWrapperRandomTokenMask(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer):
        assert isinstance(
            model, (transformers.PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        # self.tfidf_vocab = tfidf_vocab

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """

        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        """
        Option 2 Make a function that masks vectorized tokens
        """
#####################
        # take in the inputs_dict and randomly attention_mask = 0
        # Mask random tokens        
        # don't touch the first and last tokens
        percent_zero=40
        for i in range(len(inputs_dict['attention_mask'])):
            length = torch.count_nonzero(inputs_dict['attention_mask'][i])-2
            num_zeros = int(length * percent_zero/100)
            indices = np.arange(length)
            random_indices = np.random.choice(np.arange(length), size=num_zeros, replace=False)
            for j in random_indices:
                inputs_dict['attention_mask'][i][j+1] = 0
        
######################

        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
###
        percent_zero=40
        for i in range(len(inputs_dict['attention_mask'])):
            length = torch.count_nonzero(inputs_dict['attention_mask'][i])-2
            num_zeros = int(length * percent_zero/100)
            indices = np.arange(length)
            random_indices = np.random.choice(np.arange(length), size=num_zeros, replace=False)
            for j in random_indices:
                inputs_dict['attention_mask'][i][j+1] = 0
###
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
    


# TF-IDF masking on vectorized tokens
class MyWrapperTfidfTokenMask(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, tfidf):
        assert isinstance(
            model, (transformers.PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.tfidf = tfidf
        # self.tfidf_vocab = tfidf_vocab

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """

        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        """
        Option 2 Make a function that masks vectorized tokens
        """
#####################
        # take in the inputs_dict and attention_mask = 0
        # Mask tokens with the highest tfidf value      
        # don't touch the first and last tokens
        percent_zero=40
        for i in range(len(inputs_dict['input_ids'])):
            length = torch.count_nonzero(inputs_dict['attention_mask'][i])-2
            num_zeros = int(length * percent_zero/100)
            
            input_id = inputs_dict['input_ids'][i][1:(length+1)].tolist()

            sorted_list = sorted(input_id, key=lambda x: self.tfidf.get(x, 0), reverse=True)
            top_four = sorted_list[:num_zeros]
            indices = [input_id.index(x) for x in top_four]
            for j in indices:
                inputs_dict['attention_mask'][i][j+1] = 0
######################

        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        
        percent_zero=40
        for i in range(len(inputs_dict['input_ids'])):
            length = torch.count_nonzero(inputs_dict['attention_mask'][i])-2
            num_zeros = int(length * percent_zero/100)
            
            input_id = inputs_dict['input_ids'][i][1:(length+1)].tolist()

            sorted_list = sorted(input_id, key=lambda x: self.tfidf.get(x, 0), reverse=True)
            top_four = sorted_list[:num_zeros]
            indices = [input_id.index(x) for x in top_four]
            for j in indices:
                inputs_dict['attention_mask'][i][j+1] = 0

        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
    
# Function for building TF-IDF vocab for vectorized tokens
def tfidf_vocab(input):
    # Step 1: Calculate the term frequency (TF) for each number in each list
    tf_dict = {}
    list_count = len(input)
    for i, sub_input in enumerate(input):
        for num in sub_input:
            # Increment the TF count for this number in this list
            if num in tf_dict:
                if i in tf_dict[num]:
                    tf_dict[num][i] += 1
                else:
                    tf_dict[num][i] = 1
            else:
                tf_dict[num] = {i: 1}
    
    # Step 2: Calculate the inverse document frequency (IDF) for each number
    idf_dict = {}
    for num in tf_dict:
        idf_dict[num] = math.log(list_count / len(tf_dict[num]))
    
    # Step 3: Calculate the TF-IDF score for each number in each list
    tfidf_dict = {}
    for num in tf_dict:
        for list_index in tf_dict[num]:
            tf = tf_dict[num][list_index]
            idf = idf_dict[num]
            tfidf = tf * idf
            if list_index in tfidf_dict:
                tfidf_dict[list_index][num] = tfidf
            else:
                tfidf_dict[list_index] = {num: tfidf}
    
    # Step 4: sum up tfidf values for each unique tokens
    tfidf_corpus = {}
    for i in range(len(tfidf_dict)):
        for j in tfidf_dict[i]:
            if j in tfidf_corpus:
                tfidf_corpus[j] += tfidf_dict[i][j]
            else:
                tfidf_corpus[j] = tfidf_dict[i][j]
 
    return tfidf_corpus