import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    transformed_query = np.dot(W_mult, decoder_hidden_state)
    
    scores = np.dot(encoder_hidden_states, transformed_query)
    
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=0)
    
    attention_vector = np.dot(attention_weights, values)
    
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
       encoder_features = np.dot(encoder_hidden_states, W_add_enc)
    decoder_features = np.dot(decoder_hidden_state, W_add_dec)
    
    tanh_output = np.tanh(encoder_features + decoder_features[:, None, :])
    
    ei = np.dot(tanh_output, v_add)
    attention_weights = np.exp(ei) / np.sum(np.exp(ei), axis=0)
    
    return attention_vector
