import streamlit as st
from SP_RNN import SP_RNN

# streamlit run Assignment-2/web_demo.py


st.title('Chunk Tagger App')

weights_file = "Assignment-2\\model_weights.npz"


user_input = st.text_input('Enter POS Tag:', '')

if st.button('Submit'):

    pos_tags = [int(char) for char in user_input]
    no_of_words = len(pos_tags)
    pos_tags.insert(0, 0)
    x = []
    for i in range(no_of_words):
        xi = [0 for _ in range(9)]
        xi[pos_tags[i]] = 1
        xi[pos_tags[i + 1] + 4] = 1
        x.append(xi)

    chunk_tagger = SP_RNN(no_of_inputs=9)
    chunk_tagger.load_weights(weights_file)
    y = chunk_tagger.predict_output(x)

    st.write(f'Corresponding Chunk Tag: {y}')