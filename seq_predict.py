######################
# Import libraries
######################
from numpy import source
import pandas as pd
import PIL
import streamlit as st       

import altair as alt

######################
# Page Title
######################

image = PIL.Image.open('genome-intro.PNG')

st.image(image, use_column_width=False)

st.write("""
#  Streamlit app for counting the frequency of A, T, G, and C in a given DNA sequence!
***""")

######################
# Input Text Box
######################

# st.sidebar.header('Enter DNA sequence')
st.header('Enter DNA sequence')

sequence_input = ">DNA Query 2\nGAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\nATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\nTGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT"

# sequence = st.sidebar.text_area("Sequence input", sequence_input, height=250)
sequence = st.text_area("Sequence input", sequence_input, height=250)
sequence = sequence.splitlines()
sequence = sequence[1:]  # Skips the sequence name (first line)
sequence = ''.join(sequence)  # Concatenates list to string

st.write("""
***
""")

## Prints the input DNA sequence
st.header('Entered Sequence')
sequence

## DNA nucleotide count
st.header('Count of A, T, G, and C in a given DNA sequence')

### 1. Print dictionary
st.subheader('1. Number of each base in a enterd DNA sequence')


def DNA_nucleotide_count(seq):
    d = dict([
        ('A', seq.count('A')),
        ('T', seq.count('T')),
        ('G', seq.count('G')),
        ('C', seq.count('C'))
    ])
    return d


X = DNA_nucleotide_count(sequence)

# X_label = list(X)
# X_values = list(X.values())

X

### 2. Print text
st.subheader('2.Description of Bases with Number')
st.write('There are  ' + str(X['A']) + ' adenine (A)')
st.write('There are  ' + str(X['T']) + ' thymine (T)')
st.write('There are  ' + str(X['G']) + ' guanine (G)')
st.write('There are  ' + str(X['C']) + ' cytosine (C)')

### 3. Display DataFrame
st.subheader('3. Tabular Representation of Data')
df = pd.DataFrame.from_dict(X, orient='index')
df = df.rename({0: 'count'}, axis='columns')
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'nucleotide'})
st.write(df)

### 4. Display Bar Chart using Altair
st.subheader('4.Visual Representation of the Data')
p = alt.Chart(df).mark_bar().encode(
    x='nucleotide',
    y='count'
)

p = p.properties(
    width=alt.Step(80)  # controls width of bar.
)
st.write(p)
