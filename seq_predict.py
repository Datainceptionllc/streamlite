######################
# Import libraries
######################
import pandas as pd
import PIL
import streamlit as st       

import altair as alt

######################
# Page Title
######################

image = PIL.Image.open('Slide1.JPG')

st.image(image, use_column_width=False)

st.write("""
# DNA Nucleotide Count Web App
This app counts the nucleotide composition of query DNA!
***""")

######################
# Input Text Box
######################

# st.sidebar.header('Enter DNA sequence')
st.header('Enter DNA sequence')

sequence_input = input(">DNA Query 2\nGAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\nATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\nTGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT")

# sequence = st.sidebar.text_area("Sequence input", sequence_input, height=250)
sequence = st.text_area("Sequence input", sequence_input, height=250)
sequence = sequence.splitlines()
sequence = sequence[1:]  # Skips the sequence name (first line)
sequence = ''.join(sequence)  # Concatenates list to string

st.write("""
***
""")

## Prints the input DNA sequence
st.header('INPUT (DNA Query)')
sequence

## DNA nucleotide count
st.header('OUTPUT (DNA Nucleotide Count)')

### 1. Print dictionary
st.subheader('1. Print dictionary')


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
st.subheader('2. Print text')
st.write('There are  ' + str(X['A']) + ' adenine (A)')
st.write('There are  ' + str(X['T']) + ' thymine (T)')
st.write('There are  ' + str(X['G']) + ' guanine (G)')
st.write('There are  ' + str(X['C']) + ' cytosine (C)')

### 3. Display DataFrame
st.subheader('3. Display DataFrame')
df = pd.DataFrame.from_dict(X, orient='index')
df = df.rename({0: 'count'}, axis='columns')
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'nucleotide'})
st.write(df)

### 4. Display Bar Chart using Altair
st.subheader('4. Display Bar chart')
p = alt.Chart(df).mark_bar().encode(
    x='nucleotide',
    y='count'
)
alt.Chart(source).mark_bar().encode(
    x='nucleotide',
    y='count',
    # The highlight will be set on the result of a conditional statement
    color=alt.condition(
        alt.datum.nucleotide== A,  # If the year is 1810 this test returns True,
        alt.value('orange'),     # which sets the bar orange.
        alt.value('steelblue')   # And if it's not true it sets the bar steelblue.
    )
)
p = p.properties(
    width=alt.Step(80)  # controls width of bar.
)
st.write(p)
