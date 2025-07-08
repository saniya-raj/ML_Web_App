# In CMD write the following command
# !pip install streamlit

import streamlit as st

st.title('Streamlit App Demo')

st.header('Header')
st.subheader('Sub Header')

st.text('Welcome to the Streamlit App')

st.markdown("""
# H1 Heading
## H2 Heading
### H3 heading
:moon:
:star:
:smile:
""")


# Create a link
st.markdown("""
[Redirect to google]
(https://www.google.com)
""",True)

# 1) To wun Streamlit web app in Browser, open the terminal and write the following
# streamlit run app.py
# 2) To stop the running server, in the terminal
# Press Ctrl + C 