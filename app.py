import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

@st.cache_resource
def train_model():
    np.random.seed(50)
    size = np.random.normal(1400, 50, 100)
    price = size * 50 + np.random.normal(0, 50, 100)
    df = pd.DataFrame({'size': size, 'price': price})
    X = df[['size']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, df

model, df = train_model()

def main():
    st.title("Simple Linear Regression House Prediction App")
    size = st.number_input('House size', min_value=50, max_value=2000, value=1500)

    if st.button('Predict price'):
        prediction = model.predict([[size]])
        st.success(f'Estimated price: ${prediction[0]:,.2f}')
        fig = px.scatter(df, x='size', y='price', title='Size vs House Price')
        fig.add_scatter(x=[size], y=[prediction[0]], mode='markers',
                        marker=dict(color='red', size=15), name='Prediction')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()


 