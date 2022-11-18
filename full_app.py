import streamlit as st
def intro():
    import streamlit as st



    st.write("# Welcome to HCDR App :-) ")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Credit Risk Default prediction is one of the most important and toughest
        jobs for any finance firms as they need to consider many factors while disbursing a loan.
        In this app I have tried to build a model which will predict whether a given user will repay the loan or
        will be turn out as defaulter.
        I have used both XgBoost and LightGBM while modelling but surprisingly guess what XgBoost turned out to be winner in terms of KPI .

        **👈 Select a demo from the sidebar** to see some examples
        of what My App can do!
        ### Want to learn more?
        - Check out [github.io](https://github.com/saikirankrish1922/Home_Credit_Default_Risk)
        ## Want to connect with me ?
        - Check out my profile at [linkedIn](https://www.linkedin.com/in/sai-kiran-bandaru/)
        - Shoot your questions to me at my mail : saikirankrish1922@gmail.com

    """
    )

def page_1():
    import streamlit as st
    import time
    import numpy as np
    import pandas as pd
    import plotly.figure_factory as ff
    #import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    #import seaborn as sns

    st.markdown(f"# {list(page_names_to_funcs.keys())[0]}")
    st.markdown("# EDA on Train dataset  Demo")
    st.sidebar.header("Plotting Demo")
    st.write(
        """This demo in the page illustrates a heuristic view of the data provided to us
        I have build some charts to show the same ."""
    )



    train_df = pd.read_csv("train.csv")



    st.bar_chart(train_df['TARGET'].value_counts())

    # Pie chart of user's
    st.subheader('Pie chart of users')
    a , b = train_df['TARGET'].value_counts()
    sizes = [a,b]
    labels = 'safe_client','risky_client'
    explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)
    st.text('Here we can conclude that given dataset is highly imbalance as there will be  very few defaulters/risky customers for any fintech')

    # Finding who has accompained the user while applying for loan
    st.subheader('Finding who has accompained customer while applying for loan')
    st.bar_chart(train_df['NAME_TYPE_SUITE'].value_counts())
    st.text('We can observe that user was unaccompained in most of the cases when a customer came to apply for loan')

    st.subheader('Checking how many has own car')
    st.bar_chart(train_df['FLAG_OWN_CAR'].value_counts())
    (a,b) = train_df['FLAG_OWN_CAR'].value_counts()


    st.write('There are  {} percent of people not having cars.   '.format(str(round(a/(a+b),3))))

    st.write('There are {} percent of people not having cars.   '.format(str(round(b/(a+b),3))))
    #st.text('It looks like 70 % of the users doesnt have own car')

    st.subheader('Chart of users income type')
    #st.bar_chart(train_df['NAME_INCOME_TYPE'].value_counts())

    arr = train_df['NAME_INCOME_TYPE'].value_counts()
    lable = train_df['NAME_INCOME_TYPE'].unique()
    labels = ['Working', 'State servant', 'Commercial associate', 'Pensioner',
           'Unemployed']
    fig, ax = plt.subplots()
    ax.pie(arr,labels=labels)

    st.pyplot(fig)

    st.subheader('Chart of users income type 2')

    xx = train_df['NAME_INCOME_TYPE'].unique()
    yy = train_df['NAME_INCOME_TYPE'].value_counts()

    #import plotly.graph_objects as go


    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=xx, values=yy, hole=.3)])
    st.plotly_chart(fig)



    st.subheader('Chart of users loans types')


    st.bar_chart(train_df['NAME_CONTRACT_TYPE'].value_counts())


    st.subheader('Chart of users ORGANIZATION_TYPE')

    xx = train_df['ORGANIZATION_TYPE'].unique()
    yy = train_df['ORGANIZATION_TYPE'].value_counts()

    fig2 = go.Figure( go.Bar(x=xx, y=yy ,text=yy,
                textposition='auto',) )
    st.plotly_chart(fig2)


def page_2():
    import streamlit as st
    import time
    import numpy as np
    import pandas as pd
    import plotly.figure_factory as ff
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    #import seaborn as sns

    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")

    st.markdown("# EDA on Bureau dataset  Demo")
    st.sidebar.header("Plotting Demo")
    st.write(
        """This demo in the page illustrates a heuristic view of the data provided to us in Bureau file
        I have build some charts to show the same ."""
    )


    bureau = pd.read_csv("bureau.csv")

    #loading bureau balance

    bureau_balance = pd.read_csv("bureau_balance.csv")

    st.subheader('Chart of users with different types of credit')

    xx = bureau['CREDIT_TYPE'].unique()
    #print(xx)
    yy = bureau['CREDIT_TYPE'].value_counts()

    fig = go.Figure(data=[go.Pie(labels=xx, values=yy)])
    st.plotly_chart(fig)

    st.subheader('How many users have an active credit account with bureau  ')


    st.bar_chart(bureau['CREDIT_ACTIVE'].value_counts())

    st.text('Here we can  observe 50% users has closed their loan and 30% has acitve ' )
    #CNT_CREDIT_PROLONG
    st.subheader('How many users have an active credit account with bureau  ')


    st.bar_chart(bureau['CNT_CREDIT_PROLONG'].value_counts())

    st.text('Here we can see most of the users have 0' )

    st.header("EDA on Bureau Balance data set ")
    st.subheader('How many users have an active credit account with bureau balance   ')

    st.bar_chart(bureau_balance['STATUS'].value_counts())

    st.text ('Status of Credit Bureau loan during the month (active, closed, DPD0-30,Ö')
    st.text ('[C means closed, X means status unknown, 0 means no DPD, 1 means maximal did during month between 1-30, 2 means DPD 31-60,Ö 5 means DPD 120+ or sold or written off ]' )


def page_3():
    import streamlit as st
    import time
    import numpy as np
    import pandas as pd
    import plotly.figure_factory as ff
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    #import seaborn as sns

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")

    st.markdown("# EDA on other dataset  Demo")
    st.sidebar.header("Plotting Demo")
    st.write(
        """This demo in the page illustrates a heuristic view of the data provided to us in Bureau file
        I have build some charts to show the same ."""
    )



    #loading credit_card_balance.csv

    credit_card_balance = pd.read_csv("credit_card_balance.csv")





    #loading previous_application.csv

    previous_app = pd.read_csv("previous_app.csv")

    st.subheader('Info on how the customer came to apply loan previous times ')
    xx = previous_app['NAME_TYPE_SUITE'].unique()
    #print(xx)
    yy = previous_app['NAME_TYPE_SUITE'].value_counts()
    fig = go.Figure( go.Bar(x=xx, y=yy ,text=yy,
                textposition='auto',) )
    st.plotly_chart(fig)



    st.subheader('How many users have previous contract opened ')

    xx = credit_card_balance['NAME_CONTRACT_STATUS'].unique()
    #print(xx)
    yy = credit_card_balance['NAME_CONTRACT_STATUS'].value_counts()

    fig = go.Figure(data=[go.Pie(labels=xx, values=yy)])
    st.plotly_chart(fig)

    # NAME_CONTRACT_TYPE
def model_predict():
    import streamlit as st
    import time
    import numpy as np
    import pandas as pd
    import plotly.figure_factory as ff
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    #import seaborn as sns
    import xgboost as xgb

    st.markdown(f"# {list(page_names_to_funcs.keys())[4]}")

    st.markdown("# EDA on other dataset  Demo")
    st.sidebar.header("Plotting Demo")
    st.write(
        """This demo in the page illustrates a heuristic view of the prediction from pre-defined model ."""
    )








    model2 = xgb.XGBClassifier()
    model2.load_model("model_1.json")

    num = st.slider('Select a random number to predict using xgboost ', 0, 3000, 25)
    st.write("You have selected row number  ",num  , 'out of 3000 records')




    #loading final model

    final_model = pd.read_csv("test_df_model_12.csv")

    final_model.pop('Unnamed: 0')

    secondrow = pd.DataFrame(final_model.iloc[num]).T

    secondpred = model2.predict(secondrow)


    st.header('The predicted value for {} row is {}'.format(num,secondpred[0]))


    st.write('The column values for this are : ')

    st.dataframe(final_model.iloc[num])


page_names_to_funcs = {
    "Introduction": intro,
    "EDA on Train set Demo": page_1,
    "EDA on page 2 Demo": page_2,
    "EDA On page 3 Demo": page_3,
    "Prediction of model Demo": model_predict
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
