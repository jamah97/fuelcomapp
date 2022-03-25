
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

# ML pkgs
# ML pkgs
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def main():

	df = pd.read_csv("FuelConsumptionCo2.csv")
	df = df.drop('MODELYEAR', axis=1)
	df1 = df.select_dtypes(exclude=['object'])
	df2 = df.select_dtypes(include='object')


	st.subheader("About")
	st.write("Creator of App Hassan Jama. Application has 2 sections, Model building and EDA. In the model building section, the task is to perform a linear regression on the dataset. You will be able to select the algorithm you want to use along with the training and test split. Plus which variables will be the predictor variables and the predicted variable. At the end of the model building section, you'll be able to look at model accuracy and mean squared errors for your inputs. In the EDA section you will have the opportunity to explore and visualize the data.")
	st.write("Datasourse:https://www.kaggle.com/vishalvishwa/fuelconsumptionco2 ")



	st.subheader("Model Buidling")
	algorithm = ["Linear Regression", "Decision Tree Regressor", "SVR"]
	choice = st.selectbox("Select Algorithm",algorithm)
	testsize = st.slider('Testing size: Select percent of data that will be used as testing data.', 0, 50)


	#iv = st.multiselect("Select x Columns", df1.columns.to_list())
	#iv = df1[iv].values
	#dv = st.selectbox("Select y Column", df1.columns.to_list())
	#dv = df1[dv].values

	if choice == "Linear Regression":
		iv = st.multiselect("Select predictor variables", df1.columns.to_list())
		iv = df1[iv].values
		dv = st.selectbox("Select Target variables", df1.columns.to_list())
		dv = df1[dv].values
		dv2 = pd.DataFrame(data=dv)
		iv2 = pd.DataFrame(data=iv)
		if st.button("Model Performance"):
			train_x, test_x, train_y, test_y = train_test_split(iv2,dv2, test_size=testsize, random_state=1)
			LM = linear_model.LinearRegression()
			LM.fit(train_x, train_y)
			Yhat0=LM.predict(test_x)
			st.write("Accuracy of linear regression model", LM.score(test_x, test_y))
			st.write("Mean Squared Errors:", mean_squared_error(test_y, Yhat0))

	if choice == "Decision Tree Regressor":
		ivd = st.multiselect("Select predictor variables", df1.columns.to_list())
		ivd = df1[ivd].values
		dvd = st.selectbox("Select Target variables", df1.columns.to_list())
		dvd = df1[dvd].values
		dvd2 = pd.DataFrame(data=dvd)
		iv2d = pd.DataFrame(data=ivd)
		if st.button("Model Performance"):
			train_x, test_x, train_y, test_y = train_test_split(iv2d,dvd2, test_size=testsize, random_state=1)
			dtr = DecisionTreeRegressor(random_state=1)
			dtr.fit(train_x, train_y)
			Yhat=dtr.predict(test_x)
			st.write("Accuracy of Decision Tree Regressor model", dtr.score(test_x, test_y))
			st.write("Mean Squared Errors:", mean_squared_error(test_y, Yhat))

	if choice == "SVR":
		ivd1 = st.multiselect("Select predictor variables", df1.columns.to_list())
		ivd1 = df1[ivd1].values
		dvd1 = st.selectbox("Select Target variables", df1.columns.to_list())
		dvd1 = df1[dvd1].values
		dvd21 = pd.DataFrame(data=dvd1)
		iv2d1 = pd.DataFrame(data=ivd1)
		if st.button("Model Performance"):
			train_x, test_x, train_y, test_y = train_test_split(iv2d1,dvd21, test_size=testsize, random_state=1)
			srv = SVR()
			srv.fit(train_x, train_y)
			Yhat2=srv.predict(test_x)
			st.write("Accuracy of SVR (simple vector regressor) model", srv.score(test_x, test_y))
			st.write("Mean Squared Errors:", mean_squared_error(test_y, Yhat2))



	st.subheader("EDA")
	st.write("First 5 rows of fuel consumption Co2 dataset")
	st.write(df.head())

	if st.button("Numerical anaysis"):
	    c1, c2 = st.columns(2)
	    with cl:
                with st.expander('Statistical Summary'):
			st.write(df.describe())
		    # st.write('Here is a statistical summary of the numerical values in the dataset')
		      #st.write(df.describe())
	    with c2:
	        with st.expander('Correlation Summary'):
		    # st.write('Here is a correlation summary of the numerical values in the dataset')
		   st.write(df.corr())

	cv= df2.columns.tolist()
	#cv1= df1.columns.tolist()
	#cv2= df1.columns.tolist()

	if st.checkbox("Value counts and bar graphs of categorical variables in the dataset"):
		columnsx = st.selectbox("Select Categorical Column",cv)
		for i in df2.columns:
			if columnsx == i:
				st.dataframe(df2[i].value_counts())
		st.write(sns.countplot(x=columnsx, data = df2, palette = 'deep'))
		st.pyplot()

	#if st.checkbox("Box plot of all the numerical variable and CO2 emissions"):
		#columnsx1 = st.selectbox("Select X Column",cv1)
		#st.write(sns.set(rc={"figure.figsize":(25, 25)}))
		#st.write(sns.boxplot(x=columnsx1, y="CO2EMISSIONS", data=df1))
		#st.write(plt.xlabel(columnsx1,size=25))
		#st.write(plt.ylabel("CO2EMISSIONS",size=25))
		#st.pyplot()


	#if st.checkbox("Scatter plot of all the numerical variable and CO2 emissions"):
		#columnsx = st.selectbox("Select X Column",cv2)
		#st.write(sns.set(rc={"figure.figsize":(25, 25)}))
		#st.write(sns.scatterplot(x=columnsx, y="CO2EMISSIONS", data=df1))
		#st.write(plt.xlabel(columnsx,size=15))
		#st.write(plt.ylabel("CO2EMISSIONS",size=15))
		#st.pyplot()

	if st.checkbox("distribution plot of c02 emissions"):
		st.write(sns.distplot(df['CO2EMISSIONS']))
		st.pyplot()






	cv2= df1.columns.tolist()
	type_of_plot = st.selectbox("Select Type of Plot for numerical variable",["Scatter plot","Box plot"])
	selected_columns_names = st.selectbox("Select Columns To Plot against CO2 emissions",cv2)

	if st.button("Generate Plot"):
		st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

				# Plot By Streamlit
		if type_of_plot == 'Scatter plot':
			cust_data = df[selected_columns_names]
			st.write(sns.scatterplot(x=cust_data, y="CO2EMISSIONS", data=df1))
			#st.write(plt.xlabel(cust_data,size=25))
			#st.write(plt.ylabel("CO2EMISSIONS",size=25))
			#st.write(plt.figure(figsize=(20,5)))
			st.pyplot()

		elif type_of_plot == 'Box plot':
			cust_data = df[selected_columns_names]
			st.write(sns.boxplot(x=cust_data, y="CO2EMISSIONS", data=df1))
			#st.write(plt.xlabel(cust_data,size=25))
			#st.write(plt.ylabel("CO2EMISSIONS",size=25))
			#st.write(plt.figure(figsize=(20,5)))
			st.pyplot()



if __name__ == '__main__':
	main()
