import streamlit as st
st.set_page_config(page_title="My Portfolio", layout="wide")
st.markdown("""
    <style>
    /* เปลี่ยนสีปุ่ม */
    div.stButton > button:first-child {
        background-color: #0068C9; /* สีน้ำเงิน (เปลี่ยนรหัสสีได้) */
        color: white;               /* สีตัวอักษร */
        border: none;
        border-radius: 8px;         /* ความมนของขอบ */
    }
    /* เปลี่ยนสีตอนเอาเมาส์ไปจ่อ (Hover) */
    div.stButton > button:first-child:hover {
        background-color: #004B91;  /* สีน้ำเงินเข้มขึ้น */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go_to_page(page_name):
    st.session_state.page = page_name

def show_home():
    col1, col2 = st.columns([0.7, 0.3], gap="medium")
    with col1:
        st.title("Natrapa Srivicha")
        st.markdown("""
        ### Data Analyst/Data Science
        """)
        st.markdown("""Hello I’m **Natrapa**, a third-year Computer Science student passionate about uncovering hidden patterns in raw data to help businesses maximize growth and minimize potential losses.""")
    with col2:
        st.write("") 
        st.write("") 
        st.write("") 
        st.markdown("""
        <div style="text-align: right;">
            
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/natrapa-srivicha-8490813a2/)
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/natrapa-s)
        [![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:friendnatrapa.s@gmail.com)
        
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
            ### Education
                Khonkaen University
                Bachelor of Science Program in Computer Science
                GPAX: 3.63
    """)
    st.divider()

    
    st.divider()
    st.subheader("My Projects")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        with st.container(border=True, height=600):
            st.image("img/NASA/overview_FD001.png", use_container_width=True) 
            st.subheader("NASA Turbofan engines predictive(FD001)")
            st.write("This project utilized the NASA Turbofan engine simulated dataset.The study was divided into two parts: first, I used a Random Forest Regressor to predict RUL under single operating conditions. Second, I employed an LSTM (Long Short Term Memory) network to predict RUL under complex, multiple operating conditions.")
                
            if st.button("View", key="btn_nasa1"):
                go_to_page('nasa') 
    with col2:
        with st.container(border=True, height=600):
            st.image("img/Walmart/overview.png", caption="Dashboard(Overview)", use_container_width=True) 
            st.subheader("Walmart Sales factors analysis")
            st.write("This project focuses on Walmart sales correlate with external factors.I used a RandomForestRegressor to train the model and examine feature importance to see which factors have the highest impact on sales.")
                
            if st.button("View", key="btn_walmart"):
                go_to_page('walmart')

    with col3:
        with st.container(border=True, height=600):
            st.image("img/OECD/overview_oil.png", caption="Dashboard(Overview)", use_container_width=True)
            st.subheader("OECD Crude Oil Production")
            st.write("This project analyzes global oil production from 1960 to 2017. Since each country exhibits unique trends due to differing economic and political factors, forecasting future production helps indicate potential supply risks and evaluate energy security stability.")
            if st.button("View", key="btn_oil"):
                go_to_page('oil')
def show_nasa_project():
    if st.button("← Back to Home"):
        go_to_page('home')
        st.rerun()
        
    st.title("NASA Turbofan engines predictive(FD001)")
    st.markdown("""
    **Problem:** The NASA Turbofan engines operate perfectly initially, they gradually degrade over time due to wear and tear.Therefore, predicting the RUL is crucial to minimize maintenance costs and prevent severe failures.

    **What I do:** I trained and tested a RandomForestRegressor model to predict the RUL of NASA engines. I also optimized the model to achieve the best RMSE and R2 Score (which indicate the accuracy of the model).
    """)
    st.divider()
    tab1, tab2 = st.tabs(["FD001 Dataset", "FD004 Dataset"])

    with tab1:
        #overview
        st.header("Dashboard (FD001-Single Condition)")
        st.write("Displays KPIs including **Actual RUL**, **Predicted RUL**, and model performance metrics.")
        st.image("img/NASA/overview_FD001.png", caption="Dashboard(Overview)", use_container_width=True)
        st.divider()
        #deep drive
        st.header("Key Insights")
        st.image("img/NASA/error_by_number_edited.png", caption="RUL Prediction Dashboard: Error Analysis & Sensor Value Correlation", use_container_width=True)

        st.divider()

        st.header("Technical Skills")
        st.subheader("Example Python code")
        st.write("This snippet demonstrates how I optimized the model performance. I used Grid Search to test various combinations of trees (n_estimators) and depth (max_depth) to identify the best configuration.")

        with st.expander("Grid Search"):
            st.code('''
            from sklearn.model_selection import GridSearchCV
            param_grid = {
                "n_estimators": [50, 100, 200],      
                "max_depth": [10, 15, 20, None],     
                "min_samples_split": [2, 5]         
            }

            #make empty model
            rf = RandomForestRegressor(random_state=42)

            #grid search setup
            #cv(cross validation) = 3
            # n_jobs=-1:use cpu all of core
            # verbose=2: show message while processing
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                    cv=3, n_jobs=-1, verbose=2, scoring="neg_root_mean_squared_error")

            print("finding for the best parameter")

            #fit(3x4x2 = 24 แบบ)
            grid_search.fit(X_train, y_train_clipped)

            print(f"Best parameter: {grid_search.best_params_}")
            print(f"Best RMSE ช่วง Train: {-grid_search.best_score_:.4f}")
        ''' ,language='python')
            
        st.write("Training the Random Forest model and validating the improvement gained from using Clipped RUL targets.")
        with st.expander("RandomForestRegressor"):
            st.code('''
            #train and test model with y_train_clipped
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        rf_model.fit(X_train,y_train)
        rf_model_v2 = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf_model_v2.fit(X_train, y_train_clipped)

        #predict with X_test
        y_true = y_test["RUL"]
        y_pred = rf_model.predict(X_test)
        y_pred_v2 = rf_model_v2.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_true,y_pred))
        rmse_v2 = np.sqrt(mean_squared_error(y_true, y_pred_v2))
        r2_v2 = r2_score(y_true, y_pred_v2)

        print("Result after use clipped RUL")
        print(f"Old RMSE: {rmse:.2f} -> New RMSE: {rmse_v2:.2f}")
        print(f"R2 Score: {r2_v2:.2f}")
        ''' ,language='python')
            
    with tab2:
        #overview
        st.header("Dashboard (FD004-Multiple Conditions)")
        st.write("Displays KPIs including **Actual RUL**, **Predicted RUL**, and model performance metrics.")
        st.image("img/NASA/overview_FD004.png", caption="Dashboard(Overview)", use_container_width=True)
        st.image("img/NASA/overview_FD004_2.png", caption="Dashboard(Selected Unit)", use_container_width=True)
        st.divider()
        st.header("Key Insights")
        st.image("img/NASA/deep_learning_edited.png", caption="Deep Learning:Error Analysis & Trainig Loss", use_container_width=True)
        st.divider()

        st.header("Technical Skills")
        st.subheader("Example Python code")
        st.write("I implemented `GridSearchCV` to fine-tune the Random Forest Regressor. The process involved iterating through parameters like tree count (`n_estimators`), tree depth (`max_depth`), and split criteria to prevent overfitting and improve generalization.")

        with st.expander("Grid Search"):
            st.code('''
            from sklearn.model_selection import GridSearchCV
            param_grid = {
                "n_estimators": [50, 100, 200],      
                "max_depth": [10, 15, 20, None],     
                "min_samples_split": [2, 5]         
            }

            #make empty model
            rf = RandomForestRegressor(random_state=42)

            #grid search setup
            #cv(cross validation) = 3
            # n_jobs=-1:use cpu all of core
            # verbose=2: show message while processing
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                    cv=3, n_jobs=-1, verbose=2, scoring="neg_root_mean_squared_error")

            print("finding for the best parameter")

            #fit(3x4x2 = 24 แบบ)
            grid_search.fit(X_train, y_train_clipped)

            print(f"Best parameter: {grid_search.best_params_}")
            print(f"Best RMSE ช่วง Train: {-grid_search.best_score_:.4f}")
        ''' ,language='python')
            
        st.write("Upgraded the model to **XGBoost** and engineered **Temporal Features** (Rolling Statistics) to better capture complex degradation patterns, resulting in improved prediction accuracy for multi-condition scenarios.")
        with st.expander("XGBRegressor"):
            st.code('''
        from xgboost import XGBRegressor
        from scipy.stats import linregress
        train_base = train_norm.copy()
        test_base = test_norm.copy()
        def get_slope(series):
            #calculate slope in each window
            y = series.values
            x = np.arange(len(y))
            # slope=(Covariance / Variance)
            slope = np.polyfit(x, y, 1)[0]
            return slope

        def add_advanced_features_xgb(df, window_size=5):
            df_copy = df.copy()
            df_copy = df_copy.sort_values(["unit_number", "time_cycles"])
            cols = [c for c in sensor_names if c in df_copy.columns]

            #Rolling Mean
            rolling_mean = df_copy.groupby("unit_number")[cols].rolling(window=window_size).mean().reset_index(drop=True)
            rolling_mean.columns = [f"{c}_mean" for c in cols]

            #Rolling Diff (Trend)
            rolling_diff = df_copy.groupby("unit_number")[cols].diff(periods=window_size).fillna(0).reset_index(drop=True)
            rolling_diff.columns = [f"{c}_trend" for c in cols]

            #merge
            df_out = pd.concat([df_copy, rolling_mean, rolling_diff], axis=1)
            df_out = df_out.fillna(0)
            return df_out

        #training for xgboost
        train_xgb = add_advanced_features_xgb(train_base, window_size=5) #ใช้ Window=5 เท่ากันเพื่อเปรียบเทียบ
        test_xgb = add_advanced_features_xgb(test_base, window_size=5)

        #clean duplicate
        train_xgb = train_xgb.loc[:, ~train_xgb.columns.duplicated()]
        test_xgb = test_xgb.loc[:, ~test_xgb.columns.duplicated()]

        # Prepare Data
        X_train_xgb = train_xgb.drop(feature_to_drop + ["RUL"], axis=1)
        y_train_xgb = train_xgb["RUL"]

        # Train XGB
        xgb_model = XGBRegressor(
            n_estimators=5000,      
            learning_rate=0.005,    
            max_depth=6,            
            min_child_weight=2,     
            gamma=0.05,             
            subsample=0.5,          
            colsample_bytree=0.5,   
            reg_alpha=0.1,          
            reg_lambda=1.0,         
            n_jobs=-1,
            random_state=42,
            objective="reg:squarederror"
        )

        xgb_model.fit(X_train_xgb, y_train_xgb)

        #evaluate XGB
        test_last_xgb = test_xgb.merge(last_cycle_test, on=["unit_number"], how="inner")
        test_last_xgb = test_last_xgb[test_last_xgb["time_cycles"] == test_last_xgb["last_cycle"]]

        X_test_xgb = test_last_xgb.drop(feature_to_drop + ["last_cycle"], axis=1)
        X_test_xgb = X_test_xgb[X_train_xgb.columns]

        y_pred_xgb = xgb_model.predict(X_test_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test["RUL"], y_pred_xgb))
        r2_xgb = r2_score(y_test["RUL"], y_pred_xgb)

        print(f"XGBoost -> RMSE: {rmse_xgb:.2f}, R2: {r2_xgb:.2f}")
    ''' ,language='python')
            
        st.write("For the FD004 time-series dataset, I evaluated classical Machine Learning models as baselines. Since equipment degradation depends on temporal patterns across operating cycles, I additionally **experimented** with an LSTM model to capture sequential dependencies that static models may miss. While **the LSTM showed only marginal improvement** , this experiment helped assess the trade-offs between model performance, interpretability, and deployment complexity.")

        with st.expander("LSTM"):
            st.code('''
            from sklearn.model_selection import GridSearchCV
                param_grid = {
                        "n_estimators": [50, 100, 200],      
                        "max_depth": [10, 15, 20, None],     
                        "min_samples_split": [2, 5]         
                    }

                    #make empty model
                    rf = RandomForestRegressor(random_state=42)

                    #grid search setup
                    #cv(cross validation) = 3
                    # n_jobs=-1:use cpu all of core
                    # verbose=2: show message while processing
                    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                            cv=3, n_jobs=-1, verbose=2, scoring="neg_root_mean_squared_error")

                    print("finding for the best parameter")

                    #fit(3x4x2 = 24 แบบ)
                    grid_search.fit(X_train, y_train_clipped)

                    print(f"Best parameter: {grid_search.best_params_}")
                    print(f"Best RMSE ช่วง Train: {-grid_search.best_score_:.4f}")
                    #reshape data for LSTM
                    #3D:[จำนวนตัวอย่าง, ระยะเวลาย้อนหลัง, จำนวนฟีเจอร์]
                    window_size = 30#look back for 30 cycles
                    def gen_sequence(id_df, seq_length, seq_cols):
                        data_matrix = id_df[seq_cols].values
                        num_elements = data_matrix.shape[0]
                        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
                            yield data_matrix[start:stop, :]

                    def gen_labels(id_df, seq_length, label):
                        data_matrix = id_df[label].values
                        num_elements = data_matrix.shape[0]
                        return data_matrix[seq_length:num_elements, :]
                    #prepare X_train,y_train
                    seq_gen = (list(gen_sequence(train_lstm[train_lstm["unit_number"]==id],window_size,features))
                            for id in train_lstm["unit_number"].unique())
                    X_train_lstm = np.concatenate(list(seq_gen)).astype(np.float32)

                    label_gen = (gen_labels(train_lstm[train_lstm["unit_number"]==id],window_size,["RUL"])
                                for id in train_lstm["unit_number"].unique())
                    y_train_lstm = np.concatenate(list(label_gen)).astype(np.float32)
                    print(f"X_train shape: {X_train_lstm.shape}")
                    print(f"y_train shape: {y_train_lstm.shape}")
                    #build lstm model
                    model = Sequential()
                    #layer 1:lSTM
                    model.add(LSTM(128, input_shape=(window_size, len(features)), return_sequences=True))
                    model.add(Dropout(0.2)) # กัน Overfitting

                    #layer 2:lSTM
                    model.add(LSTM(64, return_sequences=False))
                    model.add(Dropout(0.2))

                    #layer 3:output
                    model.add(Dense(32, activation="relu"))
                    model.add(Dense(1)) #output=RUL
                    model.compile(loss="mean_squared_error", optimizer="adam")
                    #train
                    history = model.fit(X_train_lstm, y_train_lstm, epochs=30, batch_size=200, validation_split=0.05, verbose=1)
                    #evaluation
                    print("evaluating on test set")
                    X_test_lstm = []
                    y_true_lstm = []
                    for i in test_lstm["unit_number"].unique():
                        tmp_test = test_lstm[test_lstm["unit_number"]==i]
                        if len(tmp_test) >= window_size:
                            last_seq = tmp_test[features].values[-window_size:]
                            X_test_lstm.append(last_seq)
                            y_true_lstm.append(y_test_lstm.iloc[i-1]["RUL"])

                    X_test_lstm = np.array(X_test_lstm)
                    y_true_lstm = np.array(y_true_lstm)
                    y_pred_lstm = model.predict(X_test_lstm)
                    rmse_lsmt = np.sqrt(mean_squared_error(y_true_lstm, y_pred_lstm))
                    r_lsmt = r2_score(y_true_lstm, y_pred_lstm)
                    print(f"Deep Learning Result (FD004)")
                    print(f"RMSE: {rmse_lsmt:.2f} cycles")
                    print(f"R2 Score: {r_lsmt:.2f}")

                    plt.plot(y_true_lstm, label="Actual")
                    plt.plot(y_pred_lstm, label="Predicted")
                    plt.legend()
                    plt.show()  
                    
        ''' ,language="python")         
def show_walmart_project():
    if st.button("← Back to Home"):
        go_to_page('home')
        st.rerun()
        
    st.title("Walmart Sales Analysis")
    st.markdown("""
    **Problem:** Walmart is a store with many branches.Understanding how factors such as seasonality impact with sales helps in preparing for customer demand and maximizing business revenue.

    **What I do:** Analyzed the correlation between environmental and economic indicators and Walmart's revenue. I focused on identifying seasonal trends (2010–2012) to determine which factors drive business success.
    """)
    st.divider()

    #overview
    st.header("Dashboard (Overview)")
    st.write("Displays KPIs including actual sales, predicted sales, and forecast accuracy(%).")
    col1, col2 = st.columns([2, 1]) #รูปใหญ่กว่า
    with col1:
        st.image("img/Walmart/overview.png", caption="Dashboard(Overview)", use_container_width=True)
    with col2:
        st.image("img/Walmart/holiday.png", caption="Sales(None-Holiday)", use_container_width=True)
        st.divider()
        st.image("img/Walmart/none-holiday.png", caption="Sales(Holiday)", use_container_width=True)

    st.divider()
    st.header("Key Insights")

    st.image("img/Walmart/factors.png", caption="Factors Analysis", use_container_width=True)

    st.divider()

    st.header("Technical Skills")
    st.subheader("Example Python code")
    st.write("In this section, I implemented a Random Forest Regressor to predict weekly sales and capture seasonal trends. Finally, I extracted Feature Importance to quantify which factors have the highest impact on revenue.")

    with st.expander("RandomForestRegressor"):
        st.code('''
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error,mean_squared_error
            df = df.sort_values(by=["store","date"])
            print("Create a Lag and Moving Avg feature")
            #Lag1 sales of last week(short momentum)
            df["sales_lag1"] = df.groupby("store")["weekly_sales"].shift(1)
            #moving avg4 sales of latest 4 weeks(trend of this month)
            df["sales_avg4weeks"] = df.groupby("store")["weekly_sales"].shift(1).rolling(window=4).mean()
            #lag52 sales of this week in last year(seasonality yearly)
            df["sales_lag52"] = df.groupby("store")["weekly_sales"].shift(52)
            df_model = df.dropna()

            print(f"Original data rows:{len(df)}")
            print(f"Rows after Feature Engineering:{len(df_model)}")

            #train model
            features = [
                "store",
                "sales_lag1","sales_avg4weeks","sales_lag52",#major
                "temperature","fuel_price","cpi","unemployment",#external factor
                "holiday_flag","week","month"#time
            ]
            target = "weekly_sales"

            x = df_model[features]
            y = df_model[target]

            #split Train and Test(80/20)
            X_train,X_test = X.iloc[:split_point],X.iloc[split_point:]
            y_train,y_test = y.iloc[:split_point],y.iloc[split_point:]

            print(f"Train: {len(X_train)} rows")
            rf_model = RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
            rf_model.fit(X_train,y_train)

            #evaluation
            predictions = rf_model.predict(X_test)

            #error
            mae = mean_absolute_error(y_test,predictions)
            mape = np.mean(np.abs((y_test-predictions)/y_test))*100

            print(f"result(Model Performance): ")
            print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

            #fearture importance
            feature_importance = pd.DataFrame({
                "Feature" : features,
                "Importance" : rf_model.feature_importances_
            }).sort_values(by="Importance",ascending=False)

            plt.figure(figsize=(12,6))
            sns.barplot(data=feature_importance,x="Importance",y="Feature")
            plt.title("Final Verdict: What drives Weekly Sales?",fontsize=16)
            plt.xlabel("Importance Score")
            plt.ylabel("Factors")
            plt.grid(axis="x",alpha=0.3)
            plt.show()

            print("factor impact with sales")
            print(feature_importance)
    ''', language='python')

def show_oil_production_project():
    if st.button("← Back to Home"):
        go_to_page('home')
        st.rerun()
    
    st.title("OECD Crude Oil Production")
    st.markdown("""
    **Problem:** Global energy markets are highly sensitive to production fluctuations. Stakeholders struggle to identify which countries pose supply risks due to historical volatility, making long-term planning difficult.

    **What I do:** I built an Automated Forecasting Engine using Python and ARIMA models. The system evaluates multiple parameters and automatically selects the best-fitting model (lowest MAPE) for each country, ensuring high accuracy even for volatile datasets.
    """)
    st.divider()
    st.header("Dashboard (Overview)")
    st.write("A comprehensive visualization displaying production metrics for major oil suppliers. The dashboard utilizes KPIs to summarize total and average output, while the interactive map and line charts facilitate a direct comparison of production trends among the top 5 countries (Iran, Iraq, Israel, Russia, Saudi Arabia).")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("img/OECD/overview_oil.png", caption="Dashboard(Overview)", use_container_width=True)
    with col2:
        st.image("img/OECD/overview_oil2.png", caption="Dashboard(Selected Country)", use_container_width=True)
    st.divider()

    #deep drive
    st.header("Key Insights")
    st.write("The model achieved a MAPE of 1.11% (<10%), indicating high predict accuracy. As observed, the model closely tracks the historical data over the past decade. The latest production levels remain stable, while the future forecast suggests a gradual upward trend.")
    st.image("img/OECD/top5.png", caption="Top 5 highest oil production", use_container_width=True)
    st.write("The model achieved a MAPE of 13.09%. While slightly above the standard 10% threshold, this is expected given Sudan's extreme historical volatility and the structural break (production crash) in 2012. The model successfully adapts to the new low-production regime, predicting a stabilization trend (flat line) at approximately 4.3k units for 2018-2022, rather than falsely predicting a recovery.")
    st.image("img/OECD/top5volatile.png", caption="Top 5 highest volatile oil production", use_container_width=True)
    st.divider()

    st.header("Technical Skills")
    st.write("In this section, I implemented a Python function for **Trend Forecasting**. The code dynamically filters data by country, reshapes it for the model, and visualizes the results. It plots the **Actual Data** against the **Trend Line** to give a clear visual indication of future production capacity.")

    with st.expander("LinearRegression"):
        st.code('''
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import timedelta
        from sklearn.linear_model import LinearRegression

        def simple_forecast(df,locations):
            data = df[df["location"] == locations].copy()

            if locations not in df["location"].unique():
                print("No this country in db")
                return
            
            X = data["time"].values.reshape(-1,1)#horizontal to vertical
            y = data["value"].values

            model = LinearRegression()
            model.fit(X,y)

            last_year = int(data["time"].max())
            next_year = last_year + 1
            prediction = model.predict([[next_year]])[0]#attract val from array

            plt.figure(figsize=(10,5))
            future_year = np.arange(data["time"].min(),next_year+1).reshape(-1,1)
            trend_line = model.predict(future_year)

            plt.scatter(X,y,color="blue",alpha=0.6,label="Actual Data")
            plt.plot(future_year,trend_line,color="red", linestyle="--", label="Trend Line")
            plt.scatter([next_year], [prediction], color="green", s=150, marker="*", label=f"Forecast {next_year}")

            plt.title(f"Oil Production Forecast (Linear): {locations}", fontsize=14, fontweight="bold")
            plt.xlabel("Year")
            plt.ylabel("Production (KTOE)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            print(f"result of forecasting country {locations}:")
            print(f"last year ({last_year}): {y[-1]:,.2f} KTOE")
            print(f"forecasting next year({next_year}): {prediction:,.2f} KTOE")
        ''', language='python')

    st.write("Advanced Forecasting Engine (Auto-ARIMA) To capture complex patterns and volatility, I developed an **Automated ARIMA Pipeline**. Instead of using fixed parameters, the algorithm iterates through multiple hyperparameter combinations `(p,d,q)` for each country. It validates the models against the last 5 years of data and automatically selects the one with the **lowest Error (MAPE)** to generate the final 5-year forecast.")

    with st.expander("ARIMA"):
        st.code('''
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import timedelta
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_squared_error
        from sklearn.linear_model import LinearRegression
        import warnings

        #close warning
        warnings.filterwarnings("ignore")
        def forecasting_none_linear(df, target_country):
            
            if target_country not in df["location"].unique():
                print(f"No data for {target_country}")
                return

            #prepare data
            df_target = df[df["location"] == target_country].copy()
            df_target["year_date"] = pd.to_datetime(df_target["time"], format="%Y")
            df_target = df_target.set_index("year_date")
            
            series = df_target["value"].asfreq("YS").fillna(method="ffill")
            
            #split for train and test
            train = series[:-5]
            test = series[-5:]
            
            print(f"Searching for Lowest MAPE for {target_country}")
            
            candidate_orders = [
                (2,1,2), (1,1,1), (2,1,0), (0,1,1), (1,1,0), (0,1,0), (1,0,0)
            ]
            
            best_mape = float("inf")
            best_order = None
            best_forecast_values = None
            best_conf_int = None
            
            #loop to find the best forecast
            for param in candidate_orders:
                try:
                    #train model on history
                    temp_model = ARIMA(train, order=param, enforce_stationarity=False)
                    results = temp_model.fit()
                    
                    #forecast in next 5 years
                    forecast_res = results.get_forecast(steps=5)
                    pred_vals = forecast_res.predicted_mean
                    
                    abs_err = np.abs(pred_vals - test)
                    abs_val = np.abs(test)
                    
                    mape_val = np.mean(abs_err / abs_val.replace(0, 1e-6)) * 100
                    
                    if mape_val < best_mape:
                        best_mape = mape_val
                        best_order = param
                        best_forecast_values = pred_vals
                        best_conf_int = forecast_res.conf_int()
                        
                except Exception as e:
                    continue
                    
            print(f"the best ARIMA {best_order} with MAPE={best_mape:.2f}%")

            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train, label="Training Data", color="black")
            plt.plot(test.index, test, label="Actual Data (Test)", color="green", marker="o")
            plt.plot(test.index, best_forecast_values, label=f"Winner {best_order} Forecast", color="red", linestyle="--", marker="x")
            
            if best_conf_int is not None:
                plt.fill_between(test.index, best_conf_int.iloc[:, 0], best_conf_int.iloc[:, 1], color="pink", alpha=0.3)
                
            plt.title(f"Best Model Validation for {target_country} (Optimized for Error)", fontsize=14)
            plt.ylabel("Oil Production")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            #future forecast
            final_model = ARIMA(series, order=best_order, enforce_stationarity=False) 
            final_model_fit = final_model.fit()

            future_forecast = final_model_fit.get_forecast(steps=5)
            future_values = future_forecast.predicted_mean
            future_conf = future_forecast.conf_int()

            print(f"Future Forecast (2018-2022) using {best_order}")
            print(future_values)

            plt.figure(figsize=(10, 5))
            plt.plot(series.index[-10:], series[-10:], label="History (Last 10 Years)", color="black")
            plt.plot(future_values.index, future_values, label=f"Future Forecast {best_order}", color="blue", marker="o", linestyle="--")
            plt.fill_between(future_values.index, future_conf.iloc[:, 0], future_conf.iloc[:, 1], color="lightblue", alpha=0.3)
            plt.title(f"Future Prediction: {target_country}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        def generate_forecast_csv(df):
            all_results = []
            countries = df["location"].unique()
            for i, country in enumerate(countries):
                try:
                    if i % 10 == 0:
                        print(f"Processing... {i}/{len(countries)}")

                    df_target = df[df["location"] == country].copy()
                    if len(df_target) < 10: continue #skip for low data

                    df_target["year_date"] = pd.to_datetime(df_target["time"], format="%Y")
                    df_target = df_target.set_index("year_date")
                    series = df_target["value"].asfreq("YS").fillna(method="ffill")
                    last_year_actual = series.index[-1].year 

                    #model selection
                    train = series[:-5]
                    test = series[-5:]
                
                    candidate_orders = [(2,1,2), (1,1,1), (2,1,0), (0,1,1), (1,1,0), (0,1,0)]
                    
                    best_mape = float("inf")
                    best_order = (2, 1, 2) #set default
                    
                    for param in candidate_orders:
                        try:
                            #train
                            temp_model = ARIMA(train, order=param, enforce_stationarity=False)
                            results = temp_model.fit()
                            
                            #predict correlate with test
                            forecast_res = results.get_forecast(steps=5)
                            pred_vals = forecast_res.predicted_mean
                            
                            #mape
                            abs_err = np.abs(pred_vals - test)
                            abs_val = np.abs(test)
                            #replace 0 with small number to avoid div by zero
                            current_mape = np.mean(abs_err / abs_val.replace(0, 1e-6)) * 100
                        
                            if current_mape < best_mape:
                                best_mape = current_mape
                                best_order = param
                        except:
                            continue
                    
                    #forecast 
                    try:
                        final_model = ARIMA(series, order=best_order, enforce_stationarity=False)
                        final_model_fit = final_model.fit()
                        
                        #forecast in next 5 year 
                        steps = 5
                        forecast_result = final_model_fit.get_forecast(steps=steps)
                        forecast_values = forecast_result.predicted_mean.values 
                        conf_int = forecast_result.conf_int().values
                    except:
                        continue

                    #save in the list
                    #history
                    for date_idx, value in series.items():
                        all_results.append({
                            "location": country,
                            "year": date_idx.year,
                            "value": value,
                            "type": "History",
                            "lower_bound": value,
                            "upper_bound": value
                        })

                    #forecast
                    for k in range(steps):
                        next_year = int(last_year_actual + 1 + k)
                        pred_val = forecast_values[k]
                        lower = conf_int[k][0]
                        upper = conf_int[k][1]

                        all_results.append({
                            "location": country,
                            "year": next_year,
                            "value": max(pred_val, 0),
                            "type": "Forecast",
                            "lower_bound": max(lower, 0),
                            "upper_bound": max(upper, 0)
                        })

                except Exception as e:
                    print(f"Error on {country}: {e}")
                    continue

            df_final = pd.DataFrame(all_results)
            print("CSV have maked with lowest mape already")
            return df_final
        ''', language='python')

if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'nasa':
    show_nasa_project()
elif st.session_state.page == 'walmart':
    show_walmart_project()
elif st.session_state.page == 'oil':
    show_oil_production_project()
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Thank You!</h1>
    </div>
    """,
    unsafe_allow_html=True
)