# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import yfinance as yf
import pandas as pd
from siuba import *
import numpy as np
import seaborn as sns
from shiny import App, render, ui, reactive
import plotly.express as px
import plotly.io as pio
import jinja2
import shinyswatch
import plotly.offline as pyo
from PIL import Image

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from shinywidgets import output_widget, register_widget
import plotly.graph_objs as go
from plotnine import aes, geom_point, geom_smooth, ggplot,geom_line,labs,theme,geom_col
from plotnine import facet_wrap
import plotnine as p9
from plotnine import *
from pathlib import Path
import os
import asyncio
from datetime import date
from htmltools import HTML, div
import shiny.experimental as x
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui
import ipyleaflet as ipyl
from ipyleaflet import Map, basemaps, GeoJSON

# +

tickers = {"BAT": "British American Tobacco Kenya Ltd", "JUB": "Jubilee Insurance", "KCB": "Kenya Commercial Bank",
           "KQ": "Kenya Airways","SASN":"Sasini","TOTL":"Total","NSE":"Nairobi Securities Exchange","SCOM":"Safaricom",
          "GLD":"New Gold"}

navbar = ui.tags.nav(
    ui.h1("NSE Historical Data Analysis"),
    ui.tags.style("h1{color: white;text-align: left; font-size:10}"),
    style="background-color: lightblue; height: 50px; width: 100%; display: flex; flex-direction: row; align-items: center; justify-content: center; text-align: center;"
)

# App UI - Sidebar for selecting ticker and date range, chart and table output
app_ui = ui.page_fluid(
    #shinyswatch.theme.superhero(),
    navbar,
    ui.navset_tab_card(
              ui.nav("Guide",
               ui.tags.style(
                   """
                   .app-col {
                   border: 1px solid black;
                   border-radius: 5px;
                   background-color:#f0f0f0;
                   padding: 8px;
                   margin-top: 5px;
                   margin-bottom: 5px;
                   }
                   .app-col p {
                   background-color: #f0f0f0; 
                   padding: 10px;
                   }
                   """
                   ),
               ui.row(
                   ui.column(
                       12,
                       ui.div(
                           {"class": "app-col"},
                           ui.p(
                               """
                               Welcome to my Dashboard.Designed using Shiny for Python.Having fun using Nairobi Securities Exchange listed stocks data for some 10 companies.
                               The time range is different , for example , Sasini starts from 2000 and Safaricom starts from 2019 upto 2022 March.
                               For Non-Quant sarvy, a ticker is an identifier.The data provider used in pulling the data is Yahoo Finance.
k
                               The Select Stock and Date Range tab will show and download data for your prefered stock.
                               Visuals tab, I have put 4 charts.The top one will plot the volatility and returns according to the radio buttons on that tab.
                               The second plot is on closing prices of the data you selected on the first tab.
                               The third plot is the correlation heatmap of the closing prices of all the stocks within the time range you have selected on the first tab.      
                               """,
                               style="width: 80%; margin: 0 auto; text-align: justify;",
                               ),
                           ui.p(
                               """
                               Derivatives tab is still being developed!!
                               Portfolio Returns.I basically wanted to use this data to quantify diversification benefits.Assumed that you had invested Ksh 100000 on each stock you will select or both(Portifolio)
                               There is a table along side that chart showing volatility of each of those investments.
                              
                               """,
                               style="width: 80%; margin: 0 auto; text-align: justify;",
                               ),
                           ui.p(
                               """
                               Shiny for Python is now generally available.
                               Enjoy exploring the dashboard, and feel free to reach out with any feedback,questions and suggestions!
                              
                               """,
                               style="width: 80%; margin: 0 auto; text-align: justify;",
                               ),
                           ),
                       )
                   ),
                     
               ui.row(
                   ui.column(
                       12,
                       ui.div(
                           [
                               ui.a(
                                   HTML('<i class="fab fa-linkedin fa-2x"></i>'),
                                   href="https://www.linkedin.com/in/muriithi-muriuki/",
                                   target="_blank",
                                   style="margin-right: 10px;"
                                   ),
                               ui.a(
                                   HTML('<i class="fab fa-twitter fa-2x"></i>'),
                                   href="https://twitter.com/brianmuriukii",
                                   target="_blank",
                                   style="margin-right: 10px;"
                                   ),
                               ],
                           )
                       )
                   ),
                     #style="text-align: center;"
                     ),      
        ui.nav("Select Stock and Date Range",
               ui.layout_sidebar(
                   ui.panel_sidebar(
                       ui.input_select(id="ticker", label="Ticker:", choices=tickers),
                       ui.input_date(id="start_date", label="Start Date:", value=(datetime.now() - timedelta(days=8500)).strftime('%Y-%m-%d')),
                       ui.input_date(id="end_date", label="End Date:", value=(datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')),
                       ui.row(
                           ui.column(
                               4,
                           ui.input_action_button("show_data", "Show Data",style="background-color: green;"),
                           ),
                           ui.column(
                               4,
                               ui.download_button("downloadData", "Download",style="background-color: green;"),
                       ),
                       ),
                       style="padding: 10px;background-color: lightblue;width: 300px;"
                        ),
                   ui.panel_main(
                       ui.output_table("table_data"),
                       #ui.output_text("volatility_output"),
                       style="flex-grow: 1;"
                       ),
                   ),
               ),
               #),
        # Right hand side
        ui.nav("Visuals",
               shinyswatch.theme.pulse(),
                   ui.row(
                       ui.column(
                           4,
                           ui.panel_well(
                               ui.input_radio_buttons(
                                   "plot_variable", "Plot Variable", ["Volatility", "Returns"],inline=True
                                   )
                               ),
                           ),
                       ui.column(
                           4,
                           ui.panel_well(
                               ui.input_radio_buttons(
                                   "tick", "Select Ticker", list(tickers.keys()), inline=True
                                   )
                               ),
                           ),
                       ui.column(
                           8,
                           ui.output_plot("plot2",click=True),
                           ),
                       ),
              
               #ui.layout_sidebar(
                  # ui.panel_sidebar(
                       #ui.input_select(id="ticker", label="Ticker:", choices=tickers),
                       #ui.input_date(id="start_date", label="Start Date:", value=(datetime.now() - timedelta(days=2000)).strftime('%Y-%m-%d')),
                       #ui.input_date(id="end_date", label="End Date:", value=(datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')),
                       #style="padding: 10px;"

                   #),
                   ui.tags.div(
                       ui.panel_main(
                           #ui.output_plot("timeplot"),
                           ui.output_plot("vizz"),
                           #ui.output_plot("plot2"),
                           ui.output_plot("corr_heatmap"),
                           
                           style="display: flex; flex-direction: column; margin-left: 20px;"
                       ),
                       style="height: 400px; padding: 10px;"
                   ),
              # ),
        ),
                       # Right hand side
        ui.nav("Derivatives Desk",
               ui.layout_sidebar(
                   ui.panel_sidebar(
                       "Welcome to this Derivatives tab.I have tried using the data to price call and put options using the stock prices as the underlying."

                        "It might be quite technical.I have used different option pricing models such as the popular Black Scholes Model,Heston(1993), Merton'76 and compared the prices they each generate."

                        "Enjoy exploring the dashboard, and feel free to reach out with any feedback or questions!",
                        style="padding: 10px; flex-direction: row-reverse;background-color: lightblue;"
                        ),
                              
                   ui.panel_sidebar(
                       ui.input_select(id="under", label="Underlying Stock:", choices=tickers),
                       ui.input_date(id="sdate", label="Start Date:", value=(datetime.now() - timedelta(days=2000)).strftime('%Y-%m-%d')),
                       #ui.input_numeric("S", "Stock price:", 100, min=0, max=1000),
                       ui.input_numeric("K", "Strike price:", 100, min=0, max=1000),
                       ui.input_numeric("r", "Risk-free rate:", 0.05, min=0, max=1),
                       ui.input_numeric("sigma", "Volatility:", 0.2, min=0, max=1),
                       ui.input_numeric("T", "Maturity (years):", 1, min=0, max=10),
                       ui.input_numeric("t", "Time to option start (years):", 0, min=0, max=10),
                       ui.input_numeric("Ite", "Number of iterations:", 1000, min=1, max=10000),
                       ui.output_text_verbatim("value"),
                       #ui.output_text_verbatim("volatility_output"),
                       style="padding: 10px;background-color: lightblue;"

                   ),
                   ui.tags.div(
                       ui.panel_main(
                           ui.output_text("volatility_output"),
                           #ui.output_plot("corr_heatmap"),
                           
                           style="display: flex; flex-direction: column; margin-left: 20px;"
                       ),
                       style="height: 400px; padding: 10px;"
                   ),
              ),
        ),
        ui.nav("Portfolio Returns",
               ui.layout_sidebar(             
                   ui.panel_sidebar(
                       ui.input_select(id="sasn", label="", choices=["SASN","SCOM", "KQ", "KCB","GLD","BAT","TOTL","NSE","JUB"]),
                       ui.input_select(id="gld", label="", choices=["GLD","SCOM", "KQ", "KCB","SASN","BAT","TOTL","NSE","JUB"]),
                       ui.input_date(id="startdate", label="Start Date:", value=(datetime.now() - timedelta(days=2000)).strftime('%Y-%m-%d')),
                       ui.input_date(id="enddate", label="End Date:", value=(datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')),
                       ui.input_action_button("show_data", "Show Data", class_="btn-success"),
                       style="padding: 10px;width: 250px;"
                       ),
                   ui.panel_main(
                       ui.output_plot("plot"),
                       ui.output_table("portfolio_std"),
                                              
                       style="display: flex; flex-direction: row; background-color: light-grey;"
                       ),
                   ),
               ),
        #ui.nav_spacer(),
        ui.nav("About Me",
               ui.tags.style(
                   """
                   .app-col {
                   border: 1px solid black;
                   border-radius: 5px;
                   background-color: #eee !important;
                   padding: 8px;
                   margin-top: 5px;
                   margin-bottom: 5px;
                   }
                   """
                   ),
               ui.row(
                   ui.column(
                       6,
                       ui.output_ui("images",style="display: flex; justify-content: right;"),
                       ui.p("Brian Muriuki", style="text-align: right; font-size: 16px; font-weight: bold;"),
                       ),
                   ),
               
               ui.row(
                   ui.column(
                       12,
                       ui.div(
                           {"class": "app-col"},
                           ui.p(
                               """
                               Brian is a versatile Data Scientist.
                               A Python, STATA and R programmer.Shiny For Python Engineer.
                               With experience in designing Data collection tools, analysing data and presenting it using interactive dashboards.
                               He is also experienced in working with ODK (Open Data Kit) compliant platforms such as SurveyCTO ,Redcap,CsPro and Kobotoolbox
                               He has worked with MongoDB and MySQL databases and has a good understanding of REST and GraphQL APIs.
                               """,
                               style="width: 80%; margin: 0 auto; text-align: justify;",
                               ),
                           ui.p(
                               """
                               His expertise and interests extend to Quantitative Finance. Ongoing Master of Science in Financial Engineering student at Worldquant University.
                               Coursework in Financial markets, Financial Data, Financial Econometrics, Derivative Pricing,Stochastic Modelling and Machine Learning in Finance.
                               He is a Mathematics and Computer Science(Statistics option) Graduate.
                              
                               """,
                               style="width: 80%; margin: 0 auto; text-align: justify;",
                               ),
                           ),
                       )
                   ),
               ),               
        #ui.style_scope(".nav-link.active", "background-color: red; color: white;"),
    )
       
    )

# Server logic
def server(input, output, session):
    # Store data as a result of reactive calculation
    @reactive.Calc
    def data():
        ticker = yf.Ticker(input.ticker())
        df = ticker.history(start=input.start_date(), end=input.end_date())
        return df.reset_index()
    
    
    
    @output
    @render.text
    def valuee():
        states = input.stock()
        if states:
            return [s for s in states]
        else:
            return ["No state selected"]
    


        
    # Chart logic
    @output
    @render.plot
    #@render.image
    def vizz():
        fig, ax = plt.subplots()
        ax.plot(data()["Date"], data()["Close"])
        #ax.scatter(data()["Date"], data()["Close"])
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f"{input.ticker()} historical Close Price")
        return fig
    
    @output
    @render.plot()
    def plot1():
        ticker = yf.Ticker(input.ticker())
        df = ticker.history(start=input.start_date(), end=input.end_date())
        
        #ticker = yf.Ticker(input.ticker())
        #data = yf.download(ticker, start=input.start_date(), end=input.end_date())
        #df = data()      
        # Plot portfolio returns using seaborn
        #fig = sns.lineplot(x="Date", y="Close", data=df)
        #df = df.reset_index()
        # Return the plot
        #return fig.get_figure() 
        p = (
            ggplot(df, aes(x='Date', y=df['Close'])) +
            geom_line()
            )
        return p
        

    # Store data as a result of reactive calculation
    @reactive.Calc
    def returns():
        start_date = input.start_date()
        end_date = input.end_date()
        #period_df = data()[(data()["Date"] >= start_date) & (data()["Date"] <= end_date)]
        returns_df = data()["Close"].pct_change().dropna()
        return returns_df
    
    @output
    @render.text
    def volatility_output():
        if input.start_date() is None or input.end_date() is None:
            return "Please select start and end dates"
        else:
            #volatility = returns().rolling(window=252).std() * (252 ** 0.5)
            volatility = np.sqrt(252) * returns().std()
            return  volatility#f"Volatility for {input.ticker()} from {input.start_date()} to {input.end_date()}: {volatility:.2%}"



    # Table logic
    @output
    @render.table
    @reactive.event(input.show_data, ignore_none=False)
    def table_data():
        return data()
    


    # Store data as a result of reactive calculation
    @reactive.Calc
    def corr_data():
        tickers_list = list(tickers.keys())
        df = pd.DataFrame(columns=tickers_list)

        for ticker in tickers_list:
            ticker_data = yf.Ticker(ticker).history(start=input.start_date(), end=input.end_date())['Close']
            df[ticker] = ticker_data
        corr = df.corr()
        return corr

    # Chart logic
    @output
    @render.plot
    def corr_heatmap():
        fig, ax = plt.subplots()
        mask = np.triu(np.ones_like(corr_data(), dtype=bool))
        sns.heatmap(corr_data(), mask = mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title('Correlation Heatmap')
        return fig
    @output
    @render.text
    def value():
        #def data():
        ticker = yf.Ticker(input.ticker())
        df = ticker.history(start="2000-01-01", end="2022-12-31")
        df.reset_index(inplace=True)
        from pytz import timezone
        # convert input date to datetime object in the same timezone as the 'Date' column
        date_str = input.sdate().strftime('%Y-%m-%d')
        date = timezone('America/New_York').localize(datetime.strptime(date_str, '%Y-%m-%d'))

        #date = timezone('America/New_York').localize(datetime.strptime(input.sdate(), '%Y-%m-%d'))
        #Find row corresponding to input date
        row = df.loc[df['Date'] == date]
        S = row.iloc[0]['Close']
        K = input.K()
        r = input.r()
        sigma = input.sigma()
        T = input.T()
        t = input.t()
        Ite = input.Ite()

        data = np.zeros((Ite, 2))
        z = np.random.normal(0, 1, [1, Ite])
        ST = S * np.exp((T - t) * (r - 0.5 * sigma**2) + sigma * np.sqrt(T - t) * z)
        data[:, 1] = ST - K
        average = np.sum(np.amax(data, axis=1)) / float(Ite)
        
        return f"The Call option price {np.exp(-r * (T - t)) * average}"
    @output
    @render.text
    def option():
        return value()
    
    @output
    @render.plot
    def plot():
        # Get selected stocks and date range from inputs
        sasn = input.sasn()
        gld = input.gld()
        start_date = input.start_date()
        end_date = input.end_date()
        tickers = [sasn, gld]
        
        
        if "SCOM" in tickers:
            tickers.append("SCOM")
        if "KQ" in tickers:
            tickers.append("KQ")
        if "KCB" in tickers:
            tickers.append("KCB")


        # Download stock data from Yahoo Finance
        data = yf.download(tickers, start = input.startdate(), end = input.enddate())["Close"]
        data = data.dropna()

        # Calculate portfolio returns
        initial_sasn = data[sasn][0] * 100000
        initial_gld = data[gld][0] * 100000
        initial_investment = initial_sasn + initial_gld
        weight_sasn = initial_sasn / (initial_sasn + initial_gld)
        weight_gld = 1 - weight_sasn
        returns = data.pct_change()
        returns["Portfolio"] = (returns[sasn] * weight_sasn) + (returns[gld] * weight_gld)
        returns = returns + 1
        returns.iloc[0] = initial_investment
        port_values = returns.cumprod()
        port_values["Date"] = port_values.index
    
        
        fig, ax = plt.subplots()
        for col in port_values.columns[:-1]:
            ax.plot(port_values["Date"], port_values[col], label=col)
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.legend()

        # Return the plot
        return fig
    @output
    @render.table
    def portfolio_std():
        # Get selected stocks and date range from inputs
        sasn = input.sasn()
        gld = input.gld()
        start_date = input.start_date()
        end_date = input.end_date()
        tickers = [sasn, gld]
        
        
        if "SCOM" in tickers:
            tickers.append("SCOM")
        if "KQ" in tickers:
            tickers.append("KQ")
        if "KCB" in tickers:
            tickers.append("KCB")


        # Download stock data from Yahoo Finance
        data = yf.download(tickers, start = input.startdate(), end = input.enddate())["Close"]
        data = data.dropna()

        # Calculate portfolio returns
        initial_sasn = data[sasn][0] * 100000
        initial_gld = data[gld][0] * 100000
        initial_investment = initial_sasn + initial_gld
        weight_sasn = initial_sasn / (initial_sasn + initial_gld)
        weight_gld = 1 - weight_sasn
        returns = data.pct_change()
        returns["Portfolio"] = (returns[sasn] * weight_sasn) + (returns[gld] * weight_gld)
        returns = returns + 1
        returns.iloc[0] = initial_investment
        port_values = returns.cumprod()
        port_values["Date"] = port_values.index
        
        returns.drop(index=returns.index[0], axis=0, inplace=True)
        returns = returns - 1
        data = returns.std().round(3).to_frame()
        data.reset_index(inplace=True)
        data.columns = ['Stock', 'Volatility']
        data = data.rename(columns={'Stock': 'Stock', 'Volatility': 'Volatility'})

        
        fig, ax = plt.subplots()
        for col in port_values.columns[:-1]:
            ax.plot(port_values["Date"], port_values[col], label=col)
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.legend()

        # Return the plot
        return data
    @output
    @render.plot()
    def plot2():
        ticker = input.tick()  # Retrieve the selected stock ticker
        if input.plot_variable() == "Volatility":
            # Retrieve historical stock price data
            start_date = "2010-01-01"  # Replace with your desired start date
            end_date = "2021-12-31"  # Replace with your desired end date
            data = yf.download(ticker, start=start_date, end=end_date)

            # Convert data to DataFrame
            df = pd.DataFrame(data)
            df["Log Returns"] = np.log(df["Close"]) - np.log(df.Close.shift(1))#.diff()

            monthly_volatility = df.groupby(pd.Grouper(freq="M"))["Log Returns"].std().reset_index()

            # Plot monthly volatility using plotnine
            plot = (
                ggplot(monthly_volatility, aes(x="Date", y="Log Returns")) +
                geom_line() +
                labs(title="Monthly Volatility of " + tickers[ticker], x="Date", y="Volatility") +
                theme(figure_size=(12, 6))
                )
            return plot
        elif input.plot_variable() == "Returns":
            # Retrieve historical stock price data
            start_date = "2010-01-01"  # Replace with your desired start date
            end_date = "2021-12-31"  # Replace with your desired end date

            data = yf.download(ticker, start=start_date, end=end_date)

            # Convert data to DataFrame
            df = pd.DataFrame(data)

            df["Log Returns"] = np.log(df["Close"]).diff()
            df.reset_index(inplace=True)

            # Plot monthly returns using plotnine
            p = (
                ggplot(df, aes(x="Date", y="Log Returns")) +
                geom_line() +
                labs(title="Monthly Returns of " + tickers[ticker], x="Date", y="Returns") +
                theme(figure_size=(12, 6))
                )
            return p
    @output
    @render.ui()
    def images() -> ui.Tag:
        img = ui.img(src="Profile Picture.jpg", style="width: 150px;")
        return img
    
    
    @session.download(
    filename=lambda: f"{input.ticker()}-{date.today().isoformat()}-{np.random.randint(100,999)}.csv"
        )
    async def downloadData():
        await asyncio.sleep(0.25)
        df = data()  # Get the data from the reactive calculation
        csv_content = df.to_csv(index=False)
        yield csv_content



        # Plot portfolio returns using seaborn
        #fig = sns.lineplot(x="Date", y="value", hue="variable", data=port_values.melt(id_vars=["Date"]))

        # Return the plot
        #return fig.get_figure()

# Connect everything
#app = App(app_ui, server)


www_dir = Path(__file__).resolve().parent
app = App(app_ui, server, static_assets=str(www_dir))

