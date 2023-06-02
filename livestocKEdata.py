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
from shinywidgets import output_widget, register_widget,render_widget
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
import geopandas as gpd


# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Shiny App


# +
#pip install mplcursors


# +

df = pd.read_excel(r"2019 livestock population  census.xlsx",sheet_name = 'Table 34(e)')
df = df.drop(0)
df.columns = df.iloc[0]
df = df.reset_index(drop=True)
df = df.drop(0)
df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
df['SubCounty'] = df['SubCounty'].str.replace(' ', '')


data_long = (df
  >>gather('animal',"value",-_.SubCounty)
  >>mutate(SubCounty =_.SubCounty.str.replace(' ',''))
             
)
data_long['value'] = data_long['value'].astype(str).astype(int)
data_long['SubCounty'] = data_long['SubCounty'].str.capitalize()
county = ["Mombasa", "Kwale", "Kilifi", "Tanariver", "Lamu", "Taita/taveta", "Garissa", "Wajir", "Mandera", "Marsabit", "Isiolo",
                 "Meru", "Tharaka-nithi", "Embu", "Kitui", "Machakos", "Makueni", "Nyandarua", "Nyeri", "Kirinyaga", "Murang'a", "Kiambu",
                 "Turkana", "Westpokot", "Samburu", "Transnzoia", "Uasingishu", "Elgeyo/marakwet", "Nandi", "Baringo", "Laikipia", "Nakuru",
                 "Narok", "Kajiado", "Kericho", 
                 "Bomet", "Kakamega", "Vihiga", "Bungoma", "Busia", "Siaya", "Kisumu", "Homabay", "Migori", "Kisii", "Nyamira", "Nairobi"]
countyy = ["Kenya","Mombasa", "Kwale", "Kilifi", "Tanariver", "Lamu", "Taita/taveta", "Garissa", "Wajir", "Mandera", "Marsabit", "Isiolo",
                 "Meru", "Tharaka-nithi", "Embu", "Kitui", "Machakos", "Makueni", "Nyandarua", "Nyeri", "Kirinyaga", "Murang'a", "Kiambu",
                 "Turkana", "Westpokot", "Samburu", "Transnzoia", "Uasingishu", "Elgeyo/marakwet", "Nandi", "Baringo", "Laikipia", "Nakuru",
                 "Narok", "Kajiado", "Kericho", 
                 "Bomet", "Kakamega", "Vihiga", "Bungoma", "Busia", "Siaya", "Kisumu", "Homabay", "Migori", "Kisii", "Nyamira", "Nairobi"]
sub = data_long.SubCounty.unique().tolist()
animal = data_long.animal.unique().tolist()

cattle = str(['Exoticcattle-Dairy', 'Exoticcattle-Beef',
       'Indigenouscattle'])
chicken = str(['IndigenousChicken', 'ExoticChickenLayers',
               'ExoticChickenBroilers'])


mombasa = str(['Changamwe','Jomvu','Kisauni','Likoni','Mvita','Nyali'])
kwale = str(['Kinango','Lungalunga', 'Matuga','Msambweni'])
kilifi = str(['Chonyi', 'Ganze', 'Kaloleni', 'Kauma', 'Kilifinorth', 'Kilifisouth', 'Magarini', 'Malindi', 'Rabai'])
tanariver =str(['Tanadelta','Tananorth','Tanarivera'])
lamu = str(['Lamueast','Lamuwest',])
taitataveta = str(['Mwatate','Taita','Taveta','Voi'])
garissa = str(['Hulugho','Ijara','Lagdera'])
wajir = str(['Buna','Eldas','Habaswein','Tarbaj','Wajireast','Wajirnorth','Wajirsouth','Wajirwest'])
mandera = str(['Manderawest','Banisa','Kutulo','Lafey','Manderacentral','Manderaeast','Manderanorth'])
marsabit = str([ 'Loiyangalani','Marsabitcentral', 'Marsabitnorth','Marsabitsouth','Moyale','Northhorr','Sololo'])
isiolo = str(['Garbatulla','Isioloa','Merti'])
meru = str(['Buurieast','Buuriwest','Igembecentral', 'Igembenorth', 'Igembesouth', 'Imentinorth', 'Imentisouth', 'Merucentral', 'Tiganiacentral', 'Tiganiaeast', 'Tiganiawest', 'Merunationalpark', 'Mt.kenyaforest'])
tharakanithi = str(["Igambang'ombe", 'Maara', 'Merusouth', 'Tharakanorth', 'Tharakasouth'])
embu = str(['Embueast','Embunorth','Embuwest','Mbeeresouth','Mbeerenorth'])
kitui = str(['Ikutha','Katulani','Kisasi','Kituicentral','Kituiwest','Kyuso','Loweryatta','Matinyani','Migwani', 'Mumoni','Mutitu','Mutitunorth','Mutomo','Mwingicentral','Mwingieast','Nzambani','Thagicu','Tseikuru'])
machakos = str([ 'Athiriver', 'Kalama', 'Kangundo', 'Kathiani', 'Machakosa', 'Masinga', 'Matungulu', 'Mwala', 'Yatta'])
makueni = str(['Kathonzweni', 'Kibwezi', 'Kilungu', 'Makindu', 'Makuenia', 'Mboonieast', 'Mbooniwest', 'Mukaa', 'Nzaui'])
nyandarua = str(['Kinangop', 'Nyandaruasouth', 'Mirangine', 'Kipipiri', 'Nyandaruacentral', 'Nyandaruawest', 'Nyandaruanorth', 'Aberdarenationalpark'])
nyeri = str(['Tetu','Kienieast','Kieniwest','Mathiraeast','Mathirawest', 'Nyerisouth','Mukurwe-ini','Nyericentral','Aberdareforest'])
kirinyaga = str(['Kirinyagacentral', 'Kirinyagaeast', 'Kirinyagawest', 'Mweaeast', 'Mweawest'])
muranga = str(["Murang'aeast", 'Kangema', 'Mathioya', 'Kahuro', "Murang'asouth", 'Gatanga', 'Kigumo', 'Kandara'])
kiambu = str(['Gatundunorth', 'Gatundusouth', 'Githunguri', 'Juja', 'Kabete', 'Kiambaa', 'Kiambua', 'Kikuyu', 'Lari', 'Limuru', 'Ruiru', 'Thikaeast', 'Thikawest'])




navbar = ui.tags.nav(
    ui.h1("Kenya Counties :2019 Census Livestock Data"),
    ui.tags.style("h1{color: white;text-align: left; font-size:10}"),
    style=" height: 50px; width: 100%; display: flex; flex-direction: row; align-items: left; justify-content: left; text-align: left;"
)


app_ui = ui.page_fluid(
    shinyswatch.theme.superhero(),
    ui.tags.style(
            """
            #visuals { height: 900px !important; }
            #mapkenya { height: 900px !important; }
            #animaldist {width: 500px !important;}
            """
        ),
    x.ui.layout_sidebar(
        x.ui.sidebar(
            ui.input_selectize(
                "animall", "Livestock Type:For the Map",multiple=False,choices = animal
            ),
            ui.hr(),
            ui.input_selectize(
                "counties", "County", countyy,multiple=False
            ),
            ui.hr(),
            ui.input_selectize(
                "x","Cattle & Chicken Categories",{cattle:"Cattle",chicken:"Chicken"},multiple = False,
                ),
            ui.input_selectize(
                "county", "County",multiple=True,choices = county
            ),
            ui.hr(),
            ui.input_switch(
                "somevalue", "Show Data", False
            ),
        ),
        navbar,
        ui.navset_tab_card(
            ui.nav("Data",
                   ui.panel_main(
                       ui.row(
                           ui.column(
                               4,
                               ui.panel_well(
                                   ui.input_selectize(
                                       "countiess", "County", {mombasa:"Mombasa",kwale:"Kwale",kilifi:"Kilifi",tanariver:"Tanariver",lamu:"Lamu", taitataveta :"Taita/taveta", garissa:"Garissa", wajir :"Wajir",mandera: "Mandera", marsabit:"Marsabit",isiolo:"Isiolo",meru:"Meru", tharakanithi:"Tharaka-nithi",embu:"Embu", kitui :"Kitui",machakos:"Machakos",makueni:"Makueni",nyandarua:"Nyandarua", nyeri:"Nyeri", kirinyaga:"Kirinyaga",muranga: "muranga", kiambu:"Kiambu"},multiple = False
                                       )
                                   ),
                               ),
                           ui.column(
                               4,
                               ui.panel_well(
                                   ui.input_selectize(
                                       "animal", "Livestock",data_long.animal.unique().tolist() ,multiple = False
                                       )
                                   ),
                               ),
                           
                           ),
                       
                       ui.output_table("table_data", fill=True),
                       output_widget("piechat"),
                      ),
                   ), 
            ui.nav("Visuals",
                   ui.panel_main(   
                       output_widget("piechart"),
                       x.ui.output_plot("animal_dist", fill=True),
                       ),
                  ),
            ui.nav("Map",
                   ui.panel_main(
                       ui.output_plot("mapkenya"),
                       ),
                   ),
            ),
    ),
)

def server(input, output, session):
    @reactive.Calc
    def data():
        df = pd.read_excel(r"2019 livestock population  census.xlsx",sheet_name = 'Table 34(e)')
        df = df.drop(0)
        df.columns = df.iloc[0]
        df = df.reset_index(drop=True)
        df = df.drop(0)
        df_t = df.T
        df_t.columns = df_t.iloc[0]
        #df = df_t.iloc[1:].reset_index(drop=True)
        df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
        df['Cattle'] = df['Exoticcattle-Dairy'] + df['Exoticcattle-Beef'] + df['Indigenouscattle']
        df['SubCounty'] = df['SubCounty'].str.replace(' ', '')
        df['SubCounty'] = df['SubCounty'].str.capitalize()
        return df   

    @output
    @render.plot()
    def mapkenya():
        data_long = (df
                     >>gather('animal',"value",-_.SubCounty)
                     >>mutate(SubCounty =_.SubCounty.str.replace(' ',''))
                     )
        data_long['value'] = data_long['value'].astype(str).astype(int)
        data_long['SubCounty'] = data_long['SubCounty'].str.capitalize()
        listt = ["Mombasa", "Kwale", "Kilifi", "Tanariver", "Lamu", "Taita/taveta", "Garissa", "Wajir", "Mandera", "Marsabit", "Isiolo",
                 "Meru", "Tharaka-nithi", "Embu", "Kitui", "Machakos", "Makueni", "Nyandarua", "Nyeri", "Kirinyaga", "Murang'a", "Kiambu",
                 "Turkana", "Westpokot", "Samburu", "Transnzoia", "Uasingishu", "Elgeyo/marakwet", "Nandi", "Baringo", "Laikipia", "Nakuru",
                 "Narok", "Kajiado", "Kericho", 
                 "Bomet", "Kakamega", "Vihiga", "Bungoma", "Busia", "Siaya", "Kisumu", "Homabay", "Migori", "Kisii", "Nyamira", "Nairobi"]
        #capital_list = [county.upper() for county in listt]
        kenya_shapefile = r'County.shp'
        kenya_data = gpd.read_file(kenya_shapefile)
        kenya_data = kenya_data[['COUNTY', 'geometry']]
        hh = (
            data_long
            >> filter(_.SubCounty.isin(listt))
            >> filter(_.animal.isin([input.animall()]))
            )
       
        
        #Startsfromhere
        kenya_data= (kenya_data
                     #>>gather('animal',"value",-_.SubCounty)
                     >>mutate(COUNTY =_.COUNTY.str.replace(' ',''))
                     )
        kenya_data['COUNTY'] = kenya_data['COUNTY'].replace('Keiyo-Marakwet', 'Elgeyo/marakwet')
        kenya_data['COUNTY'] = kenya_data['COUNTY'].replace('Tharaka', 'Tharaka-nithi')
        kenya_data['COUNTY'] = kenya_data['COUNTY'].str.replace(' ', '')
        kenya_data['COUNTY'] = kenya_data['COUNTY'].str.capitalize()
        sh_list=kenya_data['COUNTY'].unique().tolist()

        county_order = [county.capitalize() for county in sh_list]
        # Create a new column for sorting
        hh['order'] = pd.Categorical(hh['SubCounty'], county_order)

        # Sort the DataFrame based on the new column
        hh = hh.sort_values('order')

        # Remove the 'order' column
        hh = hh.drop('order', axis=1)
        

        # Print the reordered DataFrame
        kenya_data = kenya_data.copy()
        kenya_data['value'] = hh['value'].values
        animal = input.animall()
        fig, ax = plt.subplots(figsize=(20,15))
        #linewidth=0.8, edgecolor='0.8', legend=False,
        kenya_data.plot(column='value',linewidth=0.8, edgecolor='0.8', legend=True,cmap='Reds',ax=ax)#.set_axis_off()
        # Add county names
        for x, y, label, value in zip(kenya_data.geometry.centroid.x, kenya_data.geometry.centroid.y, kenya_data['COUNTY'], kenya_data['value']): 
            plt.text(x, y, f'{label}', fontsize=5, ha='center', va='center')
        plt.axis('off')
        # Add figure title
        plt.title(f"{animal} Distribution in Kenya")
    
            
        return fig#kenya_data.plot(column='value', cmap='YlOrRd', legend=True, figsize=(20, 20)).set_axis_off()
    
    @output
    @render.plot
    def animal_dist(): 
        animals = eval(input.x())
        indx_c = data_long["SubCounty"].isin(input.county())
        indx_a = data_long["animal"].isin(animals)

        # subset data to keep only selected Counties and breeds
        sub_df = data_long[indx_c & indx_a]
        sub_df["dummy"] = 1
        fig = (
            sub_df
            >> ggplot(aes(x='SubCounty', y='value', fill='animal')) + geom_bar(stat='identity', width=0.7) +
            scale_fill_manual(values=['#FFC000', '#5B9BD5', '#ED7D31']) +
            labs(x='SubCounty', y='Livestock Count', fill='Livestock') +
            theme(axis_text_x=element_text(rotation=45, ha='right'))
            )
        return fig
    
    
    @reactive.Calc
    def dataa():
        df = pd.read_excel(r"2019 livestock population  census.xlsx",sheet_name = 'Table 34(e)')
        df = df.drop(0)
        df.columns = df.iloc[0]
        df = df.reset_index(drop=True)
        df = df.drop(0)
        df_t = df.T
        df_t.columns = df_t.iloc[0]
        #df = df_t.iloc[1:].reset_index(drop=True)
        df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
        df['Cattle'] = df['Exoticcattle-Dairy'] + df['Exoticcattle-Beef'] + df['Indigenouscattle']
        df['SubCounty'] = df['SubCounty'].str.replace(' ', '')
        df['SubCounty'] = df['SubCounty'].str.capitalize()
        scounties = eval(input.countiess())
        indx_a = df['SubCounty'].isin(scounties)
        df = df[indx_a]
        return df 
    
    
    
    @output
    @render.table()
    def table_data():
        return dataa()
    @reactive.Effect
    def generate_pie_chart():
        subcounty = input.counties()
        
        # Retrieve the livestock data for the specified subcounty
        livestock = data().loc[data()['SubCounty'] == subcounty].squeeze()
        labels = list(livestock.index[1:])
        values = list(livestock.values[1:])

        # Create the pie chart using Plotly
        pie_chart = go.FigureWidget(
            go.Figure(
                data=[go.Pie(labels=labels, values=values, textinfo='label+percent')],
                layout={"title": f"Livestock Distribution in {subcounty}"}#'width':500, 'height':600}
                )
            )

        # Register the pie chart widget
        register_widget("piechart", pie_chart)
    @reactive.Effect
    def generate_livestock_pie_chart():       
        livestock_type = input.animal()
        livestock = dataa()[livestock_type].tolist()
        labels = dataa()['SubCounty']
        values = livestock
        
        pie_chat = go.FigureWidget(
            go.Figure(
                data=[go.Pie(labels=labels, values=values, textinfo='label+percent')],
                layout={"title": f"Distribution of {livestock_type} across Subcounties"}
                )
            )

        register_widget("piechat", pie_chat)
        
app = App(app_ui, server)

# -







