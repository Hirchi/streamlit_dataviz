import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image

uber_path = "uber-raw-data-apr14.csv"
ny_path = "ny-trips-data.csv"

image = Image.open('data:image/png;base64,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')
st.image(image, width=(500))
def myDecorator(function):
    def modified_function(df):
        time_ = time.time()
        res = function(df)
        time_ = time.time()-time_
        with open(f"{function.__name__}_exec_time.txt","w") as f:
            f.write(f"{time_}")
        return res
    return modified_function


@st.cache
def load_data(path):
    df = pd.read_csv(path)[:10000]
    return df


@myDecorator
@st.cache
def df1_data_transformation(df_):
    df = df_.copy()
    df["Date/Time"] = df["Date/Time"].map(pd.to_datetime)

    def get_dom(dt):
        return dt.day
    def get_weekday(dt):
        return dt.weekday()
    def get_hours(dt):
        return dt.hour

    df["weekday"] = df["Date/Time"].map(get_weekday)
    df["dom"] = df["Date/Time"].map(get_dom)
    df["hours"] = df["Date/Time"].map(get_hours)

    return df


@myDecorator
@st.cache
def df2_data_transformation(df_):
    df = df_.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    def get_hours(dt):
        return dt.hour

    df["hours_pickup"] = df["tpep_pickup_datetime"].map(get_hours)
    df["hours_dropoff"] = df["tpep_dropoff_datetime"].map(get_hours)

    return df

@st.cache(allow_output_mutation=True)
def frequency_by_dom(df):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_title("Frequency by DoM - Uber - April 2014")
    ax.set_xlabel("Date of the month")
    ax.set_ylabel("Frequency")
    ax = plt.hist(x=df.dom, bins=30, rwidth=0.8, range=(0.5,30.5))
    return fig

@st.cache
def map_data(df):
    df_ = df[["Lat","Lon"]]
    df_.columns=["lat","lon"]
    return df_

@st.cache(allow_output_mutation=True)
def data_by(by,df):
    def count_rows(rows):
        return len(rows)

    if by == "dom":
        fig, ax = plt.subplots(1,2, figsize=(10,6))
        ax[0].set_ylim(40.72,40.75)
        ax[0].bar(x=sorted(set(df["dom"])),height=df[["dom","Lat"]].groupby("dom").mean().values.flatten())
        ax[0].set_title("Latitude moyenne par jour du mois")

        ax[1].set_ylim(-73.96,-73.98)
        ax[1].bar(x=sorted(set(df["dom"])),height=df[["dom","Lon"]].groupby("dom").mean().values.flatten(), color="orange")
        ax[1].set_title("Longitude moyenne par jour du mois")
        return fig

    elif by == "hours":
        fig, ax= plt.subplots(figsize=(10,6))
        ax = plt.hist(x=df.hours, bins=24, range=(0.5,24))
        return fig

    elif by == "dow":
        fig, ax= plt.subplots(figsize=(10,6))
        ax = plt.hist(x=df.weekday, bins=7, range=(-5,6.5))
        return fig

    elif by == "dow_xticks":
        fig, ax= plt.subplots(figsize=(10,6))
        ax.set_xticklabels('Mon Tue Wed Thu Fri Sat Sun'.split())
        ax.set_xticks(np.arange(7))
        ax = plt.hist(x=df.weekday, bins=7, range=(0,6))
        return fig

    else:
        pass

@st.cache
def group_by_wd(df):
    def count_rows(rows):
        return len(rows)
    grp_df = df.groupby(["weekday","hours"]).apply(count_rows).unstack()
    return grp_df

@st.cache(allow_output_mutation=True)
def grp_heatmap(df):
    fig, ax= plt.subplots(figsize=(10,6))
    ax = sns.heatmap(grp_df)
    return fig

@st.cache(allow_output_mutation=True)
def lat_lon_hist(df,fusion=False):
    lat_range = (40.5,41)
    lon_range = (-74.2,-73.6)

    if fusion:
        fig, ax = plt.subplots()
        ax1 = ax.twiny()
        ax.hist(df.Lon, range=lon_range, color="yellow")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Frequency")

        ax1.hist(df.Lat, range=lat_range)
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Frequency")
        return fig

    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5))


        ax[0].hist(df.Lat, range=lat_range, color="red")
        ax[0].set_xlabel("Latitude")
        ax[0].set_ylabel("Frequence")

        ax[1].hist(df.Lon, range=lon_range, color="green")
        ax[1].set_xlabel("Longitude")
        ax[1].set_ylabel("Frequence")
        return fig

@st.cache(allow_output_mutation=True)
def display_points(data, color=None):
    fig, ax= plt.subplots(figsize=(10,6))
    ax = sns.scatterplot(data=data) if color == None else sns.scatterplot(data=data, color=color)
    return fig

@st.cache(allow_output_mutation=True)
def passengers_graphs_per_hour(df):
    fig, ax = plt.subplots(2,2, figsize=(10,6))

    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks(np.arange(24))

    ax[0,0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","passenger_count"]].groupby("hours_pickup").sum().values.flatten(), color="red")
    ax[0,0].set_title("Total Number of passengers per pickup hour")

    ax[0,1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","passenger_count"]].groupby("hours_pickup").mean().values.flatten(), color="yellow")
    ax[0,1].set_title("Average Number of passengers per pickup hour")

    ax[1,0].bar(x=sorted(set(df["hours_pickup"])), height=df["hours_pickup"].value_counts().sort_index().values.flatten(), color="green")
    ax[1,0].set_title("Total number of passages per pickup hour")
    return fig

@st.cache(allow_output_mutation=True)
def passengers_graphs_per_dropoff_hour(df):
    fig, ax = plt.subplots(2,2, figsize=(12,6))

    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks(np.arange(24))

    ax[0,0].bar(x=sorted(set(df["hours_dropoff"])), height=df[["hours_dropoff","passenger_count"]].groupby("hours_dropoff").sum().values.flatten())
    ax[0,0].set_title("Total Number of passengers per dropoff hour")

    ax[0,1].bar(x=sorted(set(df["hours_dropoff"])), height=df[["hours_dropoff","passenger_count"]].groupby("hours_dropoff").mean().values.flatten(), color="black")
    ax[0,1].set_title("Average Number of passengers per dropoff hour")

    ax[1,0].bar(x=sorted(set(df["hours_dropoff"])), height=df["hours_dropoff"].value_counts().sort_index().values.flatten(), color="orange")
    ax[1,0].set_title("Total number of passages per dropoff hour")
    return fig

@st.cache(allow_output_mutation=True)
def amount_graphs_per_hour(df):
    fig, ax = plt.subplots(1,2, figsize=(12,6))

    for ax_ in ax:
        ax_.set_xticks(np.arange(24))

    ax[0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","trip_distance"]].groupby("hours_pickup").sum().values.flatten(), color="grey")
    ax[0].set_title("Total trip distance per pickup hour")

    ax[1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","trip_distance"]].groupby("hours_pickup").mean().values.flatten())
    ax[1].set_title("Average trip distance per pickup hour")
    return fig

@st.cache(allow_output_mutation=True)
def distance_graphs_per_hour(df):
    fig, ax = plt.subplots(1,2, figsize=(10,6))

    for ax_ in ax:
        ax_.set_xticks(np.arange(24))

    ax[0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","total_amount"]].groupby("hours_pickup").sum().values.flatten(), color="lime")
    ax[0].set_title("Total amount per hour")

    ax[1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","total_amount"]].groupby("hours_pickup").mean().values.flatten(), color="pink")
    ax[1].set_title("Average amount per hour")
    return fig

@st.cache(allow_output_mutation=True)
def corr_heatmap(df):
    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.heatmap(df.corr())
    return fig
option = st.sidebar.selectbox(
    'Quelle dataset on choisi ?',
    ("Uber-raw-data-apr14 dataset","Ny-trips-data dataset", "doc"))
if option == "Uber-raw-data-apr14 dataset":
#Uber-raw-data-apr14 dataset
    st.title("Uber-raw-data-apr14 dataset")

## Load the Data
    st.text(" ")
    st.header("Load the Data")
    df1 = load_data(uber_path)
    if st.checkbox('Show dataframe'):
        df1


## Perform Data Transformation
    df1_ = df1_data_transformation(df1)

## Visual representation

#
    st.text(" ")
    st.text(" ")
    st.header("Visual representation")
    if st.checkbox("Show graphs"):
        st.text(" ")
        st.markdown("`Fréquence par jour du mois`")
        st.pyplot(frequency_by_dom(df1_))

    #
        st.text(" ")
        st.markdown("`Visualisation des points sur une carte`")
        st.map(map_data(df1_))

    #
        st.text(" ")
        st.markdown("`Latitude et longitude moyenne par jour du mois`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(data_by("dom",df1_))


    #
        st.text(" ")
        st.markdown("`Visualisation des données par heure`")
        st.pyplot(data_by("hours",df1_))

    #
        st.text(" ")
        st.markdown("`Visualisation des données par jour de la semaine`")
        st.pyplot(data_by("dow",df1_))

    #
        st.text(" ")
        st.markdown("`Visualisation des données par jour de la semaine avec les noms des jours en abcisse`")
        st.pyplot(data_by("dow_xticks",df1_))


        st.text(" ")
        st.text(" ")
## Performing Cross Analysis
    st.header("Performing Cross Analysis")

    if st.checkbox('Show cross analysis'):

    #
        grp_df = group_by_wd(df1_)


    #
        st.text(" ")
        st.markdown("`Histogramme de la latitude et de la longitude`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(lat_lon_hist(df1_))

    #
        st.text(" ")
        st.markdown("`Fusions des histogrammes de latitude et de longitude`")
        st.pyplot(lat_lon_hist(df1_, fusion=True))

    #
        st.text(" ")
        st.markdown("`affichage de la latitude des points sur un graphique`")
        st.pyplot(display_points(df1_.Lat))

    #
        st.text(" ")
        st.markdown("`affichage de la longitude des points sur un graphique`")
        st.pyplot(display_points(df1_.Lon, color="orange"))
        if st.sidebar.checkbox('show heatmap'):
            st.text(" ")
            st.markdown("`Carte de chaleur avec les données groupées`")
            st.pyplot(grp_heatmap(df1_))



elif option == "Ny-trips-data dataset":
#ny-trips-data dataset
    st.text(" ")
    st.text(" ")
    st.title("Ny-trips-data dataset")

## Load the Data
    st.text(" ")
    st.text(" ")
    st.header("Load the Data")
    st.text(" ")
    df2 = load_data(ny_path)
    if st.checkbox('Show dataframe 2'):
        df2


## Perform Data Transformation
    st.text(" ")
    df2_ = df2_data_transformation(df2)


## Visual representation

#
    st.text(" ")
    st.text(" ")
    st.header("Visual representation")
    if st.checkbox('Show graphs 2'):
        st.text(" ")
        st.text(" ")
        st.markdown("`Nombre total, moyen de passagers et nombre total de passages par heure de départ`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(passengers_graphs_per_hour(df2_))

    #
        st.text(" ")
        st.text(" ")
        st.markdown("`Nombre total, moyen de passagers et nombre total de passages par heure de d'arrivée`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(passengers_graphs_per_dropoff_hour(df2_))

    #
        st.text(" ")
        st.text(" ")
        st.markdown("`Montant total et montant moyen perçu en fonction de l'heure de départ`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(amount_graphs_per_hour(df2_))

    #
        st.text(" ")
        st.text(" ")
        st.markdown("`Distance totale parcourue et distance moyenne en fonction de l'heure de départ`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(distance_graphs_per_hour(df2_))




## Performing Cross Analysis

#
    st.text(" ")
    st.text(" ")
    st.header("Performing Cross Analysis")
    if st.checkbox('Show cross analysis 2'):
        if st.sidebar.checkbox('show la corrélation entre les différentes features'):
            st.text(" ")
            st.markdown("`Carte de chaleur permettant de visualiser la corrélation entre les différentes features`")
            st.pyplot(corr_heatmap(df2_.corr()))

        else:
            st.text(" ")
            st.markdown("`Carte de chaleur montrant la corrélation entre le nombre de passagers, la ditance totale, le montant du tarif, du pourboire et le montant total groupés par heure de départ`")
            grp = df2_[["passenger_count", "hours_pickup", "trip_distance", "fare_amount", "tip_amount", "total_amount"]].groupby("hours_pickup").sum()
            st.pyplot(corr_heatmap(grp.corr()))
else :
    st.components.v1.iframe("https://ip-api.com/", scrolling=True, height=400)

    st.components.v1.html("<body bgcolor='pink' style='display: flex; justify-content:center'><h1>My new component</h1></body>")

st.text(" ")
st.text(" ")
st.text(" ")
st.text(" ")
st.write("<h6 style='text-align: right; color: black; bottom:0px;'>by Abdelhadi Hirchi</h6>", unsafe_allow_html=True)
