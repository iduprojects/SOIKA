from flair.models import SequenceTagger
from flair.data import Sentence
import flair, torch
import pandas as pd
import requests
import osm2geojson
import osmnx as ox
import re

import geopandas as gpd
from shapely.geometry import Point
import pymorphy2
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from tqdm import tqdm

tqdm.pandas()


class Location:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="soika")
        self.addr = []
        self.book = {}

    def geocode_with_retry(self, query):
        max_tries = 3
        for _ in range(max_tries):
            try:
                geocode = self.geolocator.geocode(
                    query, addressdetails=True, language="ru"
                )
                return geocode
            except GeocoderUnavailable:
                pass
        # If geocode still cannot be obtained after max_tries, return None
        return None

    def query(self, row):
        address = row["addr_to_geocode"]
        if address in self.book.keys():
            location = self.book[address]
        else:
            query = f"{address}"
            res = self.geocode_with_retry(query)

            if res:
                location = res
            else:
                location = None

            self.book[address] = location

        return location


class Streets:
    def get_city_bounds(osm_city_name, osm_city_level):
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
                area[name="{osm_city_name}"]->.searchArea;
                (
                relation["admin_level"="{osm_city_level}"](area.searchArea);
                );
        out geom;
        """
        result = requests.get(overpass_url, params={"data": overpass_query}).json()
        resp = osm2geojson.json2geojson(result)
        city_bounds = gpd.GeoDataFrame.from_features(resp["features"]).set_crs(4326)
        return city_bounds

    def get_drive_graph(city_bounds):
        G_drive = ox.graph_from_polygon(
            city_bounds.dissolve()["geometry"].squeeze(), network_type="drive"
        )
        return G_drive

    def graph_to_gdf(G_drive):
        gdf = ox.graph_to_gdfs(G_drive, nodes=False)
        gdf["name"].dropna(inplace=True)
        gdf = gdf[["name", "length", "geometry"]]
        gdf.reset_index(inplace=True)
        gdf = gpd.GeoDataFrame(data=gdf, geometry="geometry")
        return gdf

    def get_street_names(gdf):
        names = set(gdf["name"].explode().dropna())
        df_streets = pd.DataFrame(names, columns=["street"])

        return df_streets

    @staticmethod
    def drop_words_from_name(x):
        try:
            lst = re.split(
                r"путепровод|улица|набережная реки|проспект"
                r"|бульвар|мост|переулок|площадь|переулок"
                r"|набережная|канала|канал|дорога на|дорога в"
                r"|шоссе|аллея|проезд",
                x,
            )
            lst.remove("")

            return lst[0].strip().lower()

        except ValueError:
            return x

    def clear_names(streets_df):
        streets_df["street_name"] = streets_df["street"].progress_apply(
            lambda x: Streets.drop_words_from_name(x)
        )
        return streets_df

    def run(osm_city_name, osm_city_level):
        city_bounds = Streets.get_city_bounds(osm_city_name, osm_city_level)
        streets_graph = Streets.get_drive_graph(city_bounds)
        streets_gdf = Streets.graph_to_gdf(streets_graph)
        streets_df = Streets.get_street_names(streets_gdf)
        streets_df = Streets.clear_names(streets_df)

        return streets_df


class Geocoder:
    def __init__(
        self,
        model_path="Geor111y/flair-ner-addresses-extractor",
        device="cpu",
        osm_city_level=5,
        osm_city_name="Санкт-Петербург",
    ):
        self.device = device
        flair.device = torch.device(device)
        self.classifier = SequenceTagger.load(model_path)
        self.osm_city_level = osm_city_level
        self.osm_city_name = osm_city_name

    def extract_ner_street(self, t):
        try:
            t = re.sub(r"\[.*?\]", "", t)
        except Exception:
            return pd.Series([None, None])

        sentence = Sentence(t)
        self.classifier.predict(sentence)
        try:
            res = (
                sentence.get_labels("ner")[0]
                .labeled_identifier.split("]: ")[1]
                .split("/")[0]
                .replace('"', "")
            )
            score = round(sentence.get_labels("ner")[0].score, 3)
            return pd.Series([res, score])

        except IndexError:
            return pd.Series([None, None])

    @staticmethod
    def get_stem(strts):
        # initialize PyMorphy2 analyzer
        morph = pymorphy2.MorphAnalyzer()

        # define the cases
        cases = ["nomn", "gent", "datv", "accs", "ablt", "loct"]

        # add a column for each case with the respective form of the word
        for case in cases:
            strts[case] = strts["street_name"].progress_apply(
                lambda x: morph.parse(x)[0].inflect({case}).word
                if morph.parse(x)[0].inflect({case})
                else None
            )
        return strts

    def find_word_form(self, df, strts):
        df["full_street_name"] = None

        for idx, row in df.iterrows():
            search_val = row["Street"]
            val_num = row["Numbers"]

            for col in strts.columns[2:]:
                if search_val in strts[col].values:
                    streets_full = strts.loc[strts[col] == search_val, "street"].values
                    streets_full = [
                        street + f" {val_num}" + f" {self.osm_city_name}" + " Россия"
                        for street in streets_full
                    ]

                    df.loc[idx, "full_street_name"] = ",".join(streets_full)

        df.dropna(subset="full_street_name", inplace=True)
        df["location_options"] = df["full_street_name"].str.split(",")

        new_df = df["location_options"].explode()
        new_df.name = "addr_to_geocode"
        df = df.merge(new_df, left_on=df.index, right_on=new_df.index)

        df["location_options"] = df["location_options"].astype(str)

        return df

    @staticmethod
    def get_level(row):
        if (not pd.isna(row["Street"])) and (row["Numbers"] == ""):
            return "street"
        elif (not pd.isna(row["Street"])) and (row["Numbers"] != ""):
            return "house"
        else:
            return "global"

    def get_street(self, df, text_column):
        df[text_column].dropna(inplace=True)
        df[["Street", "Score"]] = df[text_column].progress_apply(
            lambda t: self.extract_ner_street(t)
        )
        df = df[df.Street.notna()]
        df = df[df["Street"].str.contains("[а-яА-Я]")]

        pattern1 = re.compile(r"(\D)(\d)(\D)")
        df["Street"] = df["Street"].apply(lambda x: pattern1.sub(r"\1 \2\3", x))

        pattern2 = re.compile(r"\d+")
        df["Numbers"] = df["Street"].apply(lambda x: " ".join(pattern2.findall(x)))
        df["Street"] = df["Street"].apply(lambda x: pattern2.sub("", x).strip())
        df["Street"] = df["Street"].str.lower()

        return df

    def create_gdf(self, df):
        df["Location"] = df.progress_apply(Location().query, axis=1)
        df = df.dropna(subset=["Location"])
        df["geometry"] = df.Location.apply(lambda x: Point(x.longitude, x.latitude))
        df["Location"] = df.Location.apply(lambda x: x.address)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=4326)

        return gdf
    
    def set_global_repr_point(self, gdf):
        try:
            gdf.loc[gdf["level"] == "global", "geometry"] = gdf.loc[
                gdf["level"] != "global", "geometry"
            ].unary_union.representative_point()
        except AttributeError:
            pass

        return gdf
    
    def merge_to_initial_df(self, gdf, initial_df):
        initial_df.reset_index(drop=False, inplace=True)

        gdf = initial_df.merge(
            gdf[
                [
                    "key_0",
                    "Street",
                    "Numbers",
                    "Score",
                    "location_options",
                    "Location",
                    "geometry",
                ]
            ],
            left_on="index",
            right_on="key_0",
            how="outer",
        )

        gdf.drop(columns=["key_0"], inplace=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=4326)

        return gdf


    def run(self, df, text_column="Текст комментария"):
        initial_df = df.copy()
        street_names = Streets.run(self.osm_city_name, self.osm_city_level)

        df = self.get_street(df, text_column)
        street_names = self.get_stem(street_names)
        df = self.find_word_form(df, street_names)
        gdf = self.create_gdf(df)
        gdf = self.merge_to_initial_df(gdf, initial_df)
        
        # Add a new 'level' column using the get_level function
        gdf["level"] = gdf.apply(self.get_level, axis=1)
        gdf = self.set_global_repr_point(gdf)

        return gdf
