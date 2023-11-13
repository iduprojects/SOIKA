"""
This module is aimed to provide necessary tools to find mentioned
location in the text. 
In this scenario texts are comments in social networks (e.g. Vkontakte).
Thus the model was trained on the corpus of comments on Russian language.
"""
import numpy as np
import re
import warnings
from typing import List, Optional

import flair
import geopandas as gpd
import networkx as nx
import osm2geojson
import osmnx as ox
import pandas as pd
import pymorphy2
import requests
import os
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from geopy.exc import GeocoderUnavailable
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from tqdm import tqdm
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    DatesExtractor,
    MoneyExtractor,
    AddrExtractor,
    Doc,
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
money_extractor = MoneyExtractor(morph_vocab)
addr_extractor = AddrExtractor(morph_vocab)


warnings.simplefilter(action="ignore", category=FutureWarning)

tqdm.pandas()


class Location:
    """
    This class is aimed to efficiently geocode addresses using Nominatim.
    Geocoded addresses are stored in the 'book' dictionary argument.
    Thus, if the address repeats -- it would be taken from the book.
    """

    max_tries = 3

    def __init__(self):
        self.geolocator = Nominatim(user_agent="soika")
        self.addr = []
        self.book = {}

    def geocode_with_retry(self, query: str) -> Optional[List[float]]:
        """
        Function to handle 403 error while geocoding using Nominatim.
        TODO: 1. Provide an option to use alternative geocoder
        TODO: 2. Wrap this function as a decorator
        """

        for _ in range(Location.max_tries):
            try:
                geocode = self.geolocator.geocode(
                    query, addressdetails=True, language="ru"
                )
                return geocode
            except GeocoderUnavailable:
                pass
        # If geocode still cannot be obtained after max_tries, return None
        return None

    def query(self, address: str) -> Optional[List[float]]:
        if address not in self.book:
            query = f"{address}"
            res = self.geocode_with_retry(query)
            self.book[address] = res

        return self.book.get(address)


class Streets:
    """
    This class encapsulates functionality for retrieving street data
    for a specified city from OSM and processing it to extract useful
    information for geocoding purposes.
    """

    global_crs: int = 4326

    @staticmethod
    def get_city_bounds(
        osm_city_name: str, osm_city_level: int
    ) -> gpd.GeoDataFrame:
        """
        Method retrieves the boundary of a specified city from OSM
        using Overpass API and returns a GeoDataFrame representing
        the boundary as a polygon.
        """
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
                area[name="{osm_city_name}"]->.searchArea;
                (
                relation["admin_level"="{osm_city_level}"](area.searchArea);
                );
        out geom;
        """

        result = requests.get(
            overpass_url, params={"data": overpass_query}
        ).json()
        resp = osm2geojson.json2geojson(result)
        city_bounds = gpd.GeoDataFrame.from_features(resp["features"]).set_crs(
            Streets.global_crs
        )
        return city_bounds

    @staticmethod
    def get_drive_graph(city_bounds: gpd.GeoDataFrame) -> nx.MultiDiGraph:
        """
        Method uses the OSMnx library to retrieve the street network for a
        specified city and returns it as a NetworkX MultiDiGraph object, where
        each edge represents a street segment and each node represents
        an intersection.
        """

        G_drive = ox.graph_from_polygon(
            city_bounds.dissolve()["geometry"].squeeze(), network_type="drive"
        )

        return G_drive

    @staticmethod
    def graph_to_gdf(G_drive: nx.MultiDiGraph) -> gpd.GeoDataFrame:
        """
        Method converts the street network from a NetworkX MultiDiGraph object
        to a GeoDataFrame representing the edges (streets) with columns
        for street name, length, and geometry.
        """

        gdf = ox.graph_to_gdfs(G_drive, nodes=False)
        gdf["name"].dropna(inplace=True)
        gdf = gdf[["name", "length", "geometry"]]
        gdf.reset_index(inplace=True)
        gdf = gpd.GeoDataFrame(data=gdf, geometry="geometry")

        return gdf

    @staticmethod
    def get_street_names(gdf: gpd.GeoDataFrame):
        """
        Method extracts the unique street names from a
        GeoDataFrame of street segments.
        """

        names = set(gdf["name"].explode().dropna())
        df_streets = pd.DataFrame(names, columns=["street"])

        return df_streets

    @staticmethod
    def drop_words_from_name(x: str) -> str:
        """
        This function drops parts of street names that are not the name
        of the street (e.g. avenue).
        """

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

    @staticmethod
    def clear_names(streets_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function pre-process the street names from the OSM.
        This step is necessary to match recognised street addresses later.
        We need to do this match because Nominatim is very sensitive geocoder
        and requires almost exact match between addresses in the OSM database
        and the geocoding address.
        """

        streets_df["street_name"] = streets_df["street"].progress_apply(
            lambda x: Streets.drop_words_from_name(x)
        )
        return streets_df

    @staticmethod
    def run(osm_city_name: str, osm_city_level: int) -> pd.DataFrame:
        city_bounds = Streets.get_city_bounds(osm_city_name, osm_city_level)
        streets_graph = Streets.get_drive_graph(city_bounds)
        streets_gdf = Streets.graph_to_gdf(streets_graph)
        streets_df = Streets.get_street_names(streets_gdf)
        streets_df = Streets.clear_names(streets_df)

        return streets_df


class Geocoder:
    """
    This class provides a functionality of simple geocoder
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))

    global_crs: int = 4326
    exceptions = pd.merge(
        pd.read_csv(
            os.path.join(dir_path, "exceptions_countries.csv"),
            encoding="utf-8",
            sep=",",
        ),
        pd.read_csv(
            os.path.join(dir_path, "exсeptions_city.csv"),
            encoding="utf-8",
            sep=",",
        ),
        on="Сокращенное наименование",
        how="outer",
    )

    def __init__(
        self,
        model_path: str = "Geor111y/flair-ner-addresses-extractor",
        device: str = "cpu",
        osm_city_level: int = 5,
        osm_city_name: str = "Санкт-Петербург",
    ):
        self.device = device
        flair.device = torch.device(device)
        self.classifier = SequenceTagger.load(model_path)
        self.osm_city_level = osm_city_level
        self.osm_city_name = osm_city_name

    def extract_ner_street(self, text: str) -> pd.Series:
        """
        Function calls the pre-trained custom NER model to extract mentioned
        addresses from the texts (usually comment) in social networks in
        russian language.
        The model scores 0.8 in F1 and other metrics.
        """

        try:
            text = re.sub(r"\[.*?\]", "", text)
        except Exception:
            return pd.Series([None, None])

        sentence = Sentence(text)
        self.classifier.predict(sentence)
        try:
            res = (
                sentence.get_labels("ner")[0]
                .labeled_identifier.split("]: ")[1]
                .split("/")[0]
                .replace('"', "")
            )
            score = round(sentence.get_labels("ner")[0].score, 3)
            if score > 0.7:
                return pd.Series([res, score])
            else:
                return pd.Series([None, None])

        except IndexError:
            return pd.Series([None, None])

    # Блок с Наташей
    def get_ner_address_natasha(
        row, exceptions, text_col
    ):  # input: string, list, series... output: string
        if row["Street"] == None or row["Street"] == np.nan:
            i = row[text_col]
            location_final = []
            i = re.sub(r"\[.*?\]", "", i)
            doc = Doc(i)
            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)
            doc.parse_syntax(syntax_parser)
            doc.tag_ner(ner_tagger)
            for span in doc.spans:
                span.normalize(morph_vocab)
            location = list(filter(lambda x: x.type == "LOC", doc.spans))
            for span in location:
                if (
                    span.normal.lower()
                    not in exceptions["Сокращенное наименование"]
                    .str.lower()
                    .values
                ):
                    location_final.append(span)
            location_final = [(span.text) for span in location_final]
            if not location_final:
                return None
            return location_final[0]
        else:
            return row["Street"]

    @staticmethod
    def get_stem(street_names_df: pd.DataFrame) -> pd.DataFrame:
        """
        Function finds the stem of the word to find this stem in the street
        names dictionary (df).
        """

        # initialize PyMorphy2 analyzer
        morph = pymorphy2.MorphAnalyzer()

        # define the cases
        cases = ["nomn", "gent", "datv", "accs", "ablt", "loct"]

        # add a column for each case with the respective form of the word
        for case in cases:
            street_names_df[case] = street_names_df[
                "street_name"
            ].progress_apply(
                lambda x: morph.parse(x)[0].inflect({case}).word
                if morph.parse(x)[0].inflect({case})
                else None
            )
        return street_names_df

    def find_word_form(
        self, df: pd.DataFrame, strts_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        In the russian language any word has different forms.
        Since addresses are extracted from the texts in social networks
        they might be in any possible form. This function is aimed to match that
        free form to the one that is used in the OSM database.

        Since the stem is found there would be several streets with that stem
        in their name.
        However the searching street name has its specific ending (form) and
        not each matched street name could have it.

        TODO: add spellcheker since there might be misspelled words.
        """

        df["full_street_name"] = None

        for idx, row in df.iterrows():
            search_val = row["Street"]
            val_num = row["Numbers"]

            for col in strts_df.columns[2:]:
                if search_val in strts_df[col].values:
                    streets_full = strts_df.loc[
                        strts_df[col] == search_val, "street"
                    ].values
                    streets_full = [
                        street
                        + f" {val_num}"
                        + f" {self.osm_city_name}"
                        + " Россия"
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
    def get_level(row: pd.Series) -> str:
        """
        Addresses in the messages are recognized on different scales:
        1. Where we know the street name and house number -- house level;
        2. Where we know only street name -- street level (with the centroid
        geometry of the street);
        3. Where we don't know any info but the city -- global level.
        """

        if (not pd.isna(row["Street"])) and (row["Numbers"] == ""):
            return "street"
        elif (not pd.isna(row["Street"])) and (row["Numbers"] != ""):
            return "house"
        else:
            return "global"

    def get_street(
        self, df: pd.DataFrame, text_column: str
    ) -> gpd.GeoDataFrame:
        """
        Function calls NER model and post-process result in order to extract
        the address mentioned in the text.
        """

        df[text_column].dropna(inplace=True)
        df[["Street", "Score"]] = df[text_column].progress_apply(
            lambda t: self.extract_ner_street(t)
        )
        df["Street"] = df[[text_column, "Street"]].progress_apply(
            lambda row: Geocoder.get_ner_address_natasha(
                row, self.exceptions, text_column
            ),
            axis=1,
        )
        df = df[df.Street.notna()]
        df = df[df["Street"].str.contains("[а-яА-Я]")]

        pattern1 = re.compile(r"(\D)(\d)(\D)")
        df["Street"] = df["Street"].apply(lambda x: pattern1.sub(r"\1 \2\3", x))

        pattern2 = re.compile(r"\d+")
        df["Numbers"] = df["Street"].apply(
            lambda x: " ".join(pattern2.findall(x))
        )
        df["Street"] = df["Street"].apply(lambda x: pattern2.sub("", x).strip())
        df["Street"] = df["Street"].str.lower()

        return df

    def create_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Function simply creates gdf from the recognised geocoded geometries.
        """

        df["Location"] = df["addr_to_geocode"].progress_apply(Location().query)
        df = df.dropna(subset=["Location"])
        df["geometry"] = df.Location.apply(
            lambda x: Point(x.longitude, x.latitude)
        )
        df["Location"] = df.Location.apply(lambda x: x.address)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=Geocoder.global_crs)

        return gdf

    def set_global_repr_point(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        This function set the centroid (actually, representative point) of the
        geocoded addresses to those texts that weren't geocoded (or didn't
        contain any addresses according to the trained NER model).
        """

        try:
            gdf.loc[gdf["level"] == "global", "geometry"] = gdf.loc[
                gdf["level"] != "global", "geometry"
            ].unary_union.representative_point()
        except AttributeError:
            pass

        return gdf

    def merge_to_initial_df(
        self, gdf: gpd.GeoDataFrame, initial_df: pd.DataFrame
    ) -> gpd.GeoDataFrame:
        """
        This function merges geocoded df to the initial df in order to keep
        all original attributes.
        """

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
        gdf = gpd.GeoDataFrame(
            gdf, geometry="geometry", crs=Geocoder.global_crs
        )

        return gdf

    def run(self, df: pd.DataFrame, text_column: str = "Текст комментария"):
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
