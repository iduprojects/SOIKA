from transformers.pipelines import pipeline
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer, util

import pandas as pd
from pandas import DataFrame
import geopandas as gpd
from shapely.geometry import LineString
from shapely.wkt import loads

from itertools import chain, combinations
import re
import requests
import osm2geojson
import osmnx as ox

def read_data(filepath='messages.csv') -> gpd.GeoDataFrame:
  """
  Read the data from the csv file, add representative point geometry for texts without it and return a GeoDataframe.
  """
  df = pd.read_csv(filepath)
  messages_with_geometry = df[df['geometry'].notna()]
  messages_with_geometry.geometry = messages_with_geometry.geometry.map(loads)
  rep_point = gpd.GeoDataFrame(messages_with_geometry, geometry='geometry').set_crs(4326).geometry.unary_union.representative_point()
  messages_without_geometry = df[df['geometry'].isna()]
  messages_without_geometry['geometry'] = [rep_point] * len(messages_without_geometry)
  df = pd.concat([messages_with_geometry, messages_without_geometry])
  gdf = gpd.GeoDataFrame(df, geometry='geometry').set_crs(4326).drop(columns=['Unnamed: 0'])
  return gdf

def get_roads(city_name, city_crs) -> gpd.GeoDataFrame:
  """
  Get the road network of a city as road links and roads
  """
  links = ox.graph_from_place(city_name, network_type="drive")
  links = ox.utils_graph.graph_to_gdfs(links, nodes=False).to_crs(city_crs)
  links = links.reset_index(drop=True)
  links['link_id'] = links.index
  links["geometry"] = links["geometry"].buffer(7)
  links = links.to_crs(4326)
  links = links[['link_id', 'name', 'geometry']]
  links.loc[links['name'].map(type) == list, 'name'] = links[links['name'].map(type) == list]['name'].map(lambda x: ', '.join(x))
  road_id_name = dict(enumerate(links.name.dropna().unique().tolist()))
  road_name_id = {v: k for k, v in road_id_name.items()}
  links['road_id'] = links['name'].replace(road_name_id)
  return links

def get_buildings(links:gpd.GeoDataFrame, filepath='population.geojson') -> gpd.GeoDataFrame:
  """
  Get the buildings of a city as a GeoDataFrame
  """
  buildings = gpd.read_file(filepath)
  buildings = buildings[['address', 'building_id', 'population_balanced', 'geometry']]
  buildings = buildings.to_crs(epsg=4326)
  buildings['building_id'] = buildings.index
  buildings = gpd.sjoin_nearest(buildings, links[['link_id', 'road_id', 'geometry']], how='left', max_distance = 500)\
  .drop(columns=['index_right']).drop_duplicates(subset='building_id')
  return buildings

def collect_population(buildings:gpd.GeoDataFrame) -> dict:
  pops_global = {0:buildings.population_balanced.sum()}
  pops_buildings = buildings['population_balanced'].to_dict()
  pops_links = buildings[['population_balanced', 'link_id']].groupby('link_id').sum()['population_balanced'].to_dict()
  pops_roads = buildings[['population_balanced', 'road_id']].groupby('road_id').sum()['population_balanced'].to_dict()
  pops = {'global':pops_global, 'road':pops_roads, 'link':pops_links, 'building':pops_buildings}
  return pops

def preprocess(messages:gpd.GeoDataFrame, buildings:gpd.GeoDataFrame, links:gpd.GeoDataFrame, city_crs:int) ->  gpd.GeoDataFrame:
  messages = messages[['Текст комментария', 'geometry', 'Дата и время', 'message_id', 'cats']]
  messages = messages.sjoin(buildings, how='left')[['Текст комментария', 'address', 'geometry', 'building_id', 'message_id', 'Дата и время', 'cats']]
  messages.rename(columns = {'Текст комментария':'text', 'Дата и время':'date_time'}, inplace=True)
  messages = messages.sjoin(links, how='left')[['text', 'geometry', 'building_id', 'index_right', 'name', 'message_id', 'date_time', 'cats', 'road_id']]
  messages.rename(columns = {'index_right':'link_id', 'name':'road_name'}, inplace=True)
  messages = messages.join(buildings[['link_id', 'road_id']], on='building_id', rsuffix='_from_building')
  messages.loc[messages.link_id.isna(), 'link_id'] = messages.loc[messages.link_id.isna()]['link_id_from_building']
  messages.loc[messages.road_id.isna(), 'road_id'] = messages.loc[messages.road_id.isna()]['road_id_from_building']
  messages = messages[['message_id', 'text', 'geometry', 'building_id', 'link_id', 'road_id', 'date_time', 'cats']].dropna(subset='text')
  messages['global_id'] = 0
  return messages

def create_model(min_cluster_size):
  umap_model = UMAP(n_neighbors=15, n_components=5,
                  min_dist=0.0, metric='cosine', random_state=42)
  hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric='euclidean',
                          cluster_selection_method='eom', prediction_data=True)
  embedding_model = pipeline("feature-extraction", model="cointegrated/rubert-tiny2")
  topic_model = BERTopic(embedding_model=embedding_model, hdbscan_model=hdbscan_model, umap_model=umap_model,
                        calculate_probabilities=True, verbose=True, n_gram_range = (1,3))
  return topic_model

def event_from_object(messages: gpd.GeoDataFrame, topic_model, target_column:str, population:dict, object_id: float, event_level: str):
  local_messages = messages[messages[target_column] == object_id]
  message_ids = local_messages.message_id.tolist()
  docs = local_messages.text.tolist()
  if len(docs) >= 5:
    try:
      topics, probs = topic_model.fit_transform(docs)
    except TypeError:
      print("Can't reduce dimensionality or some other problem")
      return
    try:
        topics = topic_model.reduce_outliers(docs, topics)
        topic_model.update_topics(docs, topics=topics)
    except ValueError:
      print("Can't distribute all messages in topics")
    event_model = topic_model.get_topic_info()
    event_model['level'] = event_level
    event_model['object_id'] = str(object_id)
    event_model['id'] = event_model.Topic.astype(str) + '_' + event_model.level + '_' + event_model.object_id
    try:
      event_model['potential_population'] = population[event_level][object_id]
    except KeyError:
      event_model['potential_population'] = None
    clustered_messages = pd.DataFrame(data = {'id':message_ids, 'text':docs, 'topic_id':topics})
    cluster_messages = [clustered_messages[clustered_messages['topic_id'] == topic]['id'].tolist() for topic in event_model.Topic]
    event_model['message_ids'] = [clustered_messages[clustered_messages['topic_id'] == topic]['id'].tolist() for topic in event_model.Topic]
    event_model['duration'] = event_model.message_ids.map(lambda x: \
   (pd.to_datetime(messages[messages['message_id'].isin(x)].date_time).max() - pd.to_datetime(messages[messages['message_id'].isin(x)].date_time).min()).days)
    event_model['category'] = event_model.message_ids.map(lambda x: ', '.join(messages[messages['message_id'].isin(x)].cats.mode().tolist()))
    return event_model
  else:
    return

def get_events(messages:gpd.GeoDataFrame, buildings, min_event_size:int, levels:list[str]) -> gpd.GeoDataFrame:
  messages_list = messages.text.tolist()
  index_list = messages.message_id.tolist()
  pops = collect_population(buildings)
  topic_model = create_model(min_event_size)
  events = [[event_from_object(messages, topic_model, f'{level}_id', pops, oid, level) for oid in messages[f'{level}_id'].unique().tolist()] for level in reversed(levels)]
  events = [item for sublist in events for item in sublist if item is not None]
  events = pd.concat(list(chain(events)))
  events['geometry'] = events.message_ids.map(lambda x: messages[messages.message_id.isin(x)].geometry.unary_union.representative_point())
  events = gpd.GeoDataFrame(events, geometry='geometry').set_crs(4326)
  events.index = events.id
  levels_scale = dict(zip(levels, list(range(2, 10, 2))))
  events['level_scale'] = events['level'].replace(levels_scale)
  events.rename(columns={'Name':'name', 'Representative_Docs':'docs'}, inplace=True)
  events['docs'] = events['docs'].map(lambda x: ', '.join([str(index_list[messages_list.index(text)]) for text in x]))
  events.message_ids = events.message_ids.map(lambda x: ', '.join([str(id) for id in x]))
  events['risk'] = 1
  return events

def get_event_connections(events:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
  weights = [len((set(c[0]) & set(c[1]))) for c in combinations(events.message_ids, 2)]
  nodes = [c for c in combinations(events.id, 2)]
  connections = pd.DataFrame(nodes, weights).reset_index()
  connections.columns = ['weight', 'a', 'b']
  connections = connections[connections['weight'] > 0]
  connections = connections.join(events.geometry, on='a', rsuffix='_')
  connections = connections.join(events.geometry, on='b', rsuffix='_')
  events.reset_index(drop=True, inplace=True)
  connections['geometry'] = connections.apply(lambda x: LineString([x['geometry'], x['geometry_']]), axis=1)
  connections.drop(columns=['geometry_'], inplace=True)
  connections = gpd.GeoDataFrame(connections, geometry='geometry').set_crs(32636)
  return connections

def rebalance(connections:gpd.GeoDataFrame, events:gpd.GeoDataFrame, levels:list[str], event_population:int, event_id:str):
  """
  Rebalance the population of an event by connections with events of lower levels.
  """
  connections_of_event = connections[connections.a == event_id].b
  if len(connections_of_event) > 0:
    accounted_pops = events[events.id.isin(connections_of_event) & events.level.isin(levels)].potential_population.sum()
    if event_population >= accounted_pops:
      rebalanced_pops = event_population - accounted_pops
    else:
      connections_of_event = connections[connections.b == event_id].a
      accounted_pops = events[events.id.isin(connections_of_event) & events.level.isin(levels)].potential_population.sum()
      rebalanced_pops = event_population - accounted_pops
    return rebalanced_pops
  else:
    return event_population

def rebalance_events(events:gpd.GeoDataFrame, levels:list[str]) -> gpd.GeoDataFrame:
  events_rebalanced = []
  for level in levels[1:]:
    levels_to_account = levels[:levels.index(level)]
    events_for_level = events[events.level == level]
    events_for_level['rebalanced_population'] = events_for_level.apply(lambda x: rebalance(connections, events, levels_to_account, x.potential_population, x.id), axis=1)
    events_rebalanced.append(events_for_level)
  events_rebalanced = pd.concat(events_rebalanced)
  events_rebalanced.loc[events_rebalanced.rebalanced_population.isna(), 'rebalanced_population'] = 0
  events_rebalanced.rename(columns={'rebalanced_population':'population'}, inplace=True)
  events_rebalanced.drop(columns=['potential_population'], inplace=True)
  events_rebalanced.population = events_rebalanced.population.astype(int)
  return events_rebalanced

def filter_outliers(events:gpd.GeoDataFrame, connections:gpd.GeoDataFrame):
  pattern = r'^-1.*'
  print(len(events[events.Topic == -1]), 'outlier clusters of', len(events), 'total clusters. Filtering...')
  events = events[events.name.map(lambda x: False if re.match(pattern, x) else True)]
  connections = connections[connections.a.map(lambda x: False if re.match(pattern, x) else True)]
  connections = connections[connections.b.map(lambda x: False if re.match(pattern, x) else True)]
  return [events, connections]

def todo():
    messages = read_data(filepath='messages.csv')
    links = get_roads('Санкт-Петербург', 32636)
    buildings = get_buildings(links, filepath='population.geojson')
    messages = preprocess(messages, buildings, links, 32636)
    levels = ['building', 'link', 'road', 'global']
    levels_scale = dict(zip(levels, list(range(2, 10, 2))))
    events = get_events(messages, buildings, 3, levels)

    connections = get_event_connections(events)
    events = rebalance_events(events, levels)
    events['geometry'] = events.buffer(events.level_scale)
    events = events.to_crs(4326)
    events = events[['name', 'docs', 'level', 'id', 'population', 'message_ids', 'geometry']]
    events, connections = filter_oultiers(events, connections)

    messages = messages.reset_index(drop=True)
    messages.rename(columns={'cats':'block'}, inplace=True)
    messages = messages[['message_id', 'text', 'geometry', 'date_time', 'block']]
    messages = messages.to_crs(4326)

    events.to_file('test_events.geojson', driver='GeoJSON')
    connections.to_file('test_connections.geojson', driver='GeoJSON')
    messages.to_file('test_messages.geojson', driver='GeoJSON')