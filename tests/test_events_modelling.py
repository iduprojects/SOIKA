import pytest
import torch
import geopandas as gpd
import pandas as pd
from shapely import Point
from factfinder import EventDetection

path_to_file = "data/processed/messages.geojson"

@pytest.fixture
def gdf():
    df = pd.DataFrame([[0, 1, 2, 3, 4, 5],
 ['Здравствуйте! В Санкт-Петербурге нет Генерального консульства Австралии.  По консульским вопросам можно обратиться в Посольство Австралии в Москве по телефону: +7 495 956-60-70. 24-часовая консульская помощь для граждан Австралии по телефону: +61 2 6261 3305.  Ограниченную экстренную консульскую помощь при определенных обстоятельствах может оказать Почетное консульство Австралии в Санкт-Петербурге по адресу: улица Мойка, дом 11. Тел.: +7 964 333 7572.  Почетное консульство не отвечает на визовые запросы.  С уважением, Комитета по внешним связям Санкт-Петербурга.',
  '[club143265175|Центральный район Санкт-Петербурга], здравствуйте. У меня готово уточнение - на фасаде дома Рубинштейна 2/45 со стороны Невского пр. вырубили новый дверной проем в элементе ОКН , помещение 4-Н принадлежит СПБ , несколько раз обращались в ГЖИ, КГИОП, ЖК -дверь не заделали до сих пор. Невключение дома в текущий план по капремонту фасада относится к этому же адресу',
  '1) Фурштатская, 19 Отслоение штукатурного слоя и слабодержащиеся элементы, как вы это называете',
  '2) Фурштатская, 17 Здесь прямо-таки умоляю обратить внимание на зелёную сетку, провисающую от тяжести скопления осыпавшейся штукатурки. О состоянии фасада скромно промолчу',
  '3) Фурштатская, 13 Отслоение штукатурного слоя, разъединены элементы водостока',
  '4) Фурштатская, 11 Отслоение штукатурного слоя'],
 ['2023.01.26 16:32',
  '2023.01.26 11:55',
  '2023.01.28 12:39',
  '2023.01.28 12:42',
  '2023.01.28 12:45',
  '2023.01.28 12:46'],
 ['Безопасность', 'ЖКХ', 'ЖКХ', 'Благоустройство', 'ЖКХ', 'ЖКХ'],
 [Point (30.308, 59.932),
  Point (30.346, 59.932),
  Point (30.358, 59.945),
  Point (30.358, 59.945),
  Point (30.358, 59.945),
  Point (30.358, 59.945)]]).transpose()
    df.columns = ['message_id', 'Текст комментария', 'Дата и время', 'cats', 'geometry']
    gdf = gpd.GeoDataFrame(df, geometry = 'geometry').set_crs(4326)
    return gdf

@pytest.fixture
def expected_values():
    expected_name = "0_фурштатская_штукатурного слоя_слоя_отслоение"
    expected_risk = 0.405
    expected_messages = [4, 5, 3, 2]
    return expected_name, expected_risk, expected_messages
    
@pytest.fixture
def model():
    model = EventDetection(
    )
    return model

def test_event_detection(gdf, expected_name, expected_risk, expected_messages):
    event_model = EventDetection()
    messages, events, connections = model.run(
        gdf, 'data/raw/population.geojson', 'Санкт-Петербург', 32636, min_event_size=3
        )
    event_name = events.iloc[0]['name']
    event_risk = events.iloc[0]['risk'].round(3)
    event_messages = [int(mid) for mid in events.iloc[0]['message_ids'].split(', ')]
    assert event_name == expected_name
    assert event_risk == expected_risk
    assert all(mid in event_messages for mid in expected_messages)