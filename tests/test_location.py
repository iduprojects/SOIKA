import pytest

from factfinder.src.geocoder import Location


@pytest.mark.parametrize(
    "input_address,geocode_result",
    [
        (
            "Санкт-Петербург, Дворцовая Набережная, 38",
            "Зимний дворец, 38 литА, Дворцовая набережная, Дворцовый округ, "
            "Санкт-Петербург, Северо-Западный федеральный округ, 191186, "
            "Россия",
        ),
        (
            "Санкт-Петербург ул. Итальянская, 17",
            "Yadro, 17, Итальянская улица, Дворцовый округ, Санкт-Петербург, "
            "Северо-Западный федеральный округ, 191168, Россия",
        ),
    ],
)
def test_geocode_with_retry(input_address, geocode_result):
    result = Location().geocode_with_retry(input_address)
    assert result.address == geocode_result


def test_geocode_with_retry_empty_address():
    result = Location().geocode_with_retry("")
    assert result is None


@pytest.mark.parametrize(
    "input_address,geocode_result",
    [
        (
            "Санкт-Петербург, Дворцовая Набережная, 38",
            "Зимний дворец, 38 литА, Дворцовая набережная, Дворцовый округ, "
            "Санкт-Петербург, Северо-Западный федеральный округ, 191186, "
            "Россия",
        ),
        (
            "Санкт-Петербург ул. Итальянская, 17",
            "Yadro, 17, Итальянская улица, Дворцовый округ, Санкт-Петербург, "
            "Северо-Западный федеральный округ, 191168, Россия",
        ),
    ],
)
def test_query(input_address, geocode_result):
    result = Location().query(input_address)
    assert result.address == geocode_result
