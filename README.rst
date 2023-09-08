SOIKA
==============

.. |eng| image:: https://img.shields.io/badge/lang-en-red.svg
   :target: /README.rst

.. |rus| image:: https://img.shields.io/badge/lang-ru-yellow.svg
   :target: /README_ru.rst

.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://github.com/Text-Analytics/SOIKA/blob/master/LICENSE.md
    :alt: License

.. |style| image:: https://github.com/Text-Analytics/SOIKA/actions/workflows/checks.yml/badge.svg
    :target: https://github.com/Text-Analytics/SOIKA/actions/workflows/checks.yml
    :alt: Style checks

.. start-badges
.. list-table::
   :stub-columns: 1

   * - tests
     - | |style| 
   * - license
     - | |license|
   * - languages
     - | |eng| |rus|
.. end-badges

**SOIKA** is a library aimed at identifying points of public activity and related objects of the urban environment, as well as forecasting social risks associated with objects of public activity, based on the application of natural language analysis (NLP) methods to text messages of citizens in open communication platforms. 

It consists of several modules: classification, geolocation and risk detection.

.. image:: /docs/img/pipeline_en.png
   :alt: The structure of the modeling pipeline

In the classification module we propose cascade with two classification methods: 

- First one is based on pre-trained spacy model (later will be replaced with BERT) and its goal is to determine main city function which is affected by a problem described in the complaint, for example, communal services or environmental protrection. 
- Second one assigns complaint to one of pre-determined topic clusters. These clusters are generated with topic modelling algorithm for each city function and assessed by experts in these functions.

In the geolocation module we propose ruBERT-based method of messages geolocation. It provides combination of pre-trained NER model to extract location (district, street and house number) from text and approximate string matching to assign coordinates from open data portal OpenStreetMaps to this location.

In the risk detection module... (in development)


SOIKA Features
==============

- Ready-for-use tool for researchers or analytics who work with unstructured social data. Our library can assist with extracting facts from texts from social media without any additional software
- Module structure of the library allows to get and use only neccessary parts, for example, if your only goal is to produce geolocated messages about road accidents
- This library can be used for risk assesment and modelling based on data with functional, spatial and temporal dimensions

Contribution Guide
==================

The contribution guide is available in this `repository <https://github.com/Text-Analytics/SOIKA/blob/master/CONTRIBUTING.md>`__.

Acknowledgment
==============

The project was carried out as part of the research work of masters and postgraduate students.

Contacts
==============
If you have questions or suggestions, you can contact us at the following address: mvin@itmo.ru (Maxim Natykin)

