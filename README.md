# SOIKA

![Your logo](https://i.ibb.co/qBjVx8N/soika.jpg)


[![Documentation Status](https://readthedocs.org/projects/soika/badge/?version=latest)](https://soika.readthedocs.io/en/latest/?badge=latest)
[![PythonVersion](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/scikit-learn/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## The purpose of the project

**SOIKA** is a library aimed at identifying points of public activity and related objects of the urban environment, as well as forecasting social risks associated with objects of public activity, based on the application of natural language analysis (NLP) methods to text messages of citizens in open communication platforms. 

It consists of several modules: classification, geolocation and risk detection.

**The structure of the modeling pipeline**

![Pypline](/docs/img/pipeline_en.png )

- In the classification module we propose cascade with two classification methods: 

  - First one is based on pre-trained spacy model (later will be replaced with BERT) and its goal is to determine main city function which is affected by a problem described in the complaint, for example, communal services or environmental protrection. 
  - Second one assigns complaint to one of pre-determined topic clusters. These clusters are generated with topic modelling algorithm for each city function and assessed by experts in these functions.

- In the geolocation module we propose ruBERT-based method of messages geolocation. It provides combination of pre-trained NER model to extract location (district, street and house number) from text and approximate string matching to assign coordinates from open data portal OpenStreetMaps to this location.

- In the risk detection module... (in development)

## Table of Contents

- [Core features](#soika-features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Documentation](#documentation)
- [Developing](#developing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contacts](#contacts)
- [Citation](#citation)


## SOIKA Features

- Ready-for-use tool for researchers or analytics who work with unstructured social data. Our library can assist with extracting facts from texts from social media without any additional software
- Module structure of the library allows to get and use only neccessary parts, for example, if your only goal is to produce geolocated messages about road accidents
- This library can be used for risk assesment and modelling based on data with functional, spatial and temporal dimensions

## Installation

All details about first steps with GEFEST might be found in the [install guide](https://soika.readthedocs.io/en/latest/soika/installation.html)
and in the [tutorial for novices](https://soika.readthedocs.io/en/latest/soika/quickstart.html)

## Project Structure

The latest stable release of SOIKA is on the [master branch](https://github.com/iduprojects/SOIKA/tree/master) 

The repository includes the following directories:

* Package [core](https://github.com/iduprojects/SOIKA/tree/master/factfinder)  contains the main classes and scripts. It is the *core* of SOIKA;
* Package [examples](https://github.com/iduprojects/SOIKA/tree/master/examples) includes several *how-to-use-cases* where you can start to discover how SOIKA works;
* All *unit and integration tests* can be observed in the [test]() directory;
* The sources of the documentation are in the [docs](https://github.com/iduprojects/SOIKA/tree/master/docs) 
                                                        
## Examples
You are free to use your own data, but it should match specification classes. Next examples will help to get used to the library:

1. [Classifier](examples/classifier_example.ipynb) - text
2. [Event detection](examples/event_detection_example.ipynb) -text
3. [Geocoder](examples/geocoder_example.ipynb) - text
4. [Topic classifier](examples/topic_classifier_example.ipynb) - text
5. [Pipeline example](examples/pipeline_example.ipynb) - text



## Documentation

We have a [documentation](https://soika.readthedocs.io/en/latest/?badge=latest), but our [examples](#examples) will explain the use cases cleaner.

## Developing

To start developing the library, one must perform following actions:

1. Clone repository (`git clone https://github.com/iduprojects/masterplanning`)
2. (optionally) create a virtual environment as the library demands exact packages versions: `python -m venv venv` and activate it.
3. Install the library in editable mode: `python -m pip install -e '.[dev]' --config-settings editable_mode=strict`
4. Install pre-commit hooks: `pre-commit install`
5. Create a new branch based on **develop**: `git checkout -b develop <new_branch_name>`
6. Add changes to the code
7. Make a commit, push the new branch and create a pull-request into **develop**

Editable installation allows to keep the number of re-installs to the minimum. A developer will need to repeat step 3 in case of adding new files to the library.

A more detailed guide to contributing is available in the [documentation](docs/source/contribution.rst).

## License

The project has [MIT License](./LICENSE)

## Acknowledgments

The library was developed as the main part of the ITMO University project #622264 **"Development of a service for identifying objects of the urban environment of public activity and high-risk situations on the basis of text messages of citizens"**


## Contacts

- [NCCR](https://actcognitive.org/o-tsentre/kontakty) - National Center for Cognitive Research
- [IDU](https://idu.itmo.ru/en/contacts/contacts.htm) - Institute of Design and Urban Studies
- If you have questions or suggestions, you can contact us at the following address: mvin@itmo.ru (Maxim Natykin)

## Citation

1. B. Nizomutdinov. The study of the possibilities of Telegram bots as a communication channel between authorities and citizens // Proceedings of the 2023 Communication Strategies in Digital Society Seminar (2023 ComSDS) ISBN 979-8-3503-2003-9/23/$31.00 ©2023 IEEE

2. A. Antonov, L. Vidiasova, A. Chugunov. Detecting public spaces and risk situations in them via social media data // Lecture Notes in Computer Science (LNCS), 2023, LNCS 14025, pp. 3–13, 2023. https://doi.org/10.1007/978-3-031-35915-6_1

3. Низомутдинов, Б. А. Тестирование методов обработки комментариев из Telegram-каналов и пабликов ВКонтакте для анализа социальных медиа / Б. А. Низомутдинов, О. Г. Филатова // International Journal of Open Information Technologies. – 2023. – Т. 11, № 5. – С. 137-145. – EDN RWNAOP.


