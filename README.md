<h1 align="center">Fake-News-Detection-for-Covid19-Heroku</h1>

<p align="center">
  <img alt="logo" src="https://miro.medium.com/max/1575/1*JpD6XXZheoFn3dliHnWEuw.jpeg">
</p>

[![online hosted application](https://img.shields.io/static/v1?label=Try-It-Out&message=Online-Hosted-Application&color=yellow&logo=godot-engine)](https://fake-news-covid19.herokuapp.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/release/shaunak09vb/Fake-News-Detection-for-Covid19-Heroku?include_prereleases&sort=semver)](https://github.com/shaunak09vb/Fake-News-Detection-for-Covid19-Heroku/releases/)
![Python](https://img.shields.io/badge/python-v3.8.3+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/shaunak09vb/Fake-News-Detection-for-Covid19-Heroku/issues)


## Introduction
Coronavirus (COVID-19) is an infectious disease that has resulted in an ongoing pandemic. The disease was first identified in Wuhan, China, and the first case was identified in December 2019. As of 21st August 2020, more than 22 million cases have been reported across 180 countries and territories. The sheer scale of this pandemic has led to myriad problems for the current generation. One of the acute problems that I have come across is the circulation of bogus news articles and in today’s world, spurious news articles can cause panic and mass hysteria. I realized the gravity of this problem and decided to base my next machine learning project on resolving this issue.

## Requirements

- Flask>=1.1.1
- gunicorn>=19.9.0
- itsdangerous>=1.1.0
- Jinja2>=2.10.1
- MarkupSafe>=1.1.1
- Werkzeug>=0.15.5
- numpy>=1.9.2
- scipy>=0.15.1
- scikit-learn>=0.18
- matplotlib>=1.4.3
- pandas>=0.19
- regex>=2020.6.8

## Installation

* Clone the repository 

`https://github.com/shaunak09vb/Fake-News-Detection-for-Covid19-Heroku.git`

* Install the required libraries

`pip3 install -r requirements.txt`

## Data Processing Steps

- Convert data to lower-case
- Remove punctuation
- Perform TF-IDF operation.

## Website

- To run the hosted website, access the `app.py` file and execute:

`python3 app.py`

- The website will start on your local server which can be viewed in your desired browser. You can paste any news article that you came across in the text box and determine if the news is Genuine or Fake.

- To understand the step-by-step approach undertaken for this project, you can view the Covid_19_FakeNewsClassifier.ipynb file via the following link: <a href='https://github.com/shaunak09vb/Fake-News-Detection-for-Covid19/blob/master/Covid_19_FakeNewsClassifier.ipynb'>Covid_19_FakeNewsClassifier.ipynb</a>.

## Blog Link

If you wish discover in detail, the steps taken by me for the implementation of the project. You can read my blog on <a href='https://shaunakvarudandi.medium.com/fake-news-classifier-to-tackle-covid-19-disinformation-ii-116ed2eb44e4'>Medium</a>.

## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under MIT License
