# Welcome to Deep Learning in Neuroscience session for Lviv Data Science Summer School 2021!
Here you will find a collection of code and links for the [Deep Learning in Neuroscience](https://apps.ucu.edu.ua/en/summerschool-ds/course-list/deep-learning-in-neuroscience/) taught by [Edgar Y. Walker](https://edgarwalker.com), as part of the [Lviv Data Science Summer School 2021](https://apps.ucu.edu.ua/en/summerschool-ds/).


## Most importantly, you can launch the Jupyter notebook for the workshop in Colab by clicking on the badge below!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eywalker/LVIV-2021/blob/main/notebooks/DeepLearing%20in%20Neuroscience.ipynb)

## How to run on your own setup without Colab
If you prefer to run this on your own machine and without using Colab, you can do so by using this repository content.
Start by downloading/cloning this repository. 

### Using Docker
The repository is configured to work simply with `docker-compose`, so assuming you have [Dcoker](https://www.docker.com/) and `docker-compose` configured on your machine, all you'll have to do will be to navigate to the directory and run the following command:

```bash
$ docker-compose up
```

You can then navigate to http://localhost:8888 in your browser to access the Jupyter Lab server, properly configured.

### Running natively
If you prefer to not use Docker, then launch a Jupyter server and navigate to the Jupyter notebook `Deep-Learning-in-Neuroscience.ipynb` found in `notebooks` directory. You may have to adjust a few paths for your data download.

## Acknowledgements
Big thanks to Zhuokun Ding from Baylor College of Medicine and Suhas Shrinivasan from the University of Göttigen for their immense help in preparing and testing the course material!

