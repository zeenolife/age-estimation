# age-estimation
Train age-estimation model inside a docker container

## Data preparation
#### Please download the following datasets:

  - IMDB faces only dataset - [link](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar)
  - UTKFace in-the-wild dataset all 3 parts - [link](https://drive.google.com/drive/u/0/folders/0BxYys69jI14kSVdWWllDMWhnN2c)
  - APPA-REAL dataset - [link](http://158.109.8.102/AppaRealAge/appa-real-release.zip)
  - SoF dataset - [link](https://drive.google.com/file/d/0BwO0RMrZJCioR2FNQ3k1Z3FtODg/view?usp=sharing). Please extract the SoF dataset manually because it has spaces in its name.

#### The datasets directory should have the following structure:
    .
    ├── appa-real-release.zip   # APPA-REAL dataset
    ├── part1.tar.gz            # UTKFace part1
    ├── part2.tar.gz            # UTKFace part2
    ├── part3.tar.gz            # UTKFace part3
    ├── imdb_crop.tar           # IMDB dataset
    ├── original images.rar     # SoF dataset archived
    └── original images         # SoF dataset unarchived

## Docker
#### Build docker image
```sh
cd age-estimation
docker build -t age-estimation:v1
```

#### Run the docker image and link the datasets folder to `/age-estimation/data`
```sh
nvidia-docker run -p 8888:8888 -v /path/to/datasets_folder/:/age-estimation/data -d age-estimation:v1
```

This will automatically run Jupyter Notebook with 8888 port, so it can be accessed from the host. To get the token for jupyter notebook run the command  `docker ps` and find the container name and `docker logs container_name` afterwards.

## Training
The `train_by_parts.ipynb` does the data preparation and training.