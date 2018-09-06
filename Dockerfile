# uses info from: https://github.com/jupyter/docker-stacks/blob/master/base-notebook/Dockerfile
FROM jupyter/scipy-notebook

ADD . /home/jovyan/binscatter
WORKDIR /home/jovyan/binscatter

USER root
RUN pip install .

EXPOSE 8888

ENV NAME World
USER $NB_UID
WORKDIR $HOME
CMD ["start.sh", "jupyter lab"]
