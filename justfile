default: 
  @just --list

build-container tag="latest":
  docker build -f docker/Dockerfile -t lumeny/vjepa:{{tag}} .

run-container:
  docker compose -f docker/docker-compose.yml up -d

enter-jepa:
  docker exec -it vjepa bash

test-image:
  docker run -it --rm lumeny/vjepa:latest bash


alias e := enter-jepa

alias b := build-container
